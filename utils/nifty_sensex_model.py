"""
utils/nifty_sensex_model.py
────────────────────────────
Core ML Ensemble Engine for Nifty & Sensex Intraday Options Predictor.

Architecture:
  Layer 1 — Base Models (RF + HGB + optional LGBM + optional XGBoost)
  Layer 2 — FrozenEstimator Sigmoid Calibration (with Brier guardrail)
  Layer 3 — Regime-Gated Conviction Thresholding
  Layer 4 — VWAP Gate check (called from nifty_trade_signal.py)

Fix #2: Uses only features from nifty_features.ALL_ML_FEATURES (Groups A+C+D+E).
         No Group B features. Lookahead-safe by construction (all shifted T-1).
Fix #3: Group B OI/PCR is injected AFTER this model runs (in nifty_trade_signal.py).
"""

import os
import sys
import logging
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, precision_score, brier_score_loss

# Optional boosting libraries — graceful degradation
try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    from sklearn.frozen import FrozenEstimator
    HAS_FROZEN = True
except ImportError:
    HAS_FROZEN = False

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.nifty_features import build_feature_matrix, ALL_ML_FEATURES
from utils.nifty_regime import classify_regime, get_conviction_threshold
from config import (
    NIFTY_TICKER, SENSEX_TICKER,
    ML_N_ESTIMATORS, ML_MAX_DEPTH, ML_MIN_SAMPLES_LEAF,
    ML_TRAIN_SPLIT, ML_CAL_SPLIT, ML_WARMUP_DAYS,
    NIFTY_MIN_CONVICTION_WEEKDAY, NIFTY_MIN_CONVICTION_FRIDAY,
    NIFTY_MIN_CONVICTION_EXPIRY_DAY, NIFTY_MIN_CONVICTION_CHOP,
    NIFTY_MIN_CONVICTION_VOLATILE,
)

logger = logging.getLogger("nifty_sensex_model")


# ─────────────────────────────────────────────────────────────────────────────
# BSM Option Price Estimator (for backtest P&L approximation)
# ─────────────────────────────────────────────────────────────────────────────

def _bsm_price(spot: float, strike: float, T: float, sigma: float,
               r: float = 0.065, is_call: bool = True) -> float:
    """Black-Scholes-Merton option price. T in years."""
    from math import log, sqrt, exp
    from scipy.stats import norm
    if T <= 0 or sigma <= 0:
        return max(0.01, abs(spot - strike))
    try:
        d1 = (log(spot / strike) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt(T))
        d2 = d1 - sigma * sqrt(T)
        if is_call:
            return spot * norm.cdf(d1) - strike * exp(-r * T) * norm.cdf(d2)
        else:
            return strike * exp(-r * T) * norm.cdf(-d2) - spot * norm.cdf(-d1)
    except Exception:
        return max(0.5, abs(spot - strike) * 0.5)


def estimate_atm_premium(spot: float, days_to_expiry: int,
                         atr: float, strike_step: int = 50,
                         is_call: bool = True) -> float:
    """Estimate ATM option premium using BSM with ATR-derived IV."""
    atm_strike = round(spot / strike_step) * strike_step
    T = max(days_to_expiry, 1) / 252.0
    sigma = max(atr / spot / np.sqrt(T), 0.08) if spot > 0 else 0.18  # ATR-implied IV
    sigma = min(sigma, 0.80)  # Cap at 80%
    return round(_bsm_price(spot, atm_strike, T, sigma, is_call=is_call), 2)


# ─────────────────────────────────────────────────────────────────────────────
# Model Training
# ─────────────────────────────────────────────────────────────────────────────

def _build_base_models():
    """Return list of (name, estimator) base classifiers."""
    models = [
        ("RF",  RandomForestClassifier(
            n_estimators=ML_N_ESTIMATORS, max_depth=ML_MAX_DEPTH,
            min_samples_leaf=ML_MIN_SAMPLES_LEAF, random_state=42, n_jobs=-1,
        )),
        ("HGB", HistGradientBoostingClassifier(
            max_iter=150, max_depth=4, min_samples_leaf=ML_MIN_SAMPLES_LEAF,
            random_state=42,
        )),
    ]
    if HAS_LGBM:
        models.append(("LGBM", LGBMClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            num_leaves=31, min_child_samples=ML_MIN_SAMPLES_LEAF,
            random_state=42, n_jobs=-1, verbosity=-1,
        )))
    if HAS_XGB:
        models.append(("XGB", XGBClassifier(
            n_estimators=150, max_depth=4, learning_rate=0.05,
            min_child_weight=ML_MIN_SAMPLES_LEAF, random_state=42,
            n_jobs=-1, eval_metric="logloss", verbosity=0,
        )))
    return models


def _train_ensemble(X_train, y_train, X_cal, y_cal, X_test, y_test,
                    sample_weights):
    """
    Train all base models with exponential recency weighting.
    Apply FrozenEstimator sigmoid calibration with Brier guardrail.
    Returns: (ensemble_prob_fn, test_metrics_dict)
    """
    base_models   = _build_base_models()
    trained        = []
    cal_models     = []
    use_cal_flags  = []

    for name, mdl in base_models:
        try:
            mdl.fit(X_train, y_train, sample_weight=sample_weights)
            trained.append((name, mdl))

            # Calibration (only if FrozenEstimator available)
            if HAS_FROZEN:
                from sklearn.frozen import FrozenEstimator
                cal = CalibratedClassifierCV(FrozenEstimator(mdl), method="sigmoid")
                cal.fit(X_cal, y_cal)
                raw_brier = brier_score_loss(y_test, mdl.predict_proba(X_test)[:, 1])
                cal_brier = brier_score_loss(y_test, cal.predict_proba(X_test)[:, 1])
                use_cal   = cal_brier < raw_brier
                cal_models.append((name, cal, use_cal))
                use_cal_flags.append(use_cal)
                logger.debug(f"{name}: raw_brier={raw_brier:.4f}, cal_brier={cal_brier:.4f}, use_cal={use_cal}")
            else:
                cal_models.append((name, mdl, False))
                use_cal_flags.append(False)
        except Exception as e:
            logger.warning(f"Model {name} training failed: {e}")

    if not trained:
        raise RuntimeError("All base models failed to train.")

    n = len(trained)
    weights = np.ones(n) / n  # Equal weights; could be tuned by test Brier

    def ensemble_prob(X) -> float:
        probs = []
        for (name, mdl), (_, cal_mdl, use_cal) in zip(trained, cal_models):
            estimator = cal_mdl if use_cal else mdl
            p = estimator.predict_proba(X)[:, 1]
            probs.append(float(p[0]))
        return float(np.dot(weights[:len(probs)], probs[:len(weights)]))

    # Test metrics (on held-out test set)
    test_probs = []
    for (name, mdl), (_, cal_mdl, use_cal) in zip(trained, cal_models):
        estimator = cal_mdl if use_cal else mdl
        test_probs.append(estimator.predict_proba(X_test)[:, 1])
    avg_test_prob = np.mean(test_probs, axis=0)
    test_preds    = (avg_test_prob >= 0.50).astype(int)
    test_metrics  = {
        "accuracy":   round(float(accuracy_score(y_test, test_preds)) * 100, 1),
        "precision":  round(float(precision_score(y_test, test_preds, zero_division=0)) * 100, 1),
        "brier":      round(float(brier_score_loss(y_test, avg_test_prob)), 4),
        "n_models":   len(trained),
        "calibrated": any(use_cal_flags),
        "model_names": [m[0] for m in trained],
    }

    # Feature importance (from RF)
    try:
        rf_model = next(m for n, m in trained if n == "RF")
        test_metrics["feature_importances"] = dict(
            sorted(zip(X_train.columns, rf_model.feature_importances_),
                   key=lambda x: x[1], reverse=True)
        )
    except Exception:
        test_metrics["feature_importances"] = {}

    return ensemble_prob, test_metrics


# ─────────────────────────────────────────────────────────────────────────────
# Master Train & Predict Function
# ─────────────────────────────────────────────────────────────────────────────

def train_and_predict(
    ticker: str = NIFTY_TICKER,
    period: str = "3y",
) -> dict:
    """
    Build feature matrix, train ensemble, and generate today's prediction.

    Fix #2: Only ALL_ML_FEATURES (Groups A+C+D+E) are used — no lookahead.
    Fix #3: Group B (OI/PCR) not used here. Applied as modifier in trade signal.

    Returns dict: direction, conviction_pct, confidence, regime, test_metrics,
                  latest_close, atr_14, atm_strike, days_to_expiry, raw_prob_up.
    """
    try:
        # ── Build feature matrix ──────────────────────────────────────────────
        df = build_feature_matrix(ticker=ticker, period=period)
        clean = df.dropna(subset=ALL_ML_FEATURES + ['Target_Intraday']).copy()

        if len(clean) < ML_WARMUP_DAYS + 20:
            return {
                "status": "error",
                "message": f"Only {len(clean)} clean rows — need {ML_WARMUP_DAYS + 20}+",
            }

        X_all = clean[ALL_ML_FEATURES]
        y_all = clean['Target_Intraday'].astype(int)

        n = len(X_all)
        t_idx  = int(n * ML_TRAIN_SPLIT)
        c_idx  = int(n * ML_CAL_SPLIT)

        X_train, y_train = X_all.iloc[:t_idx],  y_all.iloc[:t_idx]
        X_cal,   y_cal   = X_all.iloc[t_idx:c_idx], y_all.iloc[t_idx:c_idx]
        X_test,  y_test  = X_all.iloc[c_idx:],  y_all.iloc[c_idx:]

        if len(X_cal) < 20 or len(X_test) < 10:
            return {"status": "error", "message": "Too few calibration/test samples."}

        # Exponential recency weighting on train set
        sw = np.exp(np.linspace(-0.5, 0.0, len(X_train)))

        ensemble_fn, metrics = _train_ensemble(
            X_train, y_train, X_cal, y_cal, X_test, y_test, sw
        )

        # ── Get latest row for live prediction ───────────────────────────────
        # Latest row = last row of df (NOT clean — we want most recent date even if
        # some features are NaN, which we fill with group medians)
        latest = df.iloc[-1].copy()
        feat_dict = {}
        for col in ALL_ML_FEATURES:
            val = latest.get(col, np.nan)
            if pd.isna(val):
                val = float(clean[col].median())
            feat_dict[col] = val
        X_live = pd.DataFrame([feat_dict])[ALL_ML_FEATURES]

        raw_prob_up = float(np.clip(ensemble_fn(X_live), 0.05, 0.95))
        direction   = "UP" if raw_prob_up >= 0.50 else "DOWN"
        conviction  = max(raw_prob_up, 1.0 - raw_prob_up) * 100.0

        # ── Regime classification ─────────────────────────────────────────────
        regime, regime_score, regime_detail = classify_regime(df, lookback=20)

        # ── Confidence label ──────────────────────────────────────────────────
        if conviction >= 75.0:
            confidence = "Super High"
        elif conviction >= 68.0:
            confidence = "High"
        elif conviction >= 62.0:
            confidence = "Moderate"
        else:
            confidence = "Low / Neutral"

        # ── Latest market data for trade signal ───────────────────────────────
        latest_close = float(latest.get("Close", 0.0))
        atr_14       = float(latest.get("ATR_14", latest_close * 0.015 or 1.0))

        # Expiry info (from E features in latest row)
        days_to_nifty  = int(latest.get("E_Days_Nifty_Expiry",  7))
        days_to_sensex = int(latest.get("E_Days_Sensex_Expiry", 7))
        is_expiry_day  = bool(latest.get("E_Is_Expiry_Day", 0.0))
        is_friday      = bool(latest.get("E_Is_Friday", 0.0))

        # Regime-aware conviction threshold
        required_threshold = get_conviction_threshold(
            regime=regime,
            is_friday=is_friday,
            is_expiry_day=is_expiry_day,
            weekday_threshold=NIFTY_MIN_CONVICTION_WEEKDAY,
            friday_threshold=NIFTY_MIN_CONVICTION_FRIDAY,
            expiry_threshold=NIFTY_MIN_CONVICTION_EXPIRY_DAY,
            chop_threshold=NIFTY_MIN_CONVICTION_CHOP,
            volatile_threshold=NIFTY_MIN_CONVICTION_VOLATILE,
        )
        is_tradeable = conviction >= required_threshold

        return {
            "status":             "success",
            "ticker":             ticker,
            "direction":          direction,
            "raw_prob_up":        round(raw_prob_up * 100, 2),
            "conviction_pct":     round(conviction, 2),
            "confidence":         confidence,
            "is_tradeable":       is_tradeable,
            "required_threshold": round(required_threshold, 1),
            "regime":             regime,
            "regime_score":       round(regime_score, 4),
            "regime_detail":      regime_detail,
            "latest_close":       round(latest_close, 2),
            "atr_14":             round(atr_14, 2),
            "days_to_nifty_expiry":  days_to_nifty,
            "days_to_sensex_expiry": days_to_sensex,
            "is_expiry_day":      is_expiry_day,
            "is_friday":          is_friday,
            "test_metrics":       metrics,
            "feature_importances": metrics.get("feature_importances", {}),
        }

    except Exception as e:
        logger.exception(f"train_and_predict failed for {ticker}: {e}")
        return {"status": "error", "message": str(e)}
