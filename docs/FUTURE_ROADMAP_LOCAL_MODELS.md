# Future Roadmap: Local Compute & Cloud Sync Engine (Hybrid Architecture)

## Status: Proposed / Backlog Architecture for StockScout v4+
- **Date Recorded:** September 21, 2026
- **Feature Title:** Local-to-Cloud Hybrid Quant Engine (Heavy Local Training + Zero-Latency Cloud Dashboard)

---

## 1. Concept & Objective
To completely decouple heavy machine learning computation (model training, hyperparameter tuning, web scraping, multi-factor analysis) from the resource-constrained 1-vCPU Streamlit Cloud container.

* **Heavy Compute (Local PC):** Runs locally on user's multi-core PC via 1-click desktop shortcut or automated 9:00 AM / 3:05 PM Windows Task Scheduler.
* **Sync Channel:** Synchronizes predictions, ATR trade blueprints, and probability signals to GitHub repository or a cloud key-value database (Supabase / Firebase / Cloudflare KV).
* **Display Layer (Streamlit Cloud):** Reads pre-computed, verified signals and displays them instantly (< 0.02s) on web/mobile with 0% CPU strain.

---

## 2. Technical Architecture Specifications

### A. Local Execution Script (`scripts/run_local_predictors.py`)
1. Fetches real-time market data, tick volumes, and financial news via residential IP (no cloud rate-limits).
2. Fits RandomForest + HistGradientBoosting + Specialist models across CDSL and HDFC Bank.
3. Formulates locked execution blueprint (Entry Zone, Target 1, Target 2, Stop Loss).
4. Serializes output to `data/latest_prediction_state.json`.

### B. Cloud Sync Channels
* **Channel 1 (Git Push Automation):**
  Script executes:
  `git add data/latest_prediction_state.json`
  `git commit -m "sync: update live market predictions [skip ci]"`
  `git push origin master`
* **Channel 2 (Instant Cloud Database - Zero Git Overhead):**
  Script executes HTTP POST request to Supabase / Firebase REST endpoint. Update latency: < 500ms.

### C. Web Dashboard (`views/ml_prediction.py` & `views/opening_prediction.py`)
* Web pages load `latest_prediction_state.json` or query Supabase.
* Model training code in web app is bypassed completely.
* Page load time: **< 0.05 seconds**.
* Server CPU usage: **0.0%**.

---

## 3. Benefits Summary
1. Eliminates cloud container RAM limits (enables 1000-tree ensembles, XGBoost, and deep sequence models).
2. Immune to Yahoo Finance / NSE cloud IP throttling.
3. Guarantees 100% stable, unflickering probabilities throughout the trading session.
