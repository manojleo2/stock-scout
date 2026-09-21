# Future Roadmap: 45-Minute Micro-Swing & Scalping Predictor

## Status: Proposed / Backlog Feature for StockScout v4+
- **Date Recorded:** September 21, 2026
- **Feature Title:** Sub-Hourly Intraday Horizon Predictor (45-Minute Horizon)
- **Target Assets:** CDSL.NS, HDFCBANK.NS, NIFTY 50

---

## 1. Executive Summary & Purpose

### The Problem in v1–v3:
Current StockScout predictors operate on **macro sessions**:
* **AI Directional Predictor:** Full trading day (Open $\rightarrow$ 3:30 PM Close).
* **3:05 PM Gap Predictor:** Overnight session (3:30 PM Close $\rightarrow$ 9:15 AM Open).

Neither model can answer the short-term scalper/day-trader question:
> *"It is 10:15 AM. CDSL is at ₹1,347. Will the price go UP or DOWN by 11:00 AM (the next 45 minutes)?"*

### The Proposed Solution:
A specialized **45-Minute Micro-Swing Engine** that runs on **5-minute intraday bars** to predict price trajectory over a rolling 9-candle horizon ($9 \times 5\text{ min} = 45\text{ min}$).

---

## 2. Mathematical Specification

### A. Target Variable Definition
* Let $t$ be the current 5-minute candle.
* The forward horizon is $t + 9$ (45 minutes into the future).
$$\text{Target}_t = \begin{cases} 1 & \text{if } \text{Close}_{t+9} > \text{Close}_t \times (1 + \text{Friction}) \\ 0 & \text{otherwise} \end{cases}$$
*(Where Friction $\approx 0.05\%$ accounts for broker/STT execution slippage).*

### B. Microstructure & Technical Features (5-Minute Candles)
1. **Intraday VWAP Distance:**  
   $$\Delta\text{VWAP}_t = \frac{\text{Close}_t - \text{VWAP}_t}{\text{VWAP}_t}$$
   *(Identifies mean-reversion pullbacks vs momentum breakouts).*
2. **5-Minute EMA Velocity:**  
   $$\text{EMA\_Vel}_t = \frac{\text{EMA}_{9}(t) - \text{EMA}_{21}(t)}{\text{Close}_t}$$
3. **Volume Surge Multiplier:**  
   $$\text{Vol\_Surge}_t = \frac{\text{Volume}_t}{\text{SMA}_{20}(\text{Volume})}$$
4. **5-Minute SuperTrend Signal:** Trend direction switch on intraday ATR multiplier 2.0.
5. **Session Time Normalized ($0.0 \rightarrow 1.0$):**  
   Minutes elapsed since 9:15 AM divided by 375 minutes (captures opening volatility vs afternoon lull).
6. **Benchmark Index Lead-Lag (Nifty 5-min correlation):**  
   Real-time 5-minute directional momentum of Nifty 50 / Bank Nifty.

---

## 3. UI / UX Design & Workflow

```
+-------------------------------------------------------------------------+
|  ⏱️ 45-MINUTE INTRADAY SCALP PREDICTOR                                  |
|  Current Time: 10:15 AM IST | LTP: ₹1,347.20                             |
+-------------------------------------------------------------------------+
|  [⚡ Predict Next 45-Minute Window (10:15 AM -> 11:00 AM)]              |
+-------------------------------------------------------------------------+
|  Probability: 🟢 67.4% UP (Bullish Micro-Swing)                        |
|  Expected 45-Min Range: ₹1,345.00 - ₹1,354.50                           |
|  Target (TP): ₹1,353.00 (+0.43%)                                        |
|  Stop Loss (SL): ₹1,343.80 (-0.25%) | R:R = 1.72 : 1                    |
|  Invalidation Condition: 5-min close below VWAP (₹1,344.10)             |
+-------------------------------------------------------------------------+
```

---

## 4. Technical Requirements for Implementation
* **Data Feed:** 5-minute intraday bars via `yfinance` (`interval="5m", period="60d"`) or direct Broker API WebSocket (Zerodha Kite / Angel One / Upstox).
* **Execution Model:** LightGBM / XGBoost classifier optimized for sub-hourly tabular data.
* **Latency Requirement:** Feature computation + inference < 1.5 seconds.
* **Storage Audit:** `logs/audit_45min_predictions.json` to track predicted 45m direction vs actual realized price at $T+45\text{m}$.
