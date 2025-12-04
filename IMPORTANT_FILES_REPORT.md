# 📊 IMPORTANT FILES - FULL FUNCTIONAL & FLOW REPORT

> **Document Author:** Antigravity AI Assistant  
> **Date:** December 4, 2025  
> **Purpose:** Complete technical explanation of two core production scripts

---

# Table of Contents

1. [rebalanced_predictive_model.py](#1-rebalanced_predictive_modelpy)
   - [Overview](#11-overview)
   - [Architecture Diagram](#12-architecture-diagram)
   - [Complete Execution Flow](#13-complete-execution-flow)
   - [Key Components Deep Dive](#14-key-components-deep-dive)
   - [Data Flow Diagram](#15-data-flow-diagram)
   
2. [portfolio_backtest_optimized_v2.py](#2-portfolio_backtest_optimized_v2py)
   - [Overview](#21-overview)
   - [Architecture Diagram](#22-architecture-diagram)
   - [Complete Execution Flow](#23-complete-execution-flow)
   - [Key Components Deep Dive](#24-key-components-deep-dive)
   - [Comparison with Production Script](#25-comparison-with-production-script)

3. [Technical Glossary](#3-technical-glossary)

---

# 1. rebalanced_predictive_model.py

## 1.1 Overview

| Property | Value |
|----------|-------|
| **Lines of Code** | 2,088 |
| **File Size** | 97.5 KB |
| **Purpose** | Live portfolio management with ML predictions |
| **Key Models** | TCN (generalized_lstm_model_improved.keras) |
| **Data Source** | Upstox API |
| **Database** | portfolio_analysis.db |
| **Rebalancing** | Every 60 trading days |

### What It Does (High Level)
This is the **production script** for live portfolio management. It:
1. Manages a 5-stock portfolio with HRP-optimized weights
2. Implements 15% stop-loss protection
3. Predicts 60-day future returns using a TCN model
4. Uses RAG (Retrieval-Augmented Generation) for risk assessment
5. Generates comprehensive performance reports
6. Saves predictions to SQLite database

---

## 1.2 Architecture Diagram

```
┌────────────────────────────────────────────────────────────────────────────────┐
│                     rebalanced_predictive_model.py                              │
├────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐      │
│  │  Upstox API │───▶│  Data Fetch │───▶│  Technical  │───▶│    HRP      │      │
│  │  (Historical)│    │  (5 Stocks) │    │  Indicators │    │ Optimization│      │
│  └─────────────┘    └─────────────┘    └─────────────┘    └──────┬──────┘      │
│                                                                   │             │
│                            ┌──────────────────────────────────────┘             │
│                            ▼                                                    │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐                         │
│  │ TCN Model   │◀───│  Portfolio  │───▶│  Stop-Loss  │                         │
│  │ (Prediction)│    │  Simulation │    │  Engine     │                         │
│  └──────┬──────┘    └──────┬──────┘    └─────────────┘                         │
│         │                  │                                                    │
│         ▼                  ▼                                                    │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐                         │
│  │ RAG System  │    │ Performance │    │  Database   │                         │
│  │ (ChromaDB)  │    │   Report    │    │   (SQLite)  │                         │
│  └─────────────┘    └──────┬──────┘    └─────────────┘                         │
│                            │                                                    │
│                            ▼                                                    │
│                     ┌─────────────┐                                             │
│                     │  Forecast   │                                             │
│                     │    Plot     │                                             │
│                     └─────────────┘                                             │
│                                                                                  │
└────────────────────────────────────────────────────────────────────────────────┘
```

---

## 1.3 Complete Execution Flow

### 🔷 PHASE 1: Initialization (Lines 1-46)
```
START
  │
  ├── Import Libraries (pandas, numpy, tensorflow, sqlite3, etc.)
  │
  ├── Set Environment Variables
  │     └── TF_CPP_MIN_LOG_LEVEL = '3' (suppress TF logs)
  │
  ├── Define Configuration Constants:
  │     ├── ACCESS_TOKEN (Upstox API auth)
  │     ├── BASE_URL = "https://api.upstox.com/v3/historical-candle"
  │     ├── START_DATE = "2018-09-03"
  │     ├── END_DATE = "2025-12-04"
  │     ├── RISK_FREE_RATE = 0.065 (6.5%)
  │     ├── TRANSACTION_COST = 0.001 (0.1%)
  │     ├── INITIAL_CAPITAL = 100000.0 (₹1,00,000)
  │     └── BENCHMARK = "NSE_INDEX|Nifty 50"
  │
  └── Define CHOSEN_PORTFOLIO (5 stocks):
        ├── BAJFINANCE
        ├── SUNPHARMA
        ├── TI (Tube Investments)
        ├── BIOCON
        └── BHARTIARTL
```

### 🔷 PHASE 2: Data Fetching (Lines 48-85)
```
fetch_historical_data(instrument_key, from_date, to_date)
  │
  ├── Encode instrument key for URL
  │
  ├── Split date range into 365-day chunks (API limit)
  │     └── Loop: current_start → current_end (±365 days)
  │
  ├── For each chunk:
  │     ├── Build URL: BASE_URL/{key}/days/1/{to}/{from}
  │     ├── Make HTTP GET request with auth header
  │     ├── Parse JSON response → candles array
  │     └── Sleep 0.1s (rate limiting)
  │
  ├── Combine all chunks into single DataFrame
  │     ├── Columns: timestamp, open, high, low, close, volume, oi
  │     ├── Set timestamp as index
  │     └── Localize timezone to None
  │
  └── Calculate returns: df['returns'] = close.pct_change()
```

### 🔷 PHASE 3: HRP Weight Optimization (Lines 108-243)
```
optimize_weights(returns_df)
  │
  ├── STEP 1: Compute Covariance & Correlation
  │     ├── cov = returns_df.cov()
  │     └── corr = returns_df.corr()
  │
  ├── STEP 2: Create Distance Matrix
  │     └── dist = sqrt((1 - corr) / 2)  ← Correlation to distance
  │
  ├── STEP 3: Hierarchical Clustering
  │     ├── condensed_dist = squareform(dist)
  │     └── link = linkage(condensed_dist, method='single')
  │
  ├── STEP 4: Quasi-Diagonalization
  │     └── get_quasi_diag(link) → reordered indices
  │         │
  │         └── Recursive seriation:
  │               if cur_index < N: return [cur_index]
  │               else: left + right (recursive)
  │
  └── STEP 5: Recursive Bisection
        │
        └── get_rec_bipart(cov, sort_ix):
              │
              ├── Initialize: w = [1.0, 1.0, 1.0, 1.0, 1.0]
              │
              └── For each level:
                    ├── Split cluster at midpoint
                    ├── Compute cluster variance:
                    │     cov_slice = cov[items, items]
                    │     inv_var_weights = 1 / diag(cov_slice)
                    │     cluster_var = w·cov·w
                    │
                    ├── Calculate alpha:
                    │     alpha = 1 - var0 / (var0 + var1)
                    │
                    └── Apply weights:
                          w[cluster0] *= alpha
                          w[cluster1] *= (1 - alpha)
```

**Example HRP Output:**
```
Asset count: 5
Assets: ['BAJFINANCE', 'SUNPHARMA', 'TI', 'BIOCON', 'BHARTIARTL']

Correlation Matrix:
           BAJFINANCE  SUNPHARMA    TI  BIOCON  BHARTIARTL
BAJFINANCE      1.000      0.312  0.456   0.289       0.534
SUNPHARMA       0.312      1.000  0.198   0.678       0.245
...

FINAL WEIGHTS:
  BAJFINANCE: 0.2143
  SUNPHARMA:  0.1876
  TI:         0.2234
  BIOCON:     0.1654
  BHARTIARTL: 0.2093
```

### 🔷 PHASE 4: Technical Indicators (Lines 246-473)
```
calculate_indicators(df)
  │
  ├── ALLIGATOR INDICATOR (Bill Williams)
  │     ├── Jaw:   SMMA(median_price, 13).shift(8)   ← Blue line
  │     ├── Teeth: SMMA(median_price, 8).shift(5)    ← Red line
  │     └── Lips:  SMMA(median_price, 5).shift(3)    ← Green line
  │
  │     Enhanced Features:
  │     ├── jaw_teeth_dist, teeth_lips_dist, jaw_lips_dist
  │     ├── formation_code (0-8): Market pattern classification
  │     ├── position_state (0-5): Price relative to alligator
  │     └── converging flags (binary)
  │
  ├── RSI (Relative Strength Index)
  │     ├── delta = close.diff()
  │     ├── gain = delta.where(>0, 0).rolling(14).mean()
  │     ├── loss = -delta.where(<0, 0).rolling(14).mean()
  │     ├── RS = gain / loss
  │     └── RSI = 100 - (100 / (1 + RS))
  │
  │     RSI States (0-8):
  │     ├── 0: Extreme oversold (<20)
  │     ├── 1: Oversold (20-30)
  │     ├── 3: Neutral (40-60)
  │     ├── 6: Extreme overbought (>80)
  │     └── 7-8: Momentum states
  │
  ├── MACD (Moving Average Convergence Divergence)
  │     ├── EMA_12 = close.ewm(span=12).mean()
  │     ├── EMA_26 = close.ewm(span=26).mean()
  │     ├── MACD = EMA_12 - EMA_26
  │     ├── Signal = MACD.ewm(span=9).mean()
  │     └── Histogram = MACD - Signal
  │
  │     MACD States (0-9):
  │     ├── 0: Bullish crossover
  │     ├── 1: Bearish crossover
  │     ├── 2-3: Trend states
  │     └── 6-7: Zero line crossovers
  │
  ├── BOLLINGER BANDS
  │     ├── Middle = close.rolling(20).mean()
  │     ├── Std = close.rolling(20).std()
  │     ├── Upper = Middle + (Std × 2)
  │     └── Lower = Middle - (Std × 2)
  │
  │     BB States (0-8):
  │     ├── 0: Price touches lower band (buy signal)
  │     ├── 2: Price touches upper band (sell signal)
  │     └── 5-6: Position within bands
  │
  └── VOLATILITY
        ├── rolling_volatility = returns.rolling(20).std() × √252
        └── volatility_normalized = rolling_vol / close.rolling(252).mean()

TOTAL FEATURES: 31 indicators for TCN model
```

### 🔷 PHASE 5: Portfolio Simulation with Stop-Loss (Lines 476-604)
```
simulate_portfolio(all_stock_data)
  │
  ├── INITIALIZATION:
  │     ├── STOP_LOSS_THRESHOLD = 0.15 (15%)
  │     ├── current_capital = ₹100,000
  │     ├── entry_value = ₹100,000
  │     ├── in_position = False
  │     └── stop_loss_triggered = False
  │
  ├── INITIAL PURCHASE:
  │     ├── Get first prices for all stocks
  │     ├── units = (capital × weights) / prices
  │     ├── in_position = True
  │     └── next_rebalance_date = start + 60 days
  │
  └── DAILY SIMULATION LOOP:
        │
        For each trading day:
        │
        ├── Get current prices (close, high, low)
        │
        ├── IF in_position:
        │     │
        │     ├── Calculate daily_value = Σ(units × current_prices)
        │     │
        │     ├── Check stop-loss:
        │     │     drawdown = (entry_value - daily_value) / entry_value
        │     │     
        │     │     IF drawdown >= 15%:
        │     │         ├── PRINT "[!] STOP-LOSS TRIGGERED"
        │     │         ├── Exit all positions → cash
        │     │         ├── units = [0, 0, 0, 0, 0]
        │     │         ├── in_position = False
        │     │         └── Wait for next scheduled rebalance
        │     │
        │     └── Record portfolio value
        │
        ├── ELSE (in cash):
        │     └── daily_value = current_capital (unchanged)
        │
        └── IF date >= next_rebalance_date:
              │
              ├── Get 120-day lookback returns
              │
              ├── Calculate new HRP weights:
              │     new_weights = optimize_weights(lookback_df)
              │
              ├── Apply transaction cost:
              │     current_capital -= capital × 0.1%
              │
              ├── Re-enter positions:
              │     units = (capital × new_weights) / prices
              │
              ├── Reset tracking:
              │     ├── entry_value = current_capital
              │     ├── in_position = True
              │     └── stop_loss_triggered = False
              │
              └── next_rebalance_date += 60 days
```

**Stop-Loss Example Output:**
```
[!] STOP-LOSS TRIGGERED on 2022-06-15
   Entry Value: Rs.1,45,234.56
   Current Value: Rs.1,21,567.89
   Drawdown: 16.30%
   Exiting positions and moving to cash until 2022-08-14

[SUMMARY] Stop-Loss Summary: Triggered 3 times during simulation
```

### 🔷 PHASE 6: TCN Prediction (Lines 1370-1550)
```
PREDICTION WORKFLOW:
  │
  ├── Load Model:
  │     model = load_model('generalized_lstm_model_improved.keras')
  │
  ├── Prepare Data:
  │     ├── Calculate indicators on portfolio_df
  │     ├── Take last 120 trading days
  │     └── Extract 31 feature columns
  │
  ├── Scale Features:
  │     scaler = MinMaxScaler(feature_range=(0, 1))
  │     scaled_features = scaler.fit_transform(features)
  │
  ├── Reshape for Model:
  │     input_data = scaled_features.reshape(1, 120, 31)
  │     └── Shape: (batch=1, timesteps=120, features=31)
  │
  ├── Predict:
  │     prediction = model.predict(input_data)
  │     predicted_return_factor = prediction[0][0]
  │     └── Example: 1.0456 means +4.56% return
  │
  └── Calculate Values:
        current_value = portfolio_df['close'].iloc[-1]
        predicted_value = current_value × predicted_return_factor
        pl_pct = (predicted_return_factor - 1) × 100
```

### 🔷 PHASE 7: RAG Analysis Integration (Lines 1597-1961)
```
RAG-ENHANCED PREDICTION:
  │
  ├── Load RAG System:
  │     load_rag_system()  → ChromaDB vector store
  │
  ├── Extract Current Indicators:
  │     current_indicators = {
  │         "rsi": 54.3,
  │         "macd": 234.56,
  │         "volatility": 0.23,
  │         "alligator_jaw": 45678.90,
  │         "alligator_teeth": 45123.45,
  │         "alligator_lips": 44890.12
  │     }
  │
  ├── Find Similar Historical Scenarios:
  │     │
  │     ├── Get 100 similar scenarios from ChromaDB
  │     │
  │     ├── Calculate cosine distances:
  │     │     query_embedding = [rsi/100, tanh(macd/10000), vol, ...]
  │     │     distance = 1 - (dot_product / norms)
  │     │
  │     ├── Dynamic K Selection:
  │     │     ├── Get median distance of top 10
  │     │     ├── threshold = median × 1.5
  │     │     └── dynamic_k = count(distances < threshold)
  │     │           └── Bounded: 10 ≤ k ≤ 50
  │     │
  │     └── Extract outcomes from k scenarios
  │
  ├── Calculate Statistics:
  │     outcomes = [s['target_return_60d'] for s in scenarios]
  │     avg_return = mean(outcomes)
  │     std_return = std(outcomes)
  │     success_rate = positives / total × 100
  │     conf_low = avg - 2×std
  │     conf_high = avg + 2×std
  │
  ├── TCN vs History Alignment:
  │     IF |tcn - avg| < std:     → "[ALIGNED]"
  │     ELIF |tcn - avg| < 2×std: → "[MODERATE]"
  │     ELSE:                      → "[DIVERGENT]"
  │
  └── Gemini AI Analysis (Optional):
        ├── Build comprehensive prompt with all statistics
        ├── Request analysis from gemini-2.0-flash-exp
        └── Generate trading recommendations
```

**RAG Output Example:**
```
HISTORICAL CONTEXT (27 Similar Scenarios):
------------------------------------------------------------
TCN Prediction:           +4.56%
Historical Average:       3.21%
Historical Median:        2.98%
Historical Range:         -12.34% to +18.76%
Standard Deviation:       5.67%
Success Rate:             74.1% (20/27 positive)
95% Confidence Interval:  -8.13% to +14.55%
TCN vs History:           [ALIGNED]
                          TCN prediction is within 1 std dev of historical average
```

### 🔷 PHASE 8: Database Operations (Lines 666-804)
```
DATABASE SCHEMA:
┌─────────────────────────────────────────────────────────────────┐
│ TABLE: portfolio_predictions                                      │
├─────────────────────────────────────────────────────────────────┤
│ id                     INTEGER PRIMARY KEY                        │
│ rebalance_date         TEXT                                       │
│ expiry_date            TEXT                                       │
│ portfolio_value_at_rebalance  REAL                               │
│ predicted_return_factor       REAL                               │
│ predicted_value_at_expiry     REAL                               │
│ weights_json           TEXT (JSON)                               │
│ correlations_json      TEXT (JSON)                               │
│ risk_metrics_json      TEXT (JSON)                               │
│ portfolio_assets       TEXT                                       │
│ created_at             TIMESTAMP                                  │
│ UNIQUE(rebalance_date, portfolio_assets)                         │
└─────────────────────────────────────────────────────────────────┘

OPERATIONS:
├── init_db()        → Create table if not exists, handle schema migration
├── get_existing_prediction(date, signature) → Check for duplicate
└── save_prediction_to_db(data) → Insert new record
```

### 🔷 PHASE 9: Report Generation (Lines 1134-1332)
```
generate_detailed_report():
  │
  ├── RISK & PERFORMANCE RATIOS:
  │     ├── Sharpe Ratio = (excess_return × √252) / std
  │     ├── Sortino Ratio = (excess_return × 252) / downside_std
  │     └── Information Ratio = (active_return × 252) / tracking_error
  │
  ├── DRAWDOWNS:
  │     cumulative_returns = (1 + returns).cumprod()
  │     peak = cumulative_returns.expanding().max()
  │     drawdown = (cumulative - peak) / peak
  │     max_drawdown = drawdown.min()
  │
  ├── RETURNS:
  │     ├── 1-Day, 5-Day, 1-Month, 3-Month, 6-Month
  │     └── CAGR: 1Y, 3Y, 5Y
  │
  ├── VOLATILITY & TRACKING:
  │     ├── Portfolio Std Dev (annualized)
  │     ├── Benchmark Std Dev
  │     ├── Tracking Error
  │     └── Rolling 20-day Volatility
  │
  ├── RISK SENSITIVITY:
  │     ├── Beta = Cov(port, bench) / Var(bench)
  │     └── Weighted Beta = Σ(stock_beta × weight)
  │
  ├── ALPHA & RELATED:
  │     ├── Jensen's Alpha = Rp - (Rf + β×(Rm - Rf))
  │     ├── R-squared
  │     ├── Alpha Skewness
  │     └── Mean Alpha on Stress Days (<-2%)
  │
  ├── VALUE AT RISK (95%):
  │     ├── 1-Day VaR
  │     ├── 1-Day CVaR (Conditional VaR)
  │     └── Annualized versions
  │
  ├── DISTRIBUTION METRICS:
  │     ├── Skewness
  │     └── Kurtosis
  │
  ├── DIVERSIFICATION RATIO:
  │     (Weighted Avg Individual Vol) / Portfolio Vol
  │
  └── CORRELATION MATRIX:
        Pairwise stock return correlations
```

### 🔷 PHASE 10: Visualization (Lines 1965-2074)
```
FORECAST PLOT GENERATION:
  │
  ├── Historical Data (Last 6 months):
  │     plt.plot(hist_data.index, hist_data['close'], 'b-')
  │
  ├── TCN Forecast (Next 60 days):
  │     ├── daily_factor = predicted_factor^(1/60)
  │     ├── forecast_values = [start × daily_factor^i for i in 1..60]
  │     └── plt.plot(forecast_dates, forecast_values, 'r--')
  │
  ├── RAG Historical Range:
  │     ├── min_forecast, max_forecast (from RAG boundaries)
  │     └── plt.fill_between(dates, min, max, alpha=0.2)
  │
  ├── Key Markers:
  │     ├── Rebalance Date (vertical orange line)
  │     ├── Current Date (vertical gray line)
  │     └── Scatter points at start/end
  │
  └── Save:
        plt.savefig('portfolio_forecast_YYYYMMDD.png', dpi=300)
```

---

## 1.4 Key Components Deep Dive

### Component A: Hierarchical Risk Parity (HRP)
```
WHY HRP?
├── Traditional Mean-Variance (Markowitz) is unstable
├── HRP uses correlation structure for robustness
└── No need to estimate expected returns (hard to predict)

HRP ALGORITHM:
1. Correlation → Distance: dist[i,j] = √((1-corr[i,j])/2)
2. Cluster using Single Linkage
3. Quasi-diagonalize (reorder by cluster)
4. Recursive bisection:
   - Split cluster
   - Allocate inversely to variance
   - Higher variance → lower weight
```

### Component B: TCN Model Input
```
TCN INPUT SHAPE: (1, 120, 31)
├── 1 = batch size
├── 120 = lookback window (trading days)
└── 31 = features:
    ├── Price: close, high, low (3)
    ├── Alligator: jaw, teeth, lips + 10 derived (13)
    ├── RSI: rsi, rsi_state, rsi_normalized (3)
    ├── MACD: macd, signal, hist + 4 derived (7)
    ├── BB: upper, middle, lower, state + 2 normalized (5)
    └── Volatility: rolling, normalized (2)

TCN OUTPUT: Single float (predicted return factor)
├── > 1.0 = positive return expected
├── = 1.0 = no change expected
└── < 1.0 = negative return expected
```

### Component C: Stop-Loss Mechanism
```
STOP-LOSS LOGIC:
┌─────────────────────────────────────────┐
│ Entry Value = Portfolio at Rebalance    │
└─────────────────┬───────────────────────┘
                  │
      ┌───────────▼───────────┐
      │ Check Daily Drawdown  │
      │ DD = (Entry - Current)│
      │      / Entry          │
      └───────────┬───────────┘
                  │
         ┌────────┴────────┐
    DD < 15%           DD ≥ 15%
         │                  │
         ▼                  ▼
    ┌────────┐       ┌────────────┐
    │ HOLD   │       │ EXIT TO    │
    │ POSITION│      │ CASH       │
    └────────┘       └─────┬──────┘
                           │
                           ▼
                    ┌────────────┐
                    │ WAIT UNTIL │
                    │ NEXT       │
                    │ REBALANCE  │
                    └────────────┘
```

---

## 1.5 Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              DATA FLOW                                       │
└─────────────────────────────────────────────────────────────────────────────┘

[Upstox API]
     │
     │ HTTP GET (JSON)
     ▼
[Raw OHLCV Data]
     │
     │ pd.DataFrame
     ▼
[5 Stock DataFrames]───────────────────┐
     │                                  │
     │ align to common dates           │
     ▼                                  ▼
[Aligned Stock Data] ────────────▶ [Individual Returns]
     │                                  │
     │ weight × price                   │ optimize_weights()
     ▼                                  ▼
[Portfolio Value Series] ◀───────── [HRP Weights]
     │
     │ calculate_indicators()
     ▼
[Portfolio with 31 Indicators]
     │
     ├── Last 120 days ──────────▶ [TCN Model] ──────▶ [Prediction]
     │                                                      │
     │                                                      ▼
     ├── Current indicators ─────▶ [ChromaDB] ──────▶ [Similar Scenarios]
     │                             (RAG Search)             │
     │                                                      ▼
     │                                              [RAG Statistics]
     │                                                      │
     │                              ┌───────────────────────┘
     │                              ▼
     └─────────────────────────▶ [Combined Analysis]
                                        │
              ┌─────────────────────────┼─────────────────────────┐
              ▼                         ▼                         ▼
     [SQLite Database]         [Performance Report]        [Forecast Plot]
     (portfolio_analysis.db)   (Console Output)            (PNG Image)
```

---

# 2. portfolio_backtest_optimized_v2.py

## 2.1 Overview

| Property | Value |
|----------|-------|
| **Lines of Code** | 1,108 |
| **File Size** | 45.7 KB |
| **Purpose** | Historical backtesting with TCN + RAG |
| **Key Models** | TCN (generalized_lstm_model_improved_v2.keras) |
| **Data Source** | Upstox API |
| **Database** | portfolio_backtest_tcn.db |
| **Rebalancing** | Every 60 trading days (simulated) |

### What It Does (High Level)
This is the **backtesting script** for validating the strategy. It:
1. Tests all possible 5-stock combinations (C(10,5) = 252 portfolios)
2. Uses TCN predictions and RAG risk assessment for selection
3. Selects best portfolio based on Risk/Reward ratio
4. Simulates holding for 60 days, then rebalances
5. Tracks full performance metrics over multi-year period
6. Saves results to SQLite database

---

## 2.2 Architecture Diagram

```
┌────────────────────────────────────────────────────────────────────────────────┐
│                    portfolio_backtest_optimized_v2.py                           │
├────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐                         │
│  │ Stock       │───▶│ Combination │───▶│ For Each    │                         │
│  │ Universe(10)│    │ Generator   │    │ Portfolio   │                         │
│  └─────────────┘    │ C(10,5)=252 │    │ Combination │                         │
│                     └─────────────┘    └──────┬──────┘                         │
│                                               │                                 │
│              ┌────────────────────────────────┼───────────────────────────┐    │
│              │                                ▼                           │    │
│              │   ┌─────────────┐    ┌─────────────┐    ┌─────────────┐   │    │
│              │   │ HRP Weights │───▶│ TCN Predict │───▶│ RAG Risk    │   │    │
│              │   │ Optimization│    │ (Next 60d)  │    │ Assessment  │   │    │
│              │   └─────────────┘    └─────────────┘    └──────┬──────┘   │    │
│              │                                                │          │    │
│              │                       ┌────────────────────────┘          │    │
│              │                       ▼                                   │    │
│              │              ┌─────────────────┐                          │    │
│              │              │ Risk/Reward     │                          │    │
│              │              │ Ratio = TCN/Risk│                          │    │
│              │              └────────┬────────┘                          │    │
│              │                       │                                   │    │
│              └───────────────────────┼───────────────────────────────────┘    │
│                                      ▼                                         │
│                             ┌─────────────────┐                                │
│                             │ SELECT BEST     │                                │
│                             │ PORTFOLIO       │                                │
│                             │ (Max R/R Ratio) │                                │
│                             └────────┬────────┘                                │
│                                      │                                         │
│                                      ▼                                         │
│                             ┌─────────────────┐                                │
│                             │ SIMULATE        │                                │
│                             │ 60-DAY HOLD     │                                │
│                             └────────┬────────┘                                │
│                                      │                                         │
│       ┌──────────────────────────────┼───────────────────────────┐            │
│       ▼                              ▼                           ▼            │
│  ┌─────────┐                  ┌─────────────┐            ┌─────────────┐      │
│  │ Database│                  │ Performance │            │ Forecast    │      │
│  │ Results │                  │ Report      │            │ Plot        │      │
│  └─────────┘                  └─────────────┘            └─────────────┘      │
│                                                                                │
└────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2.3 Complete Execution Flow

### 🔷 PHASE 1: Initialization (Lines 1-62)
```
START
  │
  ├── Import Libraries
  │
  ├── Configuration:
  │     ├── START_DATE = "2018-09-03"
  │     ├── END_DATE = "2025-12-02"
  │     ├── LOOKBACK_WINDOW = 120
  │     ├── FORWARD_WINDOW = 60
  │     ├── WARMUP_DAYS = 120
  │     ├── MIN_REQUIRED_DAYS = 240
  │     ├── RISK_FREE_RATE = 6.5%
  │     ├── TRANSACTION_COST = 0.1%
  │     └── INITIAL_CAPITAL = ₹100,000
  │
  └── STOCK_UNIVERSE (10 stocks):
        ├── BPCL
        ├── ABCAPITAL
        ├── BAJFINANCE
        ├── BHARTIARTL
        ├── SBIN
        ├── KOTAKBANK
        ├── AXISBANK
        ├── HINDUNILVR
        ├── TITAN
        └── BAJAJ-AUTO
```

### 🔷 PHASE 2: Data Fetching (Lines 114-152)
```
Same as rebalanced_predictive_model.py:
├── Fetch all 10 stocks from Upstox API
├── Calculate technical indicators for each
└── Align to common trading dates
```

### 🔷 PHASE 3: Portfolio Combinations (Lines 903-906)
```
combos = list(itertools.combinations(stock_keys, 5))
│
└── C(10, 5) = 252 unique 5-stock portfolios
```

### 🔷 PHASE 4: Main Backtest Loop (Lines 928-1072)
```
BACKTEST LOOP:
  │
  ├── Initialize:
  │     ├── capital = ₹100,000
  │     ├── current_date = first_valid_date (after 240 warmup days)
  │     ├── daily_portfolio_values = []
  │     └── daily_dates = []
  │
  └── WHILE current_date < end_date - 60 days:
        │
        ├── Cycle N: Rebalance Date = current_date
        │
        ├── For EACH of 252 portfolios:
        │     │
        │     ├── Get lookback data (last 120 days)
        │     │
        │     ├── Calculate HRP weights:
        │     │     returns_df = get_returns(combo, lookback_period)
        │     │     weights = optimize_weights_hrp(returns_df)
        │     │
        │     ├── Create weighted portfolio series:
        │     │     port_close = Σ(stock_close × weight)
        │     │     port_high = Σ(stock_high × weight)
        │     │     ... etc
        │     │
        │     ├── Get TCN Prediction:
        │     │     tcn_pred = predict_portfolio_return(port_to_date, model)
        │     │     └── Returns % (e.g., +4.56%)
        │     │
        │     ├── Get RAG Metrics:
        │     │     rag_res = get_rag_metrics(port_to_date)
        │     │     ├── ci_lower (95% lower bound)
        │     │     ├── ci_upper (95% upper bound)
        │     │     └── success_rate
        │     │
        │     └── Calculate Risk/Reward:
        │           risk = abs(ci_lower)
        │           if risk < 0.1: risk = 0.1  # Floor
        │           rr_ratio = tcn_pred / risk
        │
        ├── SELECT BEST PORTFOLIO:
        │     best = argmax(rr_ratio) across all 252 combos
        │
        ├── SIMULATE HOLDING PERIOD (60 days):
        │     │
        │     └── For each day in next 60 trading days:
        │           ├── Calculate portfolio value:
        │           │     day_val = 0
        │           │     for each stock:
        │           │         entry_price = price at rebalance
        │           │         current_price = price today
        │           │         return = current / entry
        │           │         day_val += (entry_capital × weight) × return
        │           │
        │           ├── Track daily values:
        │           │     daily_portfolio_values.append(day_val)
        │           │     daily_dates.append(day)
        │           │
        │           └── Update capital:
        │                 capital = day_val
        │
        ├── Calculate actual return:
        │     actual_return = (exit_val / entry_val - 1) × 100
        │
        ├── Store result to database
        │
        └── Move to next cycle:
              current_date += 60 trading days
```

### 🔷 PHASE 5: TCN Prediction Function (Lines 352-397)
```
predict_portfolio_return(portfolio_df, model):
  │
  ├── Calculate all indicators on full data
  │
  ├── Check data sufficiency:
  │     if len(data) < 121: return None
  │
  ├── Extract last 120 days
  │
  ├── Get 31 feature columns (EXACT match to training):
  │     ['close', 'high', 'low', 
  │      'alligator_jaw', 'alligator_teeth', 'alligator_lips',
  │      'alligator_jaw_teeth_dist_norm', ... ,
  │      'rolling_volatility', 'volatility_normalized']
  │
  ├── Normalize with MinMaxScaler
  │
  ├── Reshape: (1, 120, 31)
  │
  ├── Model prediction:
  │     prediction = model.predict(X)[0][0]
  │
  └── Convert to percentage:
        return_pct = (prediction - 1) × 100
```

### 🔷 PHASE 6: RAG Risk Metrics (Lines 399-446)
```
get_rag_metrics(portfolio_df):
  │
  ├── Calculate indicators
  │
  ├── Extract latest values:
  │     indicators = {
  │         'rsi': latest['rsi'],
  │         'volatility': latest['rolling_volatility'],
  │         'macd': latest['macd'],
  │         'bb_upper': latest['bb_upper'],
  │         'bb_lower': latest['bb_lower'],
  │         'close': latest['close']
  │     }
  │
  ├── Find 50 similar scenarios from ChromaDB
  │
  ├── Extract historical outcomes:
  │     outcomes = [s['target_return_60d'] for s in scenarios]
  │
  └── Return statistics:
        {
            'ci_lower': mean - 1.96×std,
            'ci_upper': mean + 1.96×std,
            'mean': mean(outcomes),
            'std': std(outcomes),
            'median_return': median(outcomes),
            'success_rate': positive_count / total × 100,
            'scenarios_count': 50
        }
```

### 🔷 PHASE 7: Performance Report (Lines 449-670)
```
generate_backtest_performance_report():
  │
  ├── Create portfolio DataFrame from daily values
  │
  ├── Calculate all metrics (same as production):
  │     ├── Sharpe, Sortino, Information Ratio
  │     ├── Max Drawdown (portfolio & benchmark)
  │     ├── Returns (1d, 5d, 1m, 3m, 6m, CAGR)
  │     ├── Volatility & Tracking Error
  │     ├── Beta (weighted & general)
  │     ├── Jensen's Alpha, R-squared
  │     ├── VaR, CVaR (95%)
  │     ├── Skewness, Kurtosis
  │     ├── Diversification Ratio
  │     └── Correlation Matrix
  │
  └── Print formatted report
```

### 🔷 PHASE 8: Forecast Plot (Lines 672-840)
```
plot_portfolio_forecast():
  │
  ├── Get last 120 days of actual performance
  │
  ├── Get TCN prediction for next 60 days
  │
  ├── Get RAG confidence intervals
  │
  ├── Create matplotlib figure (14×7 inches)
  │
  ├── Plot:
  │     ├── Historical line (blue, solid)
  │     ├── TCN forecast line (purple, dashed)
  │     ├── RAG confidence band (orange, filled)
  │     └── Current date vertical line
  │
  ├── Add annotations:
  │     ├── Current value label
  │     └── Predicted value with percentage
  │
  └── Save as 'portfolio_forecast_plot.png'
```

### 🔷 PHASE 9: Database Storage (Lines 65-112)
```
DATABASE SCHEMA:
┌─────────────────────────────────────────────────────────────────┐
│ TABLE: backtest_results                                          │
├─────────────────────────────────────────────────────────────────┤
│ id                     INTEGER PRIMARY KEY                       │
│ rebalance_date         TEXT                                      │
│ portfolio              TEXT (comma-separated symbols)            │
│ tcn_prediction         REAL                                      │
│ rag_ci_lower           REAL                                      │
│ rag_ci_upper           REAL                                      │
│ risk_reward_ratio      REAL                                      │
│ actual_return          REAL                                      │
│ exit_date              TEXT                                      │
│ exit_reason            TEXT                                      │
│ portfolio_entry_value  REAL                                      │
│ portfolio_exit_value   REAL                                      │
│ created_at             TIMESTAMP                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2.4 Key Components Deep Dive

### Component A: Portfolio Selection Logic
```
SELECTION ALGORITHM:
  │
  For each portfolio (252 total):
  │
  ├── Compute HRP Weights
  │
  ├── Get TCN Prediction (upside potential)
  │     └── tcn_pred = predicted % return
  │
  ├── Get RAG Risk (downside risk)
  │     └── ci_lower = 95% lower bound
  │
  └── Calculate Risk/Reward:
        rr = tcn_pred / |ci_lower|
        
        Higher rr means:
        ├── Higher expected upside
        └── Lower expected downside
        
        SELECT portfolio with MAX(rr)
```

### Component B: Holding Period Simulation
```
HOLDING LOGIC:
┌──────────────────────────────────────────────────────────────┐
│ Unlike production script:                                     │
│ ├── NO stop-loss (hold full 60 days)                         │
│ ├── NO early exit                                            │
│ └── Purpose: Pure strategy evaluation                        │
└──────────────────────────────────────────────────────────────┘

For each day in 60-day period:
├── Calculate weighted portfolio value
├── Track for performance analysis
└── Update capital at end
```

### Component C: Data Alignment
```
ALIGNMENT REQUIREMENT:
├── All 10 stocks must have data for same dates
├── Benchmark (Nifty 50) must align too
└── Minimum 240 days required before starting

common_index = intersection of all stock dates
aligned_data = {k: v.loc[common_index] for all stocks}
```

---

## 2.5 Comparison with Production Script

| Feature | rebalanced_predictive_model.py | portfolio_backtest_optimized_v2.py |
|---------|--------------------------------|-----------------------------------|
| **Purpose** | Live trading | Historical validation |
| **Portfolio** | Fixed 5 stocks | Tests 252 combinations |
| **Selection** | Manual/predefined | Automated (best R/R) |
| **Stop-Loss** | ✅ 15% implemented | ❌ None (pure hold) |
| **Time Range** | Current to +60 days | Full historical (2018-2025) |
| **Database** | portfolio_analysis.db | portfolio_backtest_tcn.db |
| **RAG Usage** | Analysis & AI insights | Portfolio selection |
| **Output** | Prediction + Report | Backtest results + Report |

---

# 3. Technical Glossary

| Term | Definition |
|------|------------|
| **HRP** | Hierarchical Risk Parity - Weight allocation method using clustering |
| **TCN** | Temporal Convolutional Network - Deep learning for time series |
| **RAG** | Retrieval-Augmented Generation - Finding similar historical scenarios |
| **SMMA** | Smoothed Moving Average (Alligator indicator) |
| **RSI** | Relative Strength Index (momentum indicator) |
| **MACD** | Moving Average Convergence Divergence |
| **Bollinger Bands** | Volatility bands around moving average |
| **Sharpe Ratio** | Risk-adjusted return (excess return / volatility) |
| **Sortino Ratio** | Like Sharpe but only considers downside risk |
| **Beta** | Sensitivity to market movements |
| **Jensen's Alpha** | Excess return over CAPM expected return |
| **VaR** | Value at Risk - Maximum expected loss at confidence level |
| **CVaR** | Conditional VaR - Average loss beyond VaR |
| **Drawdown** | Peak-to-trough decline |
| **CAGR** | Compound Annual Growth Rate |
| **ChromaDB** | Vector database for similarity search |
| **Quasi-Diagonalization** | Reordering matrix based on cluster structure |
| **Recursive Bisection** | Splitting clusters for weight allocation |

---

> **Document Complete** | Total Pages: ~25 equivalent | All flows documented with code references
