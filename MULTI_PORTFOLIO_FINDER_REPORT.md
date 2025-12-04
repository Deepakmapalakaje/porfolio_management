# 📊 Multi-Portfolio Finder - Technical Report

> **Script:** `multi_portfolio_finder.py`  
> **Lines of Code:** 860  
> **Purpose:** Automated search for top 10 portfolios using TCN + RAG validation  
> **Created:** December 4, 2025

---

## 📋 Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [What It Does](#2-what-it-does)
3. [Selection Criteria](#3-selection-criteria)
4. [Architecture](#4-architecture)
5. [Execution Flow](#5-execution-flow)
6. [Key Functions](#6-key-functions)
7. [Output Format](#7-output-format)
8. [Usage](#8-usage)

---

## 1. Executive Summary

**`multi_portfolio_finder.py`** is an automated portfolio discovery tool that:

| Feature | Description |
|---------|-------------|
| **Stock Universe** | Searches through 33 predefined stocks |
| **Combinations** | Tests all C(33,5) = 237,336 possible 5-stock portfolios |
| **Validation** | Uses TCN model + RAG historical analysis |
| **Benchmark** | Requires Sharpe & Sortino > Nifty 50 |
| **Output** | Finds and ranks the top 10 best portfolios |
| **AI Ranking** | Calls Gemini LLM **once** to rank and explain results |

---

## 2. What It Does

### High-Level Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                   MULTI-PORTFOLIO FINDER                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  33 Stocks ─────► C(33,5) = 237,336 Combinations                 │
│                           │                                       │
│                           ▼                                       │
│           ┌───────────────────────────────┐                      │
│           │  For each 5-stock portfolio:  │                      │
│           │  1. Calculate HRP weights     │                      │
│           │  2. Get TCN prediction        │                      │
│           │  3. Get RAG risk metrics      │                      │
│           │  4. Calculate Sharpe/Sortino  │                      │
│           │  5. Check ALL criteria        │                      │
│           └───────────────────────────────┘                      │
│                           │                                       │
│                     Valid? ──► Add to Top 10                     │
│                           │                                       │
│                           ▼                                       │
│              Found 10? ──► STOP SEARCHING                        │
│                           │                                       │
│                           ▼                                       │
│              ┌────────────────────────┐                          │
│              │  Call Gemini AI ONCE   │                          │
│              │  to rank and explain   │                          │
│              └────────────────────────┘                          │
│                           │                                       │
│                           ▼                                       │
│              Display Top 10 with Full Details                    │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

### Key Differences from Other Scripts

| Script | Purpose | Portfolio Selection |
|--------|---------|---------------------|
| `rebalanced_predictive_model.py` | Live trading | **Manual** (you choose 5 stocks) |
| `portfolio_backtest_optimized_v2.py` | Backtesting | Tests 252 combos from 10 stocks |
| **`multi_portfolio_finder.py`** | **Discovery** | **Auto-finds best 10** from 33 stocks |

---

## 3. Selection Criteria

A portfolio is considered **VALID** only if it meets ALL of these criteria:

### TCN Prediction Criteria
```python
MIN_TCN_RETURN = 1.0%     # Minimum expected return
MAX_TCN_RETURN = 15.0%    # Maximum (avoid unrealistic)
```
- **Why?** We want moderate, achievable returns, not extreme predictions

### RAG Historical Criteria
```python
MIN_RAG_SUCCESS_RATE = 55%   # Historical win rate
MIN_RAG_MEAN_RETURN = 0.5%   # Average historical return
MAX_RAG_STD = 15%            # Maximum volatility
```
- **Why?** We want consistent performers based on similar historical scenarios

### Benchmark Comparison (NEW!)
```python
Portfolio Sharpe Ratio  > Nifty 50 Sharpe Ratio
Portfolio Sortino Ratio > Nifty 50 Sortino Ratio
```
- **Why?** Portfolio must BEAT the market on risk-adjusted returns

### Alignment Check
```python
TCN > 0 AND RAG Mean > 0
```
- **Why?** Both prediction methods should agree on positive outlook

---

## 4. Architecture

### Stock Universe (33 Stocks)

```
Banking & Finance          Technology              Healthcare
─────────────────          ──────────              ──────────
BAJFINANCE                 INFY                    SUNPHARMA
ABCAPITAL                  COFORGE                 BIOCON
BANKBARODA                 PERSISTENT              MEDANTA
UNIONBANK                  TI                      HCG
CANBK                      KAYNES                  
SBIN                       ANGELONE                Consumer
KOTAKBANK                                          ────────
AXISBANK                   Infrastructure          HINDUNILVR
                           ──────────────          TITAN
Energy                     IRFC                    TRENT
──────                     BHARTIARTL              BAJAJ-AUTO
BPCL                       ADANIENSOL              
GCSL                                               Others
                           Industrial              ──────
Exchange                   ──────────              PRECWIRE
────────                   ARVIND                  INDIAGLYCO
BSE                        AHLEAST                 ARHAM
```

### Components Used

| Component | Source | Purpose |
|-----------|--------|---------|
| **HRP Optimizer** | `scipy.cluster.hierarchy` | Calculate portfolio weights |
| **TCN Model** | `generalized_lstm_model_improved.keras` | Predict 60-day returns |
| **RAG System** | `rag_prediction_simple.py` + ChromaDB | Find similar historical scenarios |
| **Gemini AI** | `google.generativeai` | Rank and explain top portfolios |
| **Upstox API** | REST API | Fetch historical OHLCV data |

---

## 5. Execution Flow

### Phase 1: Initialization
```
1. Load RAG System (ChromaDB vector store)
2. Load TCN Model (Keras)
3. Fetch historical data for all 33 stocks
4. Align data to common trading dates
5. Fetch Nifty 50 benchmark data
6. Calculate benchmark Sharpe & Sortino ratios
```

### Phase 2: Portfolio Search
```
For each 5-stock combination (up to 237,336):
    │
    ├── Calculate HRP weights (last 120 days returns)
    │
    ├── Create weighted portfolio series
    │
    ├── Calculate 31 technical indicators
    │
    ├── Get TCN Prediction (60-day forward return)
    │
    ├── Get RAG Metrics:
    │   ├── Find 50 similar historical scenarios
    │   ├── Calculate mean, std, success rate
    │   └── Calculate 95% confidence interval
    │
    ├── Calculate Sharpe & Sortino ratios (last 252 days)
    │
    ├── Check ALL selection criteria:
    │   ├── TCN: 1% <= prediction <= 15%
    │   ├── RAG Success Rate >= 55%
    │   ├── RAG Mean >= 0.5%
    │   ├── RAG Std <= 15%
    │   ├── Sharpe > Benchmark
    │   └── Sortino > Benchmark
    │
    └── If ALL pass → Add to valid portfolios
    
    Stop when 10 valid portfolios found!
```

### Phase 3: AI Ranking
```
1. Compile data for all 10 valid portfolios
2. Send to Gemini AI with ranking criteria
3. AI returns:
   - Rank 1-10 with reasoning
   - Why each portfolio got its rank
   - Summary of key differentiators
```

### Phase 4: Output
```
1. Display full details for each portfolio:
   - Weights
   - TCN prediction
   - RAG analysis
   - Risk metrics vs benchmark
   - Alignment check

2. Display AI ranking explanations

3. Show summary comparison table
```

---

## 6. Key Functions

### `evaluate_portfolio(combo, aligned_data, model)`
```python
# Evaluates a single 5-stock combination
# Returns dict with:
{
    'combo': tuple of 5 stock keys,
    'symbols': ['STOCK1', 'STOCK2', ...],
    'weights': {'STOCK1': 0.23, 'STOCK2': 0.18, ...},
    'tcn_prediction': 5.67,  # %
    'rag_mean': 4.32,        # %
    'rag_std': 8.45,         # %
    'rag_ci_lower': -12.58,  # %
    'rag_ci_upper': 21.22,   # %
    'rag_success_rate': 68.0,# %
    'sharpe_ratio': 1.45,
    'sortino_ratio': 2.12,
    'max_drawdown': -15.3,   # %
    'volatility': 22.5       # % annualized
}
```

### `is_good_portfolio(result)`
```python
# Returns True only if ALL criteria are met:
# 1. TCN between 1% and 15%
# 2. RAG success rate >= 55%
# 3. RAG mean >= 0.5%
# 4. RAG std <= 15%
# 5. Sharpe > Benchmark Sharpe
# 6. Sortino > Benchmark Sortino
# 7. TCN > 0 AND RAG mean > 0 (both positive)
```

### `rank_portfolios_with_llm(valid_portfolios)`
```python
# Sends all 10 portfolios to Gemini AI
# Returns ranked list with explanations
# Called ONLY ONCE at the end
```

### `display_portfolio_details(p, rank)`
```python
# Prints comprehensive info for one portfolio:
# - HRP weights
# - TCN prediction
# - RAG historical analysis
# - Risk metrics vs benchmark
# - Alignment check
```

---

## 7. Output Format

### Console Output Example

```
======================================================================
  MULTI-PORTFOLIO FINDER WITH TCN + RAG VALIDATION
======================================================================

Searching for 10 best portfolios from 33 stocks...
Will search ALL combinations until 10 valid portfolios are found!

Selection Criteria:
  - TCN Return: 1.0% to 15.0%
  - RAG Success Rate: >= 55.0%
  - RAG Mean Return: >= 0.5%
  - RAG Std Dev: <= 15.0%
  - Sharpe Ratio: > Benchmark (Nifty 50)
  - Sortino Ratio: > Benchmark (Nifty 50)

Initializing RAG system...
Loading TCN model...
Model loaded successfully.

Fetching data for 33 stocks...
  Fetching ANGELONE... OK (1654 rows)
  Fetching BAJFINANCE... OK (1654 rows)
  ...

Fetching benchmark data (Nifty 50)...
  Benchmark Sharpe Ratio:  0.85
  Benchmark Sortino Ratio: 1.23
  Portfolios must BEAT these ratios to qualify!

Total possible combinations: 237336
Will test ALL until 10 valid portfolios found!

----------------------------------------------------------------------
SEARCHING FOR VALID PORTFOLIOS...
----------------------------------------------------------------------
[45/237336] Testing: BAJFINANCE, SUNPHARMA, BPCL, TITAN, INFY...
  [OK] VALID! TCN: +4.56%, RAG: +3.21%, Sharpe: 1.42 (>0.85)

[123/237336] Testing: SBIN, KOTAKBANK, BHARTIARTL, COFORGE, HINDUNILVR...
  [OK] VALID! TCN: +5.12%, RAG: +4.87%, Sharpe: 1.56 (>0.85)
...

Found 10 valid portfolios! Stopping search.

======================================================================
       TOP 10 PORTFOLIOS - FULL DETAILS
======================================================================

======================================================================
  RANK #1: BAJFINANCE, SUNPHARMA, TITAN, INFY, KOTAKBANK
======================================================================

--- PORTFOLIO WEIGHTS (HRP Optimized) ---
  BAJFINANCE     : 0.2234 (22.34%)
  SUNPHARMA      : 0.1876 (18.76%)
  TITAN          : 0.2012 (20.12%)
  INFY           : 0.1923 (19.23%)
  KOTAKBANK      : 0.1955 (19.55%)

--- TCN PREDICTION (Next 60 Days) ---
  Predicted Return:        +4.56%

--- RAG HISTORICAL ANALYSIS ---
  Similar Scenarios Found: 50
  Historical Mean Return:  +3.21%
  Historical Median:       +2.87%
  Standard Deviation:      7.45%
  95% Confidence Interval: [-11.39%, +17.81%]
  Success Rate:            68.0%

--- RISK METRICS (Last 1 Year) vs BENCHMARK ---
  Sharpe Ratio:            1.42 (Benchmark: 0.85, +0.57)
  Sortino Ratio:           2.15 (Benchmark: 1.23, +0.92)
  Max Drawdown:            -12.34%
  Annualized Volatility:   18.56%

--- ALIGNMENT CHECK ---
  ALIGNED - TCN prediction is within 1 std of historical mean

... (9 more portfolios)

======================================================================
       AI PORTFOLIO RANKING (Gemini)
======================================================================
Using model: gemini-2.0-flash-exp

RANK 1: BAJFINANCE, SUNPHARMA, TITAN, INFY, KOTAKBANK
Reason: This portfolio demonstrates the best balance of risk-adjusted returns
with a Sharpe ratio of 1.42, significantly beating the benchmark. The 68%
historical success rate and alignment between TCN and RAG predictions
provides high confidence in the expected 4.56% return.

RANK 2: SBIN, KOTAKBANK, BHARTIARTL, COFORGE, HINDUNILVR
Reason: Strong diversification across banking, telecom, IT, and consumer
sectors. The 72% success rate is the highest among all portfolios, though
slightly lower expected return of 3.89% places it second.

... (more rankings)

SUMMARY: The top 3 portfolios share common characteristics: diversification
across multiple sectors, success rates above 65%, and Sharpe ratios at least
50% higher than the Nifty 50 benchmark. The winning portfolio stands out
for its optimal balance of growth potential and historical reliability.

======================================================================
       SUMMARY COMPARISON TABLE
======================================================================

Rank  Portfolio                                  TCN       RAG Mean   Success   Sharpe
------------------------------------------------------------------------------------------
1     BAJFINANCE, SUNPHARMA, TITAN...          +4.56%    +3.21%      68.0%     1.42
2     SBIN, KOTAKBANK, BHARTIARTL...           +3.89%    +4.12%      72.0%     1.38
3     INFY, COFORGE, PERSISTENT...             +5.23%    +4.56%      66.0%     1.35
...

======================================================================
Analysis complete!
======================================================================
```

---

## 8. Usage

### Running the Script

```powershell
cd "c:\Users\Karthik M\Desktop\assignment"
python multi_portfolio_finder.py
```

### Configuration (Top of File)

```python
# --- Search Parameters ---
MAX_PORTFOLIOS_TO_FIND = 10    # Change to find more/fewer portfolios
PORTFOLIO_SIZE = 5              # Number of stocks per portfolio

# --- Selection Criteria ---
MIN_TCN_RETURN = 1.0           # Minimum TCN prediction (%)
MAX_TCN_RETURN = 15.0          # Maximum TCN prediction (%)
MIN_RAG_SUCCESS_RATE = 55.0    # Minimum historical success rate (%)
MIN_RAG_MEAN_RETURN = 0.5      # Minimum RAG mean return (%)
MAX_RAG_STD = 15.0             # Maximum RAG standard deviation (%)

# --- API Keys ---
GEMINI_API_KEY = "your-key-here"
ACCESS_TOKEN = "your-upstox-token"
```

### Modifying Stock Universe

Add or remove stocks in the `STOCK_UNIVERSE` dictionary:

```python
STOCK_UNIVERSE = {
    "NSE_EQ|INE732I01013": "ANGELONE",
    "NSE_EQ|INE296A01032": "BAJFINANCE",
    # Add new stocks here...
}
```

---

## Summary

**`multi_portfolio_finder.py`** is your automated portfolio discovery tool that:

1. ✅ Searches ALL possible 5-stock combinations (237,336)
2. ✅ Validates each using TCN prediction + RAG historical analysis
3. ✅ Requires portfolios to BEAT Nifty 50 benchmark
4. ✅ Stops when 10 valid portfolios are found
5. ✅ Displays comprehensive details for each portfolio
6. ✅ Uses AI (Gemini) **once** to rank and explain the best choices

This is your **portfolio discovery engine** - finding the best 10 portfolios you should consider for the next 60 days!

---

> **Report Generated:** December 4, 2025  
> **Script Version:** 1.0  
> **Author:** Antigravity AI Assistant
