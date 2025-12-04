# 📂 Complete Project Documentation

**Generated on:** 2025-12-04  
**Project:** Portfolio Optimization, TCN/LSTM Prediction, and RAG-based Risk Assessment  
**Total Files:** 64 files + 7 directories

---

## 🚀 **1. CORE PRODUCTION SCRIPTS**

---

### **`rebalanced_predictive_model.py`**
| Property | Value |
|----------|-------|
| **Size** | 97 KB (2,088 lines) |
| **Type** | Main Production Script |
| **Status** | ✅ ESSENTIAL - Keep |

**Purpose:** The primary production script for live portfolio prediction and management with active risk controls.

**What It Does (Step-by-Step):**
1. **Portfolio Definition:** Manages a specific 5-stock portfolio (BSE, TRENT, TI, INDIAGLYCO, PRECWIRE)
2. **Data Fetching:** Retrieves historical OHLCV data from Upstox API for all assets
3. **Technical Indicators (31 features):**
   - Alligator Indicator (Jaw, Teeth, Lips + distances + formations)
   - RSI (value, state, normalized)
   - MACD (value, signal, histogram + states)
   - Bollinger Bands (upper, middle, lower, bandwidth)
   - Rolling Volatility
4. **Portfolio Simulation:**
   - 60-day rebalancing cycles
   - HRP (Hierarchical Risk Parity) weight optimization
   - 0.1% transaction costs per rebalance
5. **Stop-Loss Protection:**
   - Monitors daily if portfolio drops 15% from entry
   - Exits to cash and waits for next rebalance cycle
6. **TCN Prediction:**
   - Uses trained model to predict 60-day forward return factor
   - Input: 120-day lookback of technical indicators
7. **Database Storage:**
   - Saves predictions with portfolio signature tracking
   - Handles schema migrations automatically
8. **Performance Reporting:**
   - CAGR, Sharpe Ratio, Sortino Ratio
   - Max Drawdown, Alpha, Beta
   - VaR, CVaR at 95% confidence
   - Diversification ratio, correlation matrix
9. **Visualization:**
   - Historical equity curve
   - Future forecast with confidence bands

**Key Functions:**
- `fetch_historical_data()` - API data retrieval
- `calculate_indicators()` - Technical indicator computation
- `optimize_weights()` - HRP weight allocation
- `simulate_portfolio()` - Day-by-day portfolio simulation
- `generate_detailed_report()` - Comprehensive metrics
- `save_prediction_to_db()` - Database persistence

**Why Created:** Main production tool for running live portfolio predictions with integrated risk management.

---

### **`analysis.py`**
| Property | Value |
|----------|-------|
| **Size** | 23 KB (543 lines) |
| **Type** | Research & Discovery Script |
| **Status** | ✅ ESSENTIAL - Keep |

**Purpose:** Systematically discovers and evaluates profitable portfolio combinations from historical data.

**What It Does (Step-by-Step):**
1. **Stock Universe Definition:** Loads candidate stocks from predefined mapping
2. **Combination Generation:** Creates all unique 5-stock combinations (C(n,5))
3. **Parallel Backtesting:**
   - Uses multiprocessing for parallel evaluation
   - For each combination, simulates multi-year performance
4. **Optimization Cycle:**
   - 120-day lookback window for HRP weight calculation
   - 60-day forward performance evaluation
   - 60-day rebalancing frequency
5. **Benchmark Comparison:** Evaluates vs Nifty 50 index
6. **Filtering Criteria:**
   - Sharpe Ratio > Benchmark
   - Sortino Ratio > Benchmark
   - Positive profits (overall, last year, last 60 days)
7. **Database Persistence:** Saves all results to SQLite tables:
   - `report_overall` - Overall metrics
   - `report_last_year` - Recent 1-year performance
   - `report_quarterly_avg` - Quarterly averages
   - `report_outperforming_combinations` - Best portfolios
   - `report_rebalanced_portfolios` - All rebalance events
   - `report_rejected_portfolios` - Failed combinations
   - `report_yearly_profits` - Year-by-year breakdown

**Key Functions:**
- `fetch_historical_data()` - Chunked API data retrieval
- `get_metrics()` - Calculate Sharpe, Sortino, Drawdown, StdDev
- `optimize_weights()` - HRP optimization
- `evaluate_combination()` - Single portfolio evaluation (picklable for multiprocessing)
- `analyze_portfolio()` - Main orchestration function

**Why Created:** To systematically find profitable portfolio compositions for live trading.

---

### **`portfolio_backtest_optimized_v2.py`**
| Property | Value |
|----------|-------|
| **Size** | 46 KB (1,108 lines) |
| **Type** | Validation Engine |
| **Status** | ✅ ESSENTIAL - Keep |

**Purpose:** Advanced backtesting engine that validates the strategy using TCN predictions and RAG-based risk assessment.

**What It Does (Step-by-Step):**
1. **Setup:**
   - Initializes 10-stock universe
   - Loads pre-trained TCN model
   - Connects to ChromaDB vector store
2. **Simulation Loop (60-day chunks):**
   - Calculate current portfolio indicators
   - Generate TCN prediction for next 60 days
3. **RAG Integration:**
   - Query `rag_knowledge_base.json` via ChromaDB
   - Find historically similar market scenarios
   - Extract outcomes from similar scenarios
4. **Risk-Reward Calculation:**
   - Calculate confidence intervals from historical outcomes
   - Compute Risk-Reward (RR) Ratio
5. **Decision Making:**
   - If RR > 2.0: Enter or Hold position with HRP weights
   - If RR < 2.0: Exit to cash (too risky)
6. **Daily Value Tracking:** Records portfolio value every day
7. **Performance Report:**
   - Same comprehensive metrics as rebalanced_predictive_model.py
8. **Database Storage:** Saves results to `portfolio_backtest_tcn.db`
9. **Forecast Visualization:** Plots historical curve + future prediction with RAG confidence bands

**Key Functions:**
- `predict_portfolio_return()` - TCN model inference
- `get_rag_metrics()` - RAG similarity search
- `optimize_weights_hrp()` - HRP weight allocation
- `generate_backtest_performance_report()` - Comprehensive metrics
- `plot_portfolio_forecast()` - Visualization with RAG bounds
- `run_backtest()` - Main simulation loop

**Why Created:** To validate strategy performance using ML predictions and historical pattern matching before live deployment.

---

## 🧠 **2. MODEL TRAINING SCRIPTS**

---

### **`train_generalized_model.py`**
| Property | Value |
|----------|-------|
| **Size** | 37 KB (881 lines) |
| **Type** | Base Model Training |
| **Status** | ✅ ESSENTIAL - Keep |

**Purpose:** Trains the generalized TCN model from scratch using thousands of portfolio combinations.

**What It Does (Step-by-Step):**
1. **Stock Universe:** Defines 30+ candidate stocks
2. **Portfolio Generation:** Creates ~7,851 unique 5-stock combinations (C(30,5))
3. **Data Processing (per portfolio):**
   - Fetch historical data for all 5 stocks
   - Align to common trading dates
   - Simulate portfolio with HRP rebalancing every 60 days
   - Calculate 31 technical indicators on portfolio series
4. **Sequence Creation:**
   - Input: 120-day sliding windows of normalized indicators
   - Target: 60-day forward return factor
   - Saves to `temp_training_data/` as .npz files
5. **Model Architecture (TCN):**
   - Temporal Convolutional Network with dilated convolutions
   - Dropout and batch normalization for regularization
   - Output: Single value (predicted return multiplier)
6. **Training:**
   - tf.data pipeline for efficient batch loading
   - Huber loss function (robust to outliers)
   - ReduceLROnPlateau and EarlyStopping callbacks
7. **Output:** Saves best model as `generalized_tcn_enhanced.keras`

**Key Functions:**
- `optimize_weights()` - NumPy-optimized HRP
- `calculate_indicators()` - Full 31-feature computation
- `create_dataset()` - Sliding window generation
- `process_portfolio()` - Worker function for multiprocessing
- `get_dataset()` - tf.data pipeline builder

**Why Created:** To create a generalized model that can predict returns for ANY portfolio composition.

---

### **`train_model_with_new_portfolios.py`**
| Property | Value |
|----------|-------|
| **Size** | 28 KB (737 lines) |
| **Type** | Fine-Tuning Script |
| **Status** | ✅ ESSENTIAL - Keep |

**Purpose:** Fine-tunes the existing model with new 10-stock portfolio data for better performance on specific universe.

**What It Does (Step-by-Step):**
1. **New Portfolio Generation:**
   - Uses specific 10-stock universe (BPCL, ABCAPITAL, BAJFINANCE, etc.)
   - Creates 252 portfolios (C(10,5) = 252)
2. **Data Processing:**
   - Same HRP simulation and indicator calculation
   - Saves to `new_portfolio_training_data/`
3. **Data Combination:**
   - Loads existing ~7,851 portfolios from `temp_training_data/`
   - Merges with 252 new portfolios
4. **Transfer Learning:**
   - Loads existing model (`generalized_lstm_model_improved.keras`)
   - Reduces learning rate to 30% for gentle fine-tuning
5. **Training:**
   - Combined dataset (old + new)
   - Same callbacks and loss function
6. **Output:** Saves as `generalized_lstm_model_improved_v2.keras`

**Key Functions:**
- `generate_new_training_data()` - Creates 252 new portfolios
- `get_combined_dataset()` - Merges old and new data
- `train_model_with_combined_data()` - Fine-tuning logic

**Why Created:** To adapt the model to a new stock universe without losing knowledge from original training.

---

### **`continue_training_model.py`**
| Property | Value |
|----------|-------|
| **Size** | 11 KB (299 lines) |
| **Type** | Training Utility |
| **Status** | 📦 Useful - Keep |

**Purpose:** Continues training an existing model for additional epochs without starting from scratch.

**What It Does:**
1. Loads existing model checkpoint (`generalized_lstm_model.keras`)
2. Loads training data from `temp_training_data/`
3. Reduces learning rate by 50% for stable fine-tuning
4. Continues training with same callbacks
5. Saves improved model as `generalized_lstm_model_improved.keras`
6. Generates comparison between original and improved model
7. Creates training loss plot

**Key Functions:**
- `get_dataset()` - tf.data pipeline
- `main()` - Training orchestration

**Why Created:** To incrementally improve model performance without full retraining.

---

### **`turbo_train.py`**
| Property | Value |
|----------|-------|
| **Size** | 3 KB (89 lines) |
| **Type** | Training Wrapper |
| **Status** | 📦 Useful - Keep |

**Purpose:** Speed-optimized training wrapper for faster iteration.

**What It Does:**
1. Displays speed optimizations being applied:
   - Maximum CPU cores
   - Larger batch sizes (16 vs 8)
   - Reduced epochs (30 vs 50)
   - Aggressive early stopping (patience=2)
2. Launches `train_generalized_model.py` as subprocess
3. Reports total training time

**Estimated Times:**
- Data Generation: ~15-30 min
- Model Training: ~10-20 min
- Total: ~25-50 min (vs 60-120 min original)

**Why Created:** For faster model iteration during development.

---

## 🔍 **3. RAG (RETRIEVAL-AUGMENTED GENERATION) SYSTEM**

---

### **`create_rag_data.py`**
| Property | Value |
|----------|-------|
| **Size** | 17 KB (410 lines) |
| **Type** | RAG Knowledge Base Builder |
| **Status** | ✅ ESSENTIAL - Keep |

**Purpose:** Constructs the RAG knowledge base from historical portfolio scenarios.

**What It Does (Step-by-Step):**
1. **Stock Universe:** Same 30+ stocks as training
2. **Portfolio Processing:**
   - Simulates each portfolio with HRP rebalancing
   - At each rebalance point, captures:
     - Technical indicators (RSI, MACD, Alligator, Volatility)
     - HRP weights
     - Actual 60-day forward return (outcome)
3. **Narrative Generation:**
   - Converts technical states to natural language
   - Example: "RSI > 70" → "Market is Overbought"
4. **Embedding Creation:**
   - Creates vector representation from indicators + weights + outcome
   - 12-dimensional embedding vector
5. **Storage:**
   - JSON file: `rag_knowledge_base.json` (285 MB)
   - Vector DB: ChromaDB collection `portfolio_scenarios`

**Key Functions:**
- `calculate_indicators()` - Simplified indicator set
- `generate_narrative()` - Text description generator
- `main()` - Orchestration

**Why Created:** To enable risk assessment by finding similar historical scenarios when making predictions.

---

### **`add_new_portfolios_to_rag.py`**
| Property | Value |
|----------|-------|
| **Size** | 20 KB (471 lines) |
| **Type** | RAG Updater |
| **Status** | ✅ ESSENTIAL - Keep |

**Purpose:** Adds scenarios from new 10-stock universe to the existing RAG knowledge base.

**What It Does:**
1. Loads existing `rag_knowledge_base.json`
2. Generates 252 new portfolios from 10 stocks
3. Simulates each with HRP rebalancing
4. Creates narratives and embeddings for each rebalance point
5. Appends new entries to JSON file
6. Updates ChromaDB collection with new vectors

**Key Functions:**
- `optimize_weights()` - HRP implementation
- `calculate_indicators()` - Technical indicators
- `generate_narrative()` - Text generation
- `create_embedding_vector()` - Vector creation

**Why Created:** To expand RAG coverage with new stock combinations.

---

### **`rag_prediction_simple.py`**
| Property | Value |
|----------|-------|
| **Size** | 3 KB (87 lines) |
| **Type** | RAG Query Interface |
| **Status** | 📦 Useful - Keep |

**Purpose:** Simple CLI interface to query the RAG system for risk assessment.

**What It Does:**
1. Loads RAG knowledge base and ChromaDB
2. Takes current market indicators as input
3. Creates embedding vector
4. Queries ChromaDB for top-k similar historical scenarios
5. Returns scenarios with distances and outcomes

**Key Functions:**
- `load_rag_system()` - Initialize RAG components
- `create_embedding_vector()` - Convert indicators to vector
- `find_similar_scenarios()` - ChromaDB similarity search

**Why Created:** Simplified interface for RAG queries during backtesting and prediction.

---

## 🗄️ **4. DATABASE & ANALYSIS TOOLS**

---

### **`analysis_db.py`**
| Property | Value |
|----------|-------|
| **Size** | 21 KB (lines not counted) |
| **Type** | Database-Centric Analysis |
| **Status** | 🗑️ Superseded - Can Delete |

**Purpose:** Intermediate version that saves analysis results directly to SQLite instead of CSV.

**What It Does:**
- Contains `save_to_database()` and `get_db_connection()` functions
- Writes to `portfolio_analysis.db`
- Functionality merged into `analysis.py`

**Why Created:** Transition step from CSV to SQLite storage. Now superseded by `analysis.py`.

---

### **`view_database.py`**
| Property | Value |
|----------|-------|
| **Size** | 4 KB (142 lines) |
| **Type** | Database Viewer |
| **Status** | 📦 Useful - Keep |

**Purpose:** Interactive CLI tool to inspect database contents without external tools.

**What It Does:**
1. Lists all tables with row counts
2. Displays top 5 performing portfolios
3. Shows sample data from key tables
4. Offers menu options:
   - Export table to CSV
   - View specific table
   - Exit

**Key Functions:**
- `list_tables()` - Show all tables
- `view_table()` - Display table contents
- `query_top_portfolios()` - Best performers
- `export_table_to_csv()` - CSV export

**Why Created:** To easily inspect database contents during development and debugging.

---

### **`fetch_portfolio_predictions.py`**
| Property | Value |
|----------|-------|
| **Size** | 4 KB (114 lines) |
| **Type** | Prediction History Viewer |
| **Status** | 📦 Useful - Keep |

**Purpose:** Interactive tool to view past predictions made by the live model.

**What It Does:**
1. Connects to `portfolio_analysis.db`
2. Queries `portfolio_predictions` table
3. Displays formatted table:
   - ID, Rebalance Date, Expiry Date
   - Start Value, Predicted Factor, Target Value
4. Allows viewing detailed prediction:
   - Asset weights
   - Risk metrics
   - Correlation matrix

**Key Functions:**
- `list_predictions()` - Show all predictions
- `view_prediction_details()` - Detailed view by ID

**Why Created:** To review and analyze stored predictions.

---

### **`filter_portfolios.py`**
| Property | Value |
|----------|-------|
| **Size** | 1 KB (34 lines) |
| **Type** | Data Filter |
| **Status** | 📦 Useful - Keep |

**Purpose:** Filters portfolio results to find "safe" portfolios with acceptable drawdown.

**What It Does:**
1. Reads `report_rebalanced_portfolios.csv`
2. Filters where Max Drawdown > -0.4 (less than 40% loss)
3. Saves filtered results to `filtered_portfolios_max_drawdown_gt_minus_0_4.csv`

**Why Created:** To identify safer portfolio choices from analysis results.

---

### **`inspect_database.py`**
| Property | Value |
|----------|-------|
| **Size** | 1.3 KB (47 lines) |
| **Type** | Schema Inspector |
| **Status** | 🧪 Debug - Can Delete |

**Purpose:** Detailed database schema inspection.

**What It Does:**
1. Lists all tables
2. For each table:
   - Shows row count
   - Lists all columns with types
   - Marks primary keys

**Why Created:** Debugging tool to verify database schema.

---

### **`check_db.py`**
| Property | Value |
|----------|-------|
| **Size** | 487 bytes (18 lines) |
| **Type** | Quick Check |
| **Status** | 🧪 Debug - Can Delete |

**Purpose:** Quick database connectivity check.

**What It Does:**
1. Connects to `portfolio_analysis.db`
2. Lists all tables with row counts

**Why Created:** Quick verification that database is accessible.

---

### **`check_db_schema.py`**
| Property | Value |
|----------|-------|
| **Size** | 946 bytes (34 lines) |
| **Type** | Schema Check |
| **Status** | 🧪 Debug - Can Delete |

**Purpose:** Verify `portfolio_predictions` table schema.

**What It Does:**
1. Lists all tables
2. Shows columns in `portfolio_predictions`
3. Displays first 5 rows as sample

**Why Created:** Debugging schema issues after migrations.

---

## 🔧 **5. UTILITY & ALGORITHM SCRIPTS**

---

### **`calculate_hrp.py`**
| Property | Value |
|----------|-------|
| **Size** | 3 KB (87 lines) |
| **Type** | Algorithm Implementation |
| **Status** | 📦 Useful - Keep |

**Purpose:** Standalone HRP (Hierarchical Risk Parity) calculator.

**What It Does:**
1. Takes a correlation matrix as input
2. Converts correlation to distance matrix
3. Performs hierarchical clustering (single linkage)
4. Calculates quasi-diagonal order
5. Applies recursive bisection for weight allocation
6. Saves results to `hrp_clustering_weights.csv`

**Key Functions:**
- `get_quasi_diag()` - Tree seriation
- `get_hrp_weights()` - Recursive weight calculation

**Why Created:** Modular HRP implementation for testing and reuse.

---

### **`calculate_clustering_weights.py`**
| Property | Value |
|----------|-------|
| **Size** | 7 KB (180 lines) |
| **Type** | Educational Script |
| **Status** | 📦 Useful - Keep |

**Purpose:** Demonstrates HRP algorithm step-by-step with detailed output.

**What It Does:**
1. Uses hardcoded 5x5 correlation matrix
2. Prints each step:
   - Distance Matrix calculation
   - Linkage matrix
   - Quasi-diagonalization order
   - Recursive bisection with cluster variances
3. Compares HRP vs Equal vs Inverse Variance weights
4. Generates dendrogram visualization

**Output:**
- Console: Detailed step-by-step breakdown
- File: `clustering_dendrogram.png`

**Why Created:** To understand and verify the HRP algorithm.

---

### **`hrp_weights_simple.py`**
| Property | Value |
|----------|-------|
| **Size** | 3 KB (lines not counted) |
| **Type** | Simplified HRP |
| **Status** | 📦 Useful - Keep |

**Purpose:** Lightweight version of HRP calculator.

**What It Does:**
- Simplified HRP implementation
- Quick weight calculations
- Less verbose output

**Why Created:** For quick testing and validation.

---

### **`cleanup_temp_data.py`**
| Property | Value |
|----------|-------|
| **Size** | 859 bytes (27 lines) |
| **Type** | Maintenance Script |
| **Status** | 📦 Useful - Keep |

**Purpose:** Frees disk space by deleting temporary training data.

**What It Does:**
1. Checks if `temp_training_data/` exists
2. Waits 2 seconds for safety
3. Uses `shutil.rmtree()` to delete entire folder
4. Provides troubleshooting tips if deletion fails

**Why Created:** Training data can be 500+ MB; this frees space after training is complete.

---

### **`search.py`**
| Property | Value |
|----------|-------|
| **Size** | 2 KB (52 lines) |
| **Type** | Stock Search Tool |
| **Status** | 📦 Useful - Keep |

**Purpose:** Interactive tool to search for stock codes in NSE.csv.

**What It Does:**
1. Prompts for search term
2. Searches in `name` and `tradingsymbol` columns
3. Filters for EQUITY instrument type
4. Shows exact matches
5. Suggests similar names if no exact match

**Why Created:** To find instrument keys for new stocks.

---

### **`historical.py`**
| Property | Value |
|----------|-------|
| **Size** | 2 KB (67 lines) |
| **Type** | Data Fetcher |
| **Status** | 📦 Useful - Keep |

**Purpose:** Downloads complete history for a single stock.

**What It Does:**
1. Defines instrument key to fetch
2. Generates 10-year date chunks (API limitation)
3. Fetches all available candles from Upstox API
4. Saves to `historical_daily_data.csv`

**Key Functions:**
- `fetch_chunk()` - Single chunk API call
- `generate_date_ranges()` - Create safe date chunks
- `fetch_full_history()` - Orchestration

**Why Created:** To fetch complete historical data for analysis.

---

## 🧪 **6. TESTING & DEBUGGING SCRIPTS**

---

### **`test_hrp_fix.py`**
| Property | Value |
|----------|-------|
| **Size** | 5 KB (153 lines) |
| **Type** | Algorithm Test |
| **Status** | 🧪 Debug - Can Delete |

**Purpose:** Tests HRP algorithm against known correlation matrix.

**What It Does:**
1. Uses hardcoded 5x5 correlation matrix
2. Runs HRP algorithm step-by-step
3. Prints intermediate values (variances, alphas)
4. Verifies weights sum to 1.0
5. Compares against expected values

**Why Created:** To debug and verify HRP implementation.

---

### **`test_optimization.py`**
| Property | Value |
|----------|-------|
| **Size** | 10 KB (254 lines) |
| **Type** | Algorithm Test |
| **Status** | 🧪 Debug - Can Delete |

**Purpose:** Tests `optimize_weights()` function with random data.

**What It Does:**
1. Copies `optimize_weights()` function locally
2. Generates random return data (120 days, 5 assets)
3. Runs optimization
4. Checks if weights are equal (indicates bug)
5. Documents suspected logic issues with comments

**Why Created:** To debug why HRP was producing equal weights.

---

### **`test_single_portfolio.py`**
| Property | Value |
|----------|-------|
| **Size** | 597 bytes (20 lines) |
| **Type** | Parsing Test |
| **Status** | 🧪 Debug - Can Delete |

**Purpose:** Tests portfolio string parsing from CSV reports.

**What It Does:**
1. Loads `report_rebalanced_portfolios.csv`
2. Takes first portfolio string
3. Parses comma-separated symbols
4. Prints parsed result

**Why Created:** To verify CSV parsing logic.

---

### **`test_tf.py`**
| Property | Value |
|----------|-------|
| **Size** | 71 bytes (3 lines) |
| **Type** | Environment Check |
| **Status** | 🧪 Debug - Can Delete |

**Purpose:** Verifies TensorFlow installation.

**What It Does:**
```python
import tensorflow as tf
print("TensorFlow version:", tf.__version__)
```

**Why Created:** Quick environment verification.

---

### **`check_data.py`**
| Property | Value |
|----------|-------|
| **Size** | 1 KB (35 lines) |
| **Type** | Data Verification |
| **Status** | 🧪 Debug - Can Delete |

**Purpose:** Inspects training data file format.

**What It Does:**
1. Loads `temp_training_data/data_0.npz`
2. Prints X shape (sequences, lookback, features)
3. Prints y shape (number of targets)
4. Shows sample target values and statistics

**Why Created:** To verify training data generation worked correctly.

---

### **`inspect_training_data.py`**
| Property | Value |
|----------|-------|
| **Size** | 2 KB (54 lines) |
| **Type** | Data Inspection |
| **Status** | 🧪 Debug - Can Delete |

**Purpose:** Advanced training data inspection across multiple files.

**What It Does:**
1. Loads sample training file
2. Displays detailed shape information
3. Shows data types and memory usage
4. Checks multiple files (data_0, data_100, data_500, etc.)
5. Provides interpretation of data format

**Why Created:** To verify data generation across all portfolios.

---

### **`inspect_csv.py`**
| Property | Value |
|----------|-------|
| **Size** | 220 bytes (10 lines) |
| **Type** | CSV Debug |
| **Status** | 🧪 Debug - Can Delete |

**Purpose:** Quick CSV file inspection.

**What It Does:**
1. Reads `report_rebalanced_portfolios.csv`
2. Prints column names
3. Shows first 2 rows

**Why Created:** To check CSV formatting issues.

---

## 📁 **7. OBSOLETE & BACKUP FILES**

---

### **`analyse2.py`**
| Property | Value |
|----------|-------|
| **Size** | 12 KB (227 lines) |
| **Type** | Older Version |
| **Status** | 🗑️ Obsolete - Can Delete |

**Purpose:** Earlier version of analysis script with different selection criteria.

**Key Differences from `analysis.py`:**
- Uses volume spike detection (50x previous day)
- Single stock analysis (not portfolio combinations)
- No HRP optimization
- Different date range (2019-2025)
- Outputs to CSV files (not database)

**Why Created:** Early exploration of stock selection criteria. Superseded by `analysis.py`.

---

### **`analysis.py.backup`**
| Property | Value |
|----------|-------|
| **Size** | 15 KB |
| **Type** | Backup File |
| **Status** | 🗑️ Backup - Can Delete |

**Purpose:** Safety backup of `analysis.py` before modifications.

**Why Created:** Manual backup before making changes.

---

### **`copy previous.py`**
| Property | Value |
|----------|-------|
| **Size** | 17 KB (391 lines) |
| **Type** | Historical Copy |
| **Status** | 🗑️ Obsolete - Can Delete |

**Purpose:** Earlier version of portfolio analysis with different optimization.

**Key Differences:**
- Uses SLSQP optimization (Maximize Sharpe) instead of HRP
- 60-day lookback (not 120-day)
- Different stock universe
- No database persistence

**Why Created:** Reference copy of previous approach. No longer needed.

---

### **`requirements_rag.txt`**
| Property | Value |
|----------|-------|
| **Size** | 108 bytes (7 lines) |
| **Type** | Dependencies |
| **Status** | 📦 Reference - Keep |

**Contents:**
```
flask==3.0.0
google-generativeai==0.3.2
chromadb==0.4.22
numpy==1.24.3
pandas==2.0.3
requests==2.31.0
```

**Purpose:** Python packages needed for RAG functionality.

**Why Created:** To document RAG dependencies.

---

## 💾 **8. DATA FILES**

---

### **Databases**

#### **`database/portfolio_analysis.db`**
| Property | Value |
|----------|-------|
| **Size** | 9.9 MB |
| **Type** | SQLite Database |
| **Status** | ✅ ESSENTIAL - Keep |

**Tables:**
| Table | Description |
|-------|-------------|
| `report_overall` | Summary metrics for all tested portfolios |
| `report_last_year` | Recent 1-year performance |
| `report_quarterly_avg` | Quarterly average statistics |
| `report_outperforming_combinations` | Best performing portfolios |
| `report_rebalanced_portfolios` | Detailed logs of every rebalance event |
| `report_rejected_portfolios` | Combinations that failed criteria |
| `report_yearly_profits` | Year-by-year profit breakdown |
| `portfolio_predictions` | Live predictions with portfolio signatures |

---

#### **`database/portfolio_backtest_tcn.db`**
| Property | Value |
|----------|-------|
| **Size** | 16 KB |
| **Type** | SQLite Database |
| **Status** | ✅ ESSENTIAL - Keep |

**Contains:** Backtest results from `portfolio_backtest_optimized_v2.py`
- Entry/exit dates and reasons
- Predicted vs actual returns
- Risk-reward ratios

---

### **Model Files**

#### **`generalized_tcn_enhanced.keras`**
| Property | Value |
|----------|-------|
| **Size** | 11.7 MB |
| **Type** | Keras Model |
| **Status** | ✅ ESSENTIAL - Keep |

**Description:** Latest TCN model trained on 7,851+ portfolios. Best performing model.

---

#### **`generalized_lstm_model_improved_v2.keras`**
| Property | Value |
|----------|-------|
| **Size** | 10.3 MB |
| **Type** | Keras Model |
| **Status** | ✅ ESSENTIAL - Keep |

**Description:** LSTM model fine-tuned with new 10-stock universe data.

---

#### **`generalized_lstm_model_improved.keras`**
| Property | Value |
|----------|-------|
| **Size** | 10.3 MB |
| **Type** | Keras Model |
| **Status** | 🗑️ Superseded - Can Delete |

**Description:** Previous improved version. Superseded by v2.

---

#### **`generalized_lstm_model.keras`**
| Property | Value |
|----------|-------|
| **Size** | 10.3 MB |
| **Type** | Keras Model |
| **Status** | 🗑️ Superseded - Can Delete |

**Description:** Original LSTM model. Superseded by improved versions.

---

### **RAG Data**

#### **`rag_knowledge_base.json`**
| Property | Value |
|----------|-------|
| **Size** | 285 MB |
| **Type** | JSON File |
| **Status** | ✅ ESSENTIAL - Keep |

**Description:** The "brain" of the RAG system. Contains thousands of historical scenarios with:
- Date and portfolio composition
- Market narratives
- Technical indicators
- HRP weights
- Actual 60-day forward returns

---

### **Reference Data**

#### **`NSE.csv`**
| Property | Value |
|----------|-------|
| **Size** | 9.2 MB |
| **Type** | CSV File |
| **Status** | ✅ ESSENTIAL - Keep |

**Description:** Complete list of NSE stocks with:
- Trading Symbol
- Company Name
- Instrument Key
- Instrument Type

---

### **Generated Reports (CSV)**

All can be regenerated by running respective scripts:

| File | Size | Description |
|------|------|-------------|
| `report_overall.csv` | 58 KB | Overall portfolio metrics |
| `report_last_year.csv` | 58 KB | Recent 1-year performance |
| `report_quarterly_avg.csv` | 38 KB | Quarterly averages |
| `report_outperforming_combinations.csv` | 70 KB | Best portfolios |
| `report_rebalanced_portfolios.csv` | 1.3 MB | All rebalance events |
| `report_rejected_portfolios.csv` | 189 KB | Failed combinations |
| `report_yearly_profits.csv` | 1.4 MB | Year-by-year breakdown |
| `report_portfolio_*.csv` | Various | Specific portfolio reports |
| `filtered_portfolios_max_drawdown_gt_minus_0_4.csv` | 22 KB | Safe portfolio list |
| `hrp_clustering_weights.csv` | 358 bytes | HRP weight output |

**Status:** 📦 Generated - Can regenerate

---

## 📂 **9. DIRECTORIES**

---

### **`database/`**
| Items | Description | Status |
|-------|-------------|--------|
| 2 files | SQLite databases | ✅ ESSENTIAL |

---

### **`chroma_db/`**
| Items | Description | Status |
|-------|-------------|--------|
| 6 items | ChromaDB vector store | ✅ ESSENTIAL |

**Contains:** Vector embeddings for RAG similarity search

---

### **`temp_training_data/`**
| Items | Description | Status |
|-------|-------------|--------|
| 1,176 files | Training data (.npz) | 📦 Can Delete (after training) |

**Size:** ~500 MB
**Contents:** .npz files with X (features) and y (targets) for 7,851 portfolios

---

### **`new_portfolio_training_data/`**
| Items | Description | Status |
|-------|-------------|--------|
| 252 files | Fine-tuning data (.npz) | 📦 Can Delete (after training) |

**Contents:** Training data for 252 portfolios from 10-stock universe

---

### **`__pycache__/`**
| Items | Description | Status |
|-------|-------------|--------|
| 3 files | Python bytecode cache | 🗑️ Can Delete |

**Note:** Auto-regenerated by Python

---

### **`.vscode/`**
| Items | Description | Status |
|-------|-------------|--------|
| 1 file | VS Code settings | 📦 Optional |

---

### **`.venv/`**
| Description | Status |
|-------------|--------|
| Python virtual environment | 📦 Keep if active |

---

## 📊 **10. SUMMARY BY STATUS**

### ✅ **MUST KEEP (Essential)** - 15 items
1. `rebalanced_predictive_model.py` - Main prediction
2. `analysis.py` - Portfolio discovery
3. `portfolio_backtest_optimized_v2.py` - Backtesting
4. `train_generalized_model.py` - Base training
5. `train_model_with_new_portfolios.py` - Fine-tuning
6. `create_rag_data.py` - RAG builder
7. `add_new_portfolios_to_rag.py` - RAG updater
8. `database/` - All databases
9. `chroma_db/` - Vector store
10. `generalized_tcn_enhanced.keras` - Best model
11. `generalized_lstm_model_improved_v2.keras` - Fine-tuned model
12. `rag_knowledge_base.json` - RAG data
13. `NSE.csv` - Stock reference

### 📦 **USEFUL** - 15 items
1. `continue_training_model.py`
2. `turbo_train.py`
3. `rag_prediction_simple.py`
4. `view_database.py`
5. `fetch_portfolio_predictions.py`
6. `filter_portfolios.py`
7. `calculate_hrp.py`
8. `calculate_clustering_weights.py`
9. `hrp_weights_simple.py`
10. `cleanup_temp_data.py`
11. `search.py`
12. `historical.py`
13. `requirements_rag.txt`
14. All generated CSV reports
15. `.venv/`

### 🗑️ **CAN DELETE** - 25+ items
1. All `test_*.py` files (6 files)
2. All `check_*.py` and `inspect_*.py` files (6 files)
3. `analysis.py.backup`
4. `copy previous.py`
5. `analyse2.py`
6. `analysis_db.py`
7. `generalized_lstm_model.keras`
8. `generalized_lstm_model_improved.keras`
9. `temp_training_data/` directory
10. `new_portfolio_training_data/` directory
11. `__pycache__/` directory

### 💾 **DISK SPACE SAVINGS**
| Action | Space Saved |
|--------|-------------|
| Delete temp training data | ~500 MB |
| Delete new training data | ~50 MB |
| Delete old models | ~20 MB |
| Delete test scripts | ~50 KB |
| Delete backups/obsolete | ~50 KB |
| **TOTAL** | **~570 MB** |

---

## 🔄 **11. WORKFLOW DIAGRAM**

```
┌─────────────────────────────────────────────────────────────────┐
│                    PORTFOLIO DISCOVERY                           │
│   analysis.py → database/portfolio_analysis.db                  │
│   (Finds profitable portfolio combinations)                     │
└────────────────────────┬───────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                     MODEL TRAINING                               │
│   train_generalized_model.py → generalized_tcn_enhanced.keras   │
│   train_model_with_new_portfolios.py → _improved_v2.keras       │
│   (Trains TCN to predict 60-day returns)                        │
└────────────────────────┬───────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                   RAG KNOWLEDGE BASE                             │
│   create_rag_data.py → rag_knowledge_base.json + chroma_db/     │
│   (Builds historical scenario database)                         │
└────────────────────────┬───────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                      BACKTESTING                                 │
│   portfolio_backtest_optimized_v2.py → backtest_tcn.db          │
│   (Validates strategy with ML + RAG)                            │
└────────────────────────┬───────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                    LIVE PREDICTION                               │
│   rebalanced_predictive_model.py → portfolio_predictions        │
│   (Runs live portfolio with risk management)                    │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🎓 **12. KEY CONCEPTS**

### **HRP (Hierarchical Risk Parity)**
- Portfolio optimization method
- Allocates weights based on hierarchical clustering
- Reduces allocation to correlated assets
- Increases diversification

### **TCN (Temporal Convolutional Network)**
- Deep learning for time series
- Dilated convolutions for long-range dependencies
- Predicts 60-day forward returns

### **RAG (Retrieval-Augmented Generation)**
- Finds similar historical scenarios via vector similarity
- Provides confidence intervals from historical outcomes
- Used for risk assessment

### **Strategy Parameters**
| Parameter | Value |
|-----------|-------|
| Lookback Window | 120 days |
| Forward Window | 60 days |
| Rebalance Frequency | 60 days |
| Stop-Loss | 15% from entry |
| Transaction Cost | 0.1% per rebalance |
| Initial Capital | ₹100,000 (live) / ₹1,000,000 (training) |

---

**End of Document**
