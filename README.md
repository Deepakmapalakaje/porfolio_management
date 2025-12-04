# Portfolio Management System with TCN + RAG

An advanced portfolio optimization and prediction system using:
- **Hierarchical Risk Parity (HRP)** for weight allocation
- **Temporal Convolutional Network (TCN)** for return prediction
- **Retrieval-Augmented Generation (RAG)** for historical risk assessment
- **Google Gemini AI** for portfolio analysis and recommendations

---

## 📚 Documentation (Read First!)

To understand the project in detail, read these documentation files:

| Document | Description |
|----------|-------------|
| **[FILES_SUMMARY.md](FILES_SUMMARY.md)** | Complete summary of ALL files in the project with purpose and status |
| **[IMPORTANT_FILES_REPORT.md](IMPORTANT_FILES_REPORT.md)** | Detailed technical report on `rebalanced_predictive_model.py` and `portfolio_backtest_optimized_v2.py` |
| **[MULTI_PORTFOLIO_FINDER_REPORT.md](MULTI_PORTFOLIO_FINDER_REPORT.md)** | How the multi-portfolio finder works |

---

## Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/Deepakmapalakaje/porfolio_management.git
cd porfolio_management
```

### 2. Create Virtual Environment (Recommended)
```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux/Mac
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure API Keys

Edit the following files and add your API keys:

**`rebalanced_predictive_model.py`** (Lines 28-41):
```python
ACCESS_TOKEN = "your-upstox-api-token"  # Get from Upstox Developer Portal
GEMINI_API_KEY = "your-gemini-api-key"  # Get from https://aistudio.google.com/app/apikey
```

**`multi_portfolio_finder.py`** (Lines 37-47):
```python
ACCESS_TOKEN = "your-upstox-api-token"
GEMINI_API_KEY = "your-gemini-api-key"
```

### 5. Generate RAG Data (First Time Only)

The RAG system requires a vector database. Generate it by running:

```bash
python create_rag_data.py
```

This creates:
- `chroma_db/` - Vector database for similarity search
- `rag_knowledge_base.json` - Historical scenarios data

**Note:** This process may take 15-30 minutes depending on your system.

---

## 🤖 Model Training Guide

This project uses 3 trained models. Here's the correct training sequence:

### Step 1: Base Model
**Output:** `generalized_lstm_model.keras`  
**Script:** `train_generalized_model.py`

```bash
python train_generalized_model.py
```

**What it does:**
- Fetches historical data for 27 stocks
- Generates training data for all C(27,5) portfolio combinations
- Saves data to `temp_training_data/` folder
- Trains TCN model (120-day lookback → 60-day prediction)
- Saves as `generalized_lstm_model.keras`

**Training time:** ~2-4 hours

---

### Step 2: Improved Model
**Output:** `generalized_lstm_model_improved.keras`  
**Script:** `continue_training_model.py`

```bash
python continue_training_model.py
```

**What it does:**
- Loads `generalized_lstm_model.keras`
- Continues training with same data for 100 more epochs
- Uses lower learning rate for fine-tuning
- Saves as `generalized_lstm_model_improved.keras`

**Training time:** ~1-2 hours

---

### Step 3: V2 Model (with new stocks)
**Output:** `generalized_lstm_model_improved_v2.keras`  
**Script:** `train_model_with_new_portfolios.py`

```bash
python train_model_with_new_portfolios.py
```

**What it does:**
- Loads `generalized_lstm_model_improved.keras`
- Adds 10 NEW stocks (SBIN, KOTAKBANK, AXISBANK, etc.)
- Generates additional training data to `new_portfolio_training_data/`
- Combines old + new data for training
- Saves as `generalized_lstm_model_improved_v2.keras`

**Training time:** ~2-3 hours

---

### Quick Shortcut: Turbo Train
**Script:** `turbo_train.py`

```bash
python turbo_train.py
```

**What it does:**
- Simply runs `train_generalized_model.py` with a nice wrapper
- Shows timing and progress information
- Use this if you just want to run Step 1 easily

---

### Training Order (Required)

```
1. train_generalized_model.py      → generalized_lstm_model.keras
2. continue_training_model.py      → generalized_lstm_model_improved.keras
3. train_model_with_new_portfolios.py → generalized_lstm_model_improved_v2.keras
```

**Note:** Each step depends on the previous model. You must train in order!

---

## Main Scripts

### Production Scripts

| Script | Purpose | Usage |
|--------|---------|-------|
| `rebalanced_predictive_model.py` | Live portfolio management with predictions | `python rebalanced_predictive_model.py` |
| `multi_portfolio_finder.py` | Find top 10 portfolios from 33 stocks | `python multi_portfolio_finder.py` |
| `analysis.py` | Analyze all portfolio combinations | `python analysis.py` |

### Backtesting

| Script | Purpose | Usage |
|--------|---------|-------|
| `portfolio_backtest_optimized_v2.py` | Historical backtest with TCN+RAG | `python portfolio_backtest_optimized_v2.py` |

### Training Scripts

| Script | Purpose | Output Model |
|--------|---------|--------------|
| `train_generalized_model.py` | Step 1: Train base TCN model | `generalized_lstm_model.keras` |
| `continue_training_model.py` | Step 2: Continue training for improvement | `generalized_lstm_model_improved.keras` |
| `train_model_with_new_portfolios.py` | Step 3: Add new stocks + fine-tune | `generalized_lstm_model_improved_v2.keras` |
| `turbo_train.py` | Wrapper for Step 1 with timing | (runs train_generalized_model.py) |


### RAG System

| Script | Purpose | Usage |
|--------|---------|-------|
| `create_rag_data.py` | Create RAG knowledge base | `python create_rag_data.py` |
| `add_new_portfolios_to_rag.py` | Add new data to RAG | `python add_new_portfolios_to_rag.py` |
| `rag_prediction_simple.py` | RAG helper functions | (imported by other scripts) |

---

## Project Structure

```
porfolio_management/
├── 📁 database/                          # SQLite databases
│   ├── portfolio_analysis.db             # Analysis results
│   └── portfolio_backtest_tcn.db         # Backtest results
│
├── 📁 chroma_db/                         # RAG vector store (generated)
├── 📄 rag_knowledge_base.json            # RAG data (generated)
│
├── 🤖 generalized_lstm_model.keras       # Base TCN model
├── 🤖 generalized_lstm_model_improved.keras  # Improved model (recommended)
│
├── 📄 FILES_SUMMARY.md                   # All files documentation
├── 📄 IMPORTANT_FILES_REPORT.md          # Key files technical report
├── 📄 MULTI_PORTFOLIO_FINDER_REPORT.md   # Portfolio finder documentation
│
├── 🐍 rebalanced_predictive_model.py     # Main production script
├── 🐍 multi_portfolio_finder.py          # Portfolio discovery tool
├── 🐍 analysis.py                        # Portfolio analysis
├── 🐍 portfolio_backtest_optimized_v2.py # Backtesting engine
│
├── 🐍 create_rag_data.py                 # Generate RAG database
├── 🐍 rag_prediction_simple.py           # RAG helper functions
│
├── 🐍 train_generalized_model.py         # Model training
├── 🐍 train_model_with_new_portfolios.py # Model fine-tuning
├── 🐍 turbo_train.py                     # Fast training
│
├── 📄 requirements.txt                   # Python dependencies
├── 📄 README.md                          # This file
└── 📄 .gitignore                         # Git ignore rules
```

---

## Configuration

### Strategy Parameters (Default)

| Parameter | Value | Description |
|-----------|-------|-------------|
| Lookback Window | 120 days | Historical data for indicators |
| Forward Window | 60 days | Prediction horizon |
| Rebalance Frequency | 60 days | How often to reoptimize |
| Stop-Loss | 15% | Exit if drawdown exceeds |
| Transaction Cost | 0.1% | Cost per trade |
| Risk-Free Rate | 6.5% | For Sharpe/Sortino calculation |
| Initial Capital | ₹100,000 | Starting investment |

### Modifying Stock Universe

Edit the `STOCK_UNIVERSE` or `CHOSEN_PORTFOLIO` dictionary in the respective files to add/remove stocks.

---

## Output Files

### Reports Generated

- `portfolio_forecast_YYYYMMDD.png` - Prediction visualization
- `portfolio_comparison_YYYYMMDD.png` - Performance comparison
- `report_*.csv` - Various analysis reports

### Database Tables

**portfolio_analysis.db:**
- Portfolio predictions and analysis results

**portfolio_backtest_tcn.db:**
- Historical backtest results with TCN+RAG metrics

---

## Troubleshooting

### "RAG system failed to load"
Run `python create_rag_data.py` to generate the vector database.

### "TCN model not found"
Train models in this order:
```bash
python train_generalized_model.py
python train_model_with_new_portfolios.py
```

### API Rate Limits
The scripts include rate limiting (0.1s between requests). If you get 429 errors, increase the delay in `fetch_historical_data()`.

### Large File Warning
The `chroma_db/` and `rag_knowledge_base.json` files are large (50MB+). They are excluded from git and must be regenerated locally using `python create_rag_data.py`.

---

## License

This project is for educational and personal use.

---

## Author

Deepak Mapalakaje

