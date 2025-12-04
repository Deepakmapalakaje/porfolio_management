# Portfolio Management System with TCN + RAG

An advanced portfolio optimization and prediction system using:
- **Hierarchical Risk Parity (HRP)** for weight allocation
- **Temporal Convolutional Network (TCN)** for return prediction
- **Retrieval-Augmented Generation (RAG)** for historical risk assessment
- **Google Gemini AI** for portfolio analysis and recommendations

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

| Script | Purpose | Usage |
|--------|---------|-------|
| `train_generalized_model.py` | Train base TCN model | `python train_generalized_model.py` |
| `train_model_with_new_portfolios.py` | Fine-tune model with new data | `python train_model_with_new_portfolios.py` |

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
├── database/                          # SQLite databases
│   ├── portfolio_analysis.db          # Analysis results
│   └── portfolio_backtest_tcn.db      # Backtest results
│
├── chroma_db/                         # RAG vector store (generated)
├── rag_knowledge_base.json            # RAG data (generated)
│
├── generalized_lstm_model_improved.keras  # Trained TCN model
│
├── rebalanced_predictive_model.py     # Main production script
├── multi_portfolio_finder.py          # Portfolio discovery tool
├── analysis.py                        # Portfolio analysis
├── portfolio_backtest_optimized_v2.py # Backtesting engine
│
├── create_rag_data.py                 # Generate RAG database
├── rag_prediction_simple.py           # RAG helper functions
│
├── train_generalized_model.py         # Model training
├── train_model_with_new_portfolios.py # Model fine-tuning
│
├── requirements.txt                   # Python dependencies
├── README.md                          # This file
└── .gitignore                         # Git ignore rules
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
Ensure `generalized_lstm_model_improved.keras` exists. If not, run `python train_generalized_model.py`.

### API Rate Limits
The scripts include rate limiting (0.1s between requests). If you get 429 errors, increase the delay in `fetch_historical_data()`.

### Large File Warning
The `chroma_db/` and `rag_knowledge_base.json` files are large (50MB+). They are excluded from git and must be regenerated locally.

---

## License

This project is for educational and personal use.

---

## Author

Deepak Mapalakaje
