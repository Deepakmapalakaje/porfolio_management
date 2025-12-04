import pandas as pd
import numpy as np
from datetime import timedelta
import sys

# Import from train_generalized_model
sys.path.insert(0, r'c:\Users\Karthik M\Desktop\assignment')

# Load one portfolio
df_portfolios = pd.read_csv("report_rebalanced_portfolios.csv")
test_portfolio = df_portfolios['Portfolio'].iloc[0]

print(f"Testing portfolio: {test_portfolio}")
print(f"Type: {type(test_portfolio)}")

# Try to parse it
symbols = [s.strip() for s in test_portfolio.replace('"', '').split(',')]
print(f"Parsed symbols: {symbols}")
print(f"Number of symbols: {len(symbols)}")
