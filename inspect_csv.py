import pandas as pd

try:
    df = pd.read_csv('report_rebalanced_portfolios.csv')
    print("Columns:", df.columns.tolist())
    print("First 2 rows:")
    print(df.head(2))
except Exception as e:
    print(e)
