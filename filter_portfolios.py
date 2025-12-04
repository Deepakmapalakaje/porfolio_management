import pandas as pd
import sys

input_file = 'report_rebalanced_portfolios.csv'
output_file = 'filtered_portfolios_max_drawdown_gt_minus_0_4.csv'

try:
    df = pd.read_csv(input_file)
    
    # Clean column names (strip whitespace)
    df.columns = [c.strip() for c in df.columns]
    print(f"Columns found: {df.columns.tolist()}")
    
    # Find the Max Drawdown column
    drawdown_col = next((c for c in df.columns if 'Max Drawdown' in c), None)
    
    if not drawdown_col:
        print("Error: 'Max Drawdown' column not found.")
        sys.exit(1)
        
    print(f"Filtering using column: '{drawdown_col}'")
    
    # Filter
    # Ensure the column is numeric
    df[drawdown_col] = pd.to_numeric(df[drawdown_col], errors='coerce')
    
    filtered_df = df[df[drawdown_col] > -0.4]
    
    filtered_df.to_csv(output_file, index=False)
    print(f"Successfully created {output_file} with {len(filtered_df)} rows.")
    
except Exception as e:
    print(f"An error occurred: {e}")
