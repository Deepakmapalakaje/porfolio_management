import sqlite3
import pandas as pd
import json
import os
from datetime import datetime

DATABASE_FILE = os.path.join("database", "portfolio_analysis.db")

def get_db_connection():
    if not os.path.exists(DATABASE_FILE):
        print(f"Database not found at {DATABASE_FILE}")
        return None
    return sqlite3.connect(DATABASE_FILE)

def list_predictions():
    conn = get_db_connection()
    if not conn: return

    try:
        query = "SELECT id, rebalance_date, expiry_date, portfolio_value_at_rebalance, predicted_return_factor, predicted_value_at_expiry FROM portfolio_predictions ORDER BY rebalance_date DESC"
        df = pd.read_sql_query(query, conn)
        
        if df.empty:
            print("No predictions found in database.")
            return

        print("\n" + "="*80)
        print("  PORTFOLIO PREDICTION HISTORY")
        print("="*80)
        print(f"{'ID':<5} | {'Rebalance Date':<15} | {'Expiry Date':<15} | {'Start Value':<12} | {'Factor':<8} | {'Target Value':<12}")
        print("-" * 80)
        
        for _, row in df.iterrows():
            print(f"{row['id']:<5} | {row['rebalance_date']:<15} | {row['expiry_date']:<15} | {row['portfolio_value_at_rebalance']:<12.2f} | {row['predicted_return_factor']:<8.4f} | {row['predicted_value_at_expiry']:<12.2f}")
        print("="*80)
        
    except Exception as e:
        print(f"Error fetching predictions: {e}")
    finally:
        conn.close()

def view_prediction_details(prediction_id):
    conn = get_db_connection()
    if not conn: return

    try:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM portfolio_predictions WHERE id = ?", (prediction_id,))
        row = cursor.fetchone()
        
        if not row:
            print(f"Prediction ID {prediction_id} not found.")
            return

        # Unpack row (based on schema)
        # id, rebalance_date, expiry_date, val_rebal, factor, val_expiry, weights, corr, risk, created
        
        print("\n" + "="*60)
        print(f"  PREDICTION DETAILS (ID: {row[0]})")
        print("="*60)
        print(f"Rebalance Date:       {row[1]}")
        print(f"Expiry Date:          {row[2]}")
        print(f"Value at Rebalance:   {row[3]:,.2f}")
        print(f"Predicted Factor:     {row[4]:.4f}")
        print(f"Target Value:         {row[5]:,.2f}")
        print(f"Created At:           {row[9]}")
        
        print("\n--- Asset Weights ---")
        weights = json.loads(row[6])
        for asset, weight in weights.items():
            print(f"  {asset:<15}: {weight:.4f}")
            
        print("\n--- Risk Metrics (at Rebalance) ---")
        risk = json.loads(row[8])
        for metric, value in risk.items():
            if isinstance(value, float):
                print(f"  {metric.replace('_', ' ').title():<20}: {value:.4f}")
            else:
                print(f"  {metric.replace('_', ' ').title():<20}: {value}")

        print("\n--- Correlation Matrix ---")
        corr = json.loads(row[7])
        corr_df = pd.DataFrame(corr)
        print(corr_df.round(2))
        print("="*60)

    except Exception as e:
        print(f"Error fetching details: {e}")
    finally:
        conn.close()

def main():
    while True:
        list_predictions()
        print("\nOptions:")
        print("1. View Details by ID")
        print("2. Exit")
        
        choice = input("\nEnter choice: ").strip()
        
        if choice == '1':
            pid = input("Enter Prediction ID: ").strip()
            if pid.isdigit():
                view_prediction_details(int(pid))
            else:
                print("Invalid ID.")
        elif choice == '2':
            break
        else:
            print("Invalid choice.")

if __name__ == "__main__":
    main()
