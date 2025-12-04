import sqlite3
import os

DATABASE_FILE = os.path.join("database", "portfolio_analysis.db")

def check_schema():
    if not os.path.exists(DATABASE_FILE):
        print("Database not found.")
        return

    conn = sqlite3.connect(DATABASE_FILE)
    cursor = conn.cursor()
    
    print("Tables:")
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    print(cursor.fetchall())
    
    print("\nColumns in portfolio_predictions:")
    cursor.execute("PRAGMA table_info(portfolio_predictions)")
    columns = cursor.fetchall()
    for col in columns:
        print(col)
        
    print("\nSample Data (first 5 rows):")
    cursor.execute("SELECT id, rebalance_date, portfolio_assets, portfolio_value_at_rebalance FROM portfolio_predictions LIMIT 5")
    rows = cursor.fetchall()
    for row in rows:
        print(row)

    conn.close()

if __name__ == "__main__":
    check_schema()
