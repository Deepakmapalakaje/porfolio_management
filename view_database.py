"""
Database Viewer - Query and view portfolio analysis data from SQLite database
"""

import sqlite3
import pandas as pd
import os

DATABASE_FILE = os.path.join("database", "portfolio_analysis.db")

def get_db_connection():
    """Get database connection"""
    if not os.path.exists(DATABASE_FILE):
        print(f"[ERROR] Database file not found: {DATABASE_FILE}")
        return None
    return sqlite3.connect(DATABASE_FILE)

def list_tables():
    """List all tables in the database"""
    conn = get_db_connection()
    if not conn:
        return
    
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    
    print("="*70)
    print("  DATABASE TABLES")
    print("="*70)
    
    for table in tables:
        table_name = table[0]
        cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
        row_count = cursor.fetchone()[0]
        print(f"  {table_name}: {row_count} rows")
    
    conn.close()
    print("="*70)

def view_table(table_name, limit=10):
    """View contents of a specific table"""
    conn = get_db_connection()
    if not conn:
        return
    
    try:
        df = pd.read_sql_query(f"SELECT * FROM {table_name} LIMIT {limit}", conn)
        print(f"\n[TABLE] {table_name} (showing first {limit} rows)")
        print("="*70)
        print(df.to_string())
        print("="*70)
    except Exception as e:
        print(f"[ERROR] Failed to read table '{table_name}': {e}")
    finally:
        conn.close()

def query_top_portfolios(limit=5):
    """Query top performing portfolios"""
    conn = get_db_connection()
    if not conn:
        return
    
    try:
        query = """
        SELECT * FROM rebalanced_portfolios 
        ORDER BY "Overall Profit (%)" DESC 
        LIMIT ?
        """
        df = pd.read_sql_query(query, conn, params=(limit,))
        
        print(f"\n[QUERY] Top {limit} Portfolios by Overall Profit")
        print("="*70)
        print(df.to_string())
        print("="*70)
    except Exception as e:
        print(f"[ERROR] Failed to query portfolios: {e}")
    finally:
        conn.close()

def export_table_to_csv(table_name, output_file=None):
    """Export a database table back to CSV"""
    conn = get_db_connection()
    if not conn:
        return
    
    if output_file is None:
        output_file = f"{table_name}_export.csv"
    
    try:
        df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
        df.to_csv(output_file, index=False)
        print(f"[SUCCESS] Exported {table_name} to {output_file} ({len(df)} rows)")
    except Exception as e:
        print(f"[ERROR] Failed to export table: {e}")
    finally:
        conn.close()

def main():
    print("="*70)
    print("  PORTFOLIO ANALYSIS DATABASE VIEWER")
    print("="*70)
    
    # List all tables
    list_tables()
    
    # View top portfolios
    query_top_portfolios(5)
    
    # View sample data from key tables
    print("\n[SAMPLE DATA] Rebalanced Portfolios")
    view_table("rebalanced_portfolios", limit=5)
    
    print("\n[SAMPLE DATA] Yearly Profits")
    view_table("yearly_profits", limit=5)
    
    print("\n" + "="*70)
    print("[INFO] Database location: " + os.path.abspath(DATABASE_FILE))
    print("="*70)
    
    # Interactive menu
    print("\n[OPTIONS]")
    print("  1. Export table to CSV")
    print("  2. View specific table")
    print("  3. Exit")
    
    choice = input("\nEnter your choice (1-3): ").strip()
    
    if choice == "1":
        table_name = input("Enter table name to export: ").strip()
        export_table_to_csv(table_name)
    elif choice == "2":
        table_name = input("Enter table name to view: ").strip()
        limit = input("Number of rows to display (default 10): ").strip()
        limit = int(limit) if limit.isdigit() else 10
        view_table(table_name, limit)
    else:
        print("Exiting...")

if __name__ == "__main__":
    main()
