import sqlite3
import os

# Check if database exists
db_path = "database/portfolio_analysis.db"
if not os.path.exists(db_path):
    print(f"Database not found at: {db_path}")
    exit()

# Connect to database
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Get all tables
cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
tables = cursor.fetchall()

print("=" * 60)
print("DATABASE INSPECTION: portfolio_analysis.db")
print("=" * 60)
print(f"\nTotal Tables: {len(tables)}\n")

if tables:
    print("Tables:")
    for i, table in enumerate(tables, 1):
        table_name = table[0]
        print(f"\n{i}. {table_name}")
        
        # Get row count
        cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
        count = cursor.fetchone()[0]
        print(f"   Rows: {count}")
        
        # Get column info
        cursor.execute(f"PRAGMA table_info({table_name})")
        columns = cursor.fetchall()
        print(f"   Columns ({len(columns)}):")
        for col in columns:
            col_id, col_name, col_type, not_null, default_val, pk = col
            pk_marker = " [PRIMARY KEY]" if pk else ""
            print(f"     - {col_name} ({col_type}){pk_marker}")
else:
    print("No tables found in database.")

conn.close()
print("\n" + "=" * 60)
