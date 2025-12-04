import sqlite3

conn = sqlite3.connect("database/portfolio_analysis.db")
cursor = conn.cursor()

cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
tables = cursor.fetchall()

print(f"Total tables: {len(tables)}")
print("\nTable names:")
for i, table in enumerate(tables, 1):
    table_name = table[0]
    cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
    count = cursor.fetchone()[0]
    print(f"{i}. {table_name} ({count} rows)")

conn.close()
