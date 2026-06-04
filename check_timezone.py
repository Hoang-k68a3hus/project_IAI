import sqlite3
from datetime import datetime

conn = sqlite3.connect('logs/service_metrics.db')

# Latest data in DB
latest = conn.execute("SELECT MAX(timestamp) FROM requests").fetchone()[0]
print(f"DB latest timestamp: {latest}")

# SQLite 'now' returns UTC
sqlite_now = conn.execute("SELECT datetime('now')").fetchone()[0]
print(f"SQLite datetime('now') [UTC]: {sqlite_now}")

# SQLite 'now', 'localtime' returns local time
sqlite_local = conn.execute("SELECT datetime('now', 'localtime')").fetchone()[0]
print(f"SQLite datetime('now', 'localtime'): {sqlite_local}")

# Python now
print(f"Python datetime.now(): {datetime.now().isoformat()}")

# Check if query would return data
query_60min = f"""
    SELECT COUNT(*) FROM requests
    WHERE timestamp > datetime('now', '-60 minutes')
"""
count_utc = conn.execute(query_60min).fetchone()[0]
print(f"\nRecords in last 60 min (UTC): {count_utc}")

# With localtime
query_local = f"""
    SELECT COUNT(*) FROM requests
    WHERE timestamp > datetime('now', 'localtime', '-60 minutes')
"""
count_local = conn.execute(query_local).fetchone()[0]
print(f"Records in last 60 min (localtime): {count_local}")

# All records today
query_today = """
    SELECT COUNT(*) FROM requests
    WHERE date(timestamp) = date('now', 'localtime')
"""
count_today = conn.execute(query_today).fetchone()[0]
print(f"Records today (localtime): {count_today}")
