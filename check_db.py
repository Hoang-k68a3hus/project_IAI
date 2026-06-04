import sqlite3

# Service metrics
conn = sqlite3.connect('logs/service_metrics.db')
tables = [t[0] for t in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
print('Service Metrics Tables:', tables)
for t in tables:
    count = conn.execute(f'SELECT COUNT(*) FROM [{t}]').fetchone()[0]
    print(f'  {t}: {count} rows')

# Training metrics
conn2 = sqlite3.connect('logs/training_metrics.db')
tables2 = [t[0] for t in conn2.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
print('\nTraining Metrics Tables:', tables2)
for t in tables2:
    count = conn2.execute(f'SELECT COUNT(*) FROM [{t}]').fetchone()[0]
    print(f'  {t}: {count} rows')

# Sample data
print('\n--- Sample Request Data ---')
try:
    df = conn.execute('SELECT * FROM requests ORDER BY timestamp DESC LIMIT 5').fetchall()
    cols = [d[0] for d in conn.execute('SELECT * FROM requests LIMIT 1').description]
    print('Columns:', cols)
    for row in df:
        print(row)
except Exception as e:
    print(f'Error: {e}')
