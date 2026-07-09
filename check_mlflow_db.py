import sqlite3

conn = sqlite3.connect('mlflow.db')
cursor = conn.cursor()

# Get all tables
cursor.execute('SELECT name FROM sqlite_master WHERE type="table"')
tables = cursor.fetchall()
print("Tables in MLflow database:", tables)

if tables:
    # Get experiments
    cursor.execute('SELECT * FROM experiments')
    experiments = cursor.fetchall()
    print("\nExperiments:", experiments)
    
    # Get runs
    cursor.execute('SELECT * FROM runs')
    runs = cursor.fetchall()
    print("\nNumber of runs:", len(runs))
    if runs:
        print("First few runs:", runs[:3])

conn.close()
