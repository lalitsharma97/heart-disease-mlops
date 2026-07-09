import sqlite3

conn = sqlite3.connect('mlflow.db')
cursor = conn.cursor()

# Check runs table schema first
cursor.execute('PRAGMA table_info(runs)')
print("Runs table schema:", cursor.fetchall())

# Update artifact locations from notebooks/mlruns to mlruns
cursor.execute('UPDATE experiments SET artifact_location = REPLACE(artifact_location, "notebooks/mlruns", "mlruns")')

# Get the correct column name for runs table
cursor.execute('PRAGMA table_info(runs)')
columns = [col[1] for col in cursor.fetchall()]
print("Available columns in runs table:", columns)

# Update runs table with correct column names
if 'artifact_uri' in columns:
    cursor.execute('UPDATE runs SET artifact_uri = REPLACE(artifact_uri, "notebooks/mlruns", "mlruns")')

conn.commit()

# Check updated experiments
cursor.execute('SELECT experiment_id, name, artifact_location FROM experiments')
print("Updated Experiments:", cursor.fetchall())

# Check updated runs
if 'run_uuid' in columns:
    cursor.execute('SELECT run_uuid, name, artifact_uri FROM runs')
else:
    cursor.execute('SELECT * FROM runs LIMIT 3')
print("Sample Runs:", cursor.fetchall())

conn.close()
print("Database updated successfully!")
