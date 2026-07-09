import os
import pandas as pd

# Create directory if it doesn't exist
os.makedirs("data/raw", exist_ok=True)

# Column names for the UCI Heart Disease dataset
column_names = [
    'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
    'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target'
]

# Download the UCI Heart Disease dataset (Cleveland database)
url = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "heart-disease/processed.cleveland.data"
)

print("Downloading dataset from UCI Machine Learning Repository...")
df = pd.read_csv(url, names=column_names, na_values='?')

# Convert target to binary (0 = no disease, 1 = disease)
# Original dataset: 0 = no disease, 1-4 = disease
df['target'] = (df['target'] > 0).astype(int)

# Handle missing values - drop rows with missing values
df = df.dropna()

# Save dataset
df.to_csv("data/raw/heart.csv", index=False)

# Display information
print("Dataset downloaded successfully.")
print(f"Shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print(f"Target distribution:\n{df['target'].value_counts()}")
print("Dataset saved to: data/raw/heart.csv")
