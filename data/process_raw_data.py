import pandas as pd
import os
from sklearn.impute import SimpleImputer

def preprocess_data(df):
    """
    Clean and preprocess the Heart Disease dataset.
    """
    df = df.copy()
    
    # Replace '?' with missing values
    df.replace("?", pd.NA, inplace=True)
    
    # Convert columns to numeric where possible
    for col in df.columns:
        if df[col].dtype == 'object':
            try:
                df[col] = pd.to_numeric(df[col])
            except ValueError:
                pass
    
    # Impute missing values in numerical columns
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
    imputer = SimpleImputer(strategy="median")
    df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
    
    # Remove duplicate rows
    df.drop_duplicates(inplace=True)
    
    return df

# Load raw dataset
raw_data_path = "data/raw/heart.csv"
df = pd.read_csv(raw_data_path)

print(f"Raw dataset shape: {df.shape}")
print(f"Raw dataset columns: {list(df.columns)}")

# Preprocess the data
df_processed = preprocess_data(df)

print(f"Processed dataset shape: {df_processed.shape}")
print(f"Target distribution:\n{df_processed['target'].value_counts()}")

# Save processed dataset
os.makedirs("data/processed", exist_ok=True)
df_processed.to_csv("data/processed/processed_heart.csv", index=False)

print("Processed dataset saved to: data/processed/processed_heart.csv")
