import pandas as pd
from sklearn.impute import SimpleImputer


def preprocess_data(df):
    """
    Clean and preprocess the Heart Disease dataset.

    Args:
        df (pd.DataFrame): Input dataset.

    Returns:
        pd.DataFrame: Preprocessed dataset.
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
