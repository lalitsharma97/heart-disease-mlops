import pandas as pd


def load_data(path="data/raw/heart.csv"):
    """
    Load the Heart Disease dataset.

    Args:
        path (str): Path to the dataset.

    Returns:
        pd.DataFrame: Loaded dataset.
    """
    return pd.read_csv(path)
