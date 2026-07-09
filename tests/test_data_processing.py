import pandas as pd

from src.data.preprocess import preprocess_data


def test_preprocess_data():

    df = pd.DataFrame({
        "age": [45, None, 60],
        "chol": [200, 220, None],
        "target": [0, 1, 1]
    })

    processed_df = preprocess_data(df)

    assert processed_df.isnull().sum().sum() == 0