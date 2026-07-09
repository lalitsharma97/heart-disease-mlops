import pandas as pd

from src.features.feature_engineering import prepare_features


def test_prepare_features():

    df = pd.DataFrame({
        "age": [40, 50, 60, 55, 45, 65, 48, 58],
        "sex": [1, 0, 1, 0, 1, 0, 1, 0],
        "cp": [1, 2, 3, 4, 1, 2, 3, 4],
        "trestbps": [120, 130, 140, 135, 125, 145, 128, 138],
        "chol": [200, 210, 220, 215, 205, 225, 208, 218],
        "fbs": [0, 1, 0, 1, 0, 1, 0, 1],
        "restecg": [0, 1, 0, 1, 0, 1, 0, 1],
        "thalach": [150, 160, 170, 165, 155, 175, 158, 168],
        "exang": [0, 1, 0, 1, 0, 1, 0, 1],
        "oldpeak": [1.0, 2.0, 1.5, 2.5, 1.2, 2.2, 1.3, 2.3],
        "slope": [1, 2, 2, 1, 1, 2, 2, 1],
        "ca": [0, 1, 0, 1, 0, 1, 0, 1],
        "thal": [2, 3, 2, 3, 2, 3, 2, 3],
        "target": [0, 1, 0, 1, 0, 1, 0, 1]
    })

    X_train, X_test, y_train, y_test, preprocessor = prepare_features(df)

    assert X_train.shape[0] > 0
    assert X_test.shape[0] > 0
