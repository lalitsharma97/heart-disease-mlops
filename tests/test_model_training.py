import pandas as pd

from src.features.feature_engineering import prepare_features
from src.models.train import train_models


def test_train_models():

    df = pd.DataFrame({
        "age": [40, 50, 60, 55, 45, 65, 48, 58, 52, 62, 42, 57, 53, 63, 47, 59],
        "sex": [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
        "cp": [1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4],
        "trestbps": [
            120, 130, 140, 135, 125, 145, 128, 138, 132, 142,
            126, 136, 133, 143, 127, 137
        ],
        "chol": [200, 210, 220, 215, 205, 225, 208, 218, 212, 222, 206, 216, 213, 223, 207, 217],
        "fbs": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        "restecg": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        "thalach": [150, 160, 170, 165, 155, 175, 158, 168, 162, 172, 156, 166, 163, 173, 157, 167],
        "exang": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        "oldpeak": [1.0, 2.0, 1.5, 2.5, 1.2, 2.2, 1.3, 2.3, 1.8, 2.8, 1.1, 2.1, 1.4, 2.4, 1.6, 2.6],
        "slope": [1, 2, 2, 1, 1, 2, 2, 1, 1, 2, 2, 1, 1, 2, 2, 1],
        "ca": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        "thal": [2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3],
        "target": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
    })

    X_train, X_test, y_train, y_test, preprocessor = prepare_features(df)

    models, cv_results = train_models(
        X_train,
        y_train,
        preprocessor
    )

    assert "Random Forest" in models
    assert "Logistic Regression" in models
