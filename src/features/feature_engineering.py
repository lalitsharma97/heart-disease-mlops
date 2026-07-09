"""
Feature engineering and data preparation.
"""

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def prepare_features(df):
    """
    Prepare training and testing datasets along with
    a reusable preprocessing pipeline.

    Args:
        df (pd.DataFrame): Processed dataset.

    Returns:
        tuple:
            X_train,
            X_test,
            y_train,
            y_test,
            preprocessor
    """

    # Separate features and target
    X = df.drop("target", axis=1)
    y = df["target"]

    # Numerical columns
    numerical_features = [
        "age",
        "trestbps",
        "chol",
        "thalach",
        "oldpeak"
    ]

    # Categorical columns
    categorical_features = [
        "sex",
        "cp",
        "fbs",
        "restecg",
        "exang",
        "slope",
        "ca",
        "thal"
    ]

    # Preprocessing pipeline
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numerical_features),
            ("cat", categorical_transformer, categorical_features)
        ],
        remainder="passthrough"
    )

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    return (
        X_train,
        X_test,
        y_train,
        y_test,
        preprocessor
    )
