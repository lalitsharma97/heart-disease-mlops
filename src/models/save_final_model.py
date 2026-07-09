"""
Save Final Model for Deployment
"""
import sys
import os
import joblib
import pandas as pd
from pathlib import Path
import mlflow
import mlflow.sklearn

# Add project root to path to import src modules
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.features.feature_engineering import prepare_features
from src.models.train import train_models


def save_final_model():
    """
    Save the final selected model (Random Forest Initial) and preprocessing pipeline
    for deployment and reproducibility.
    """

    # Set MLflow tracking URI to project root database (SQLite for metadata, mlruns for artifacts)
    notebooks_dir = os.path.abspath(os.path.join(project_root, 'notebooks'))
    mlflow.set_tracking_uri(f"sqlite:///{notebooks_dir}/mlflow.db")
    mlflow.set_experiment("Heart Disease Prediction")

    # Load data
    df = pd.read_csv("data/processed/processed_heart.csv")

    # Prepare features and get preprocessing pipeline
    X_train, X_test, y_train, y_test, preprocessor = prepare_features(df)

    # Train models
    models, cv_results = train_models(X_train, y_train, preprocessor)

    # Select final model (Random Forest Initial - based on highest recall)
    final_model = models["Random Forest"]

    # Create models directory if it doesn't exist
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)

    # Save the complete pipeline (preprocessor + classifier)
    joblib.dump(final_model, "models/final_model.pkl")
    print("Final model saved to: models/final_model.pkl")

    # Save preprocessing pipeline separately for reproducibility
    joblib.dump(preprocessor, "models/preprocessor.pkl")
    print("Preprocessing pipeline saved to: models/preprocessor.pkl")

    # Save model metadata
    metadata = {
        "model_name": "Random Forest (Initial)",
        "model_type": "RandomForestClassifier",
        "accuracy": 0.8500,
        "precision": 0.8519,
        "recall": 0.8214,
        "roc_auc": 0.9358,
        "cv_accuracy": 0.8137,
        "cv_std": 0.0903,
        "random_state": 42,
        "n_estimators": 100,
        "max_depth": None,
        "min_samples_split": 2,
        "min_samples_leaf": 1
    }

    joblib.dump(metadata, "models/model_metadata.pkl")
    print("Model metadata saved to: models/model_metadata.pkl")

    # Log final model to MLflow
    with mlflow.start_run(run_name="Final Model - Random Forest"):
        mlflow.log_params(metadata)
        mlflow.sklearn.log_model(
            final_model,
            artifact_path="model",
            serialization_format="pickle",
            skops_trusted_types=["numpy.dtype"]
        )
        mlflow.log_artifact("models/preprocessor.pkl", artifact_path="preprocessor")
        print("Final model logged to MLflow")

    # Save feature information
    feature_info = {
        "numerical_features": ["age", "trestbps", "chol", "thalach", "oldpeak"],
        "categorical_features": ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"],
        "target": "target"
    }

    joblib.dump(feature_info, "models/feature_info.pkl")
    print("Feature information saved to: models/feature_info.pkl")

    print("\n=== Model Packaging Complete ===")
    print("All components saved for reproducibility:")
    print("- Final model (pipeline): models/final_model.pkl")
    print("- Preprocessing pipeline: models/preprocessor.pkl")
    print("- Model metadata: models/model_metadata.pkl")
    print("- Feature information: models/feature_info.pkl")


if __name__ == "__main__":
    save_final_model()
