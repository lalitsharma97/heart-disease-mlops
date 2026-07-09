"""
End-to-end model training pipeline.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# flake8: noqa: E402
import mlflow
import mlflow.sklearn

from src.data.load_data import load_data  # noqa: E402
from src.data.preprocess import preprocess_data  # noqa: E402
from src.features.feature_engineering import prepare_features  # noqa: E402
from src.models.train import train_models  # noqa: E402
from src.models.evaluate import evaluate_models  # noqa: E402
from src.models.model_selection import select_best_model  # noqa: E402
from src.models.save_model import save_model  # noqa: E402


def main():

    # Set MLflow tracking URI to project root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
    mlflow_db_path = os.path.join(project_root, 'mlflow.db')
    
    print(f"Project root: {project_root}")
    print(f"MLflow database path: {mlflow_db_path}")
    
    mlflow.set_tracking_uri(f"sqlite:///{project_root}/mlflow.db")
    mlflow.set_experiment("Heart Disease Prediction")

    # Start MLflow run
    with mlflow.start_run(run_name="Model Training Pipeline"):
        # Load data
        df = load_data()
        mlflow.log_param("dataset_size", len(df))

        # Preprocess data
        df = preprocess_data(df)
        mlflow.log_param("preprocessed_size", len(df))

        # Feature engineering
        X_train, X_test, y_train, y_test, preprocessor = prepare_features(df)
        mlflow.log_param("train_size", len(X_train))
        mlflow.log_param("test_size", len(X_test))

        # Train models
        models, cv_results = train_models(
            X_train,
            y_train,
            preprocessor
        )

        # Log cross-validation results
        for model_name, results in cv_results.items():
            mlflow.log_metric(f"{model_name}_cv_mean", results['cv_mean'])
            mlflow.log_metric(f"{model_name}_cv_std", results['cv_std'])

        # Evaluate models
        results = evaluate_models(
            models,
            X_train,
            X_test,
            y_train,
            y_test
        )

        print(results)

        # Select best model
        best_model = select_best_model(
            results,
            models
        )

        from src.models.evaluate import save_feature_importance

        feature_names = X_train.columns.tolist()

        save_feature_importance(
            best_model,
            feature_names
        )

        # Evaluate best model to get metrics
        best_model_dict = {
            best_model.named_steps["classifier"].__class__.__name__:
            best_model
        }
        best_results = evaluate_models(
            best_model_dict, X_train, X_test, y_train, y_test
        )
        best_metrics = best_results.iloc[0].to_dict()

        # Log best model metrics to MLflow
        for metric_name, metric_value in best_metrics.items():
            if isinstance(metric_value, (int, float)):
                mlflow.log_metric(f"best_{metric_name}", metric_value)

        save_model(
            model=best_model,
            metrics=best_metrics
        )

        from src.models.evaluate import save_classification_report

        save_classification_report(
            best_model,
            X_test,
            y_test
        )

        # Log the best model to MLflow
        mlflow.sklearn.log_model(
            best_model,
            artifact_path="model",
            serialization_format="pickle"
        )
        mlflow.log_artifact("artifacts/models/best_model.pkl", artifact_path="model_artifacts")

        print("Training pipeline completed successfully.")
    
    # Verify MLflow database was created
    print(f"Current working directory: {os.getcwd()}")
    print(f"MLflow database exists: {os.path.exists(mlflow_db_path)}")
    if os.path.exists(mlflow_db_path):
        print(f"MLflow database size: {os.path.getsize(mlflow_db_path)} bytes")


if __name__ == "__main__":
    main()
