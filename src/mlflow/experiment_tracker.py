"""
MLflow Experiment Tracking
"""

import mlflow
import mlflow.sklearn
import os
import sys
from pathlib import Path


def log_experiment(
    model_name,
    model,
    params,
    metrics,
    artifacts=None
):
    """
    Log model experiment to MLflow.

    Args:
        model_name (str): Name of the model.
        model: Trained model.
        params (dict): Model parameters.
        metrics (dict): Evaluation metrics.
        artifacts (list): List of artifact file paths.
    """

    # Get project root directory
    import os
    # Navigate from script location to project root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
    
    # Set tracking URI to project root database (SQLite for metadata, mlruns for artifacts)
    tracking_uri = f"sqlite:///{project_root}/mlflow.db"
    mlflow.set_tracking_uri(tracking_uri)
    
    mlflow.set_experiment("Heart Disease Prediction")

    with mlflow.start_run(run_name=model_name):

        # Log Parameters
        mlflow.log_params(params)

        # Log Metrics
        mlflow.log_metrics(metrics)

        # Log Model with trusted types to handle numpy.dtype
        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            serialization_format="pickle",
            skops_trusted_types=["numpy.dtype"]
        )

        # Log Artifacts
        if artifacts:
            for artifact in artifacts:
                mlflow.log_artifact(artifact)

        print(f"{model_name} logged successfully.")
