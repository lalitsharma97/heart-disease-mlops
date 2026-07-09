# import os
# import joblib


# def save_model(model, preprocessor):
#     """
#     Save trained model and preprocessing pipeline.

#     Args:
#         model: Trained machine learning model.
#         preprocessor: Fitted preprocessing pipeline.
#     """

#     model_dir = "artifacts/models"
#     os.makedirs(model_dir, exist_ok=True)

#     # Save trained model
#     joblib.dump(
#         model,
#         os.path.join(model_dir, "best_model.pkl")
#     )

#     # Save preprocessing pipeline
#     joblib.dump(
#         preprocessor,
#         os.path.join(model_dir, "preprocessing_pipeline.pkl")
#     )

#     print("Model saved successfully.")
#     print("Preprocessing pipeline saved successfully.")


import json
import os
from datetime import datetime

import joblib


def save_model(model, metrics):
    """
    Save the trained model and metadata.

    Args:
        model: Trained scikit-learn Pipeline.
        metrics (dict): Model evaluation metrics.
    """

    model_dir = "artifacts/models"
    os.makedirs(model_dir, exist_ok=True)

    # Save model
    joblib.dump(
        model,
        os.path.join(model_dir, "best_model.pkl")
    )

    # Metadata
    metadata = {
        "model_name": model.named_steps["classifier"].__class__.__name__,
        "dataset": "Heart Disease Dataset",
        "version": "1.0",
        "training_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "accuracy": round(metrics["Accuracy"], 4),
        "precision": round(metrics["Precision"], 4),
        "recall": round(metrics["Recall"], 4),
        "f1_score": round(metrics["F1 Score"], 4),
        "roc_auc": round(metrics["ROC AUC"], 4),
    }

    with open(
        os.path.join(model_dir, "model_metadata.json"),
        "w"
    ) as file:
        json.dump(metadata, file, indent=4)

    print("Model saved successfully.")
    print("Metadata saved successfully.")
