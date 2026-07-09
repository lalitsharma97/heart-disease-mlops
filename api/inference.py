import joblib
import pandas as pd

from fastapi import HTTPException

from api.logging_config import logger

MODEL_PATH = "artifacts/models/best_model.pkl"

model = None
model_loaded = False

try:
    model = joblib.load(MODEL_PATH)
    model_loaded = True
    logger.info("Model loaded successfully.")

except FileNotFoundError:
    logger.error(f"Model file not found at {MODEL_PATH}")
except Exception as e:
    logger.exception(f"Failed to load model: {e}")


def predict(data):
    if not model_loaded or model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please ensure the model file exists."
        )

    df = pd.DataFrame([data])

    prediction = model.predict(df)[0]

    probability = model.predict_proba(df)[0].max()

    return prediction, probability