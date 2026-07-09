from time import perf_counter

from fastapi import FastAPI, HTTPException
from prometheus_fastapi_instrumentator import Instrumentator

from api.schemas import HeartData
from api.inference import predict
from api.logging_config import logger

app = FastAPI(
    title="Heart Disease Prediction API",
    version="1.0"
)

# Set up Prometheus instrumentation before app starts
Instrumentator().instrument(app).expose(app)

@app.on_event("startup")
def startup_event():
    logger.info("Heart Disease Prediction API started successfully.")


@app.get("/")
def home():
    logger.info("Home endpoint accessed")

    return {
        "message": "Heart Disease Prediction API is running."
    }


@app.get("/health")
def health_check():
    logger.info("Health check endpoint accessed")

    from api.inference import model_loaded

    return {
        "status": "healthy",
        "model_loaded": model_loaded
    }


@app.post("/predict")
def predict_heart_disease(data: HeartData):

    start_time = perf_counter()

    try:

        logger.info(f"Prediction request: {data.model_dump()}")

        prediction, probability = predict(data.model_dump())

        latency = perf_counter() - start_time

        logger.info(
            f"Prediction={prediction}, "
            f"Confidence={probability:.4f}, "
            f"Latency={latency:.4f}s"
        )

        prediction_label = (
            "Heart Disease Detected"
            if prediction == 1
            else "No Heart Disease Detected"
        )

        return {
            "prediction": prediction_label,
            "prediction_class": int(prediction),
            "confidence": round(float(probability), 4)
        }

    except Exception as e:

        logger.exception("Prediction failed.")

        raise HTTPException(
            status_code=500,
            detail="Prediction failed."
        )