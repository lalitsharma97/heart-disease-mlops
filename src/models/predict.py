# import joblib


# MODEL_PATH = "artifacts/models/best_model.pkl"
# PIPELINE_PATH = "artifacts/models/preprocessing_pipeline.pkl"


# model = joblib.load(MODEL_PATH)
# preprocessor = joblib.load(PIPELINE_PATH)


# def predict(data):
#     """
#     Predict heart disease.
#     """

#     data = preprocessor.transform(data)

#     prediction = model.predict(data)

#     probability = model.predict_proba(data)

#     return prediction, probability


import joblib
import pandas as pd

MODEL_PATH = "artifacts/models/best_model.pkl"

model = joblib.load(MODEL_PATH)


def predict(data):

    df = pd.DataFrame([data])

    prediction = model.predict(df)[0]

    probability = model.predict_proba(df)[0].max()

    return prediction, probability
