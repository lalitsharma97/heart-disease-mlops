import os
import joblib


MODEL_PATH = "artifacts/models/best_model.pkl"


def test_model_file_exists():
    """
    Verify that the trained model file exists.
    """
    assert os.path.exists(MODEL_PATH)


def test_model_loads_successfully():
    """
    Verify that the trained model loads correctly.
    """
    model = joblib.load(MODEL_PATH)

    assert model is not None
    assert hasattr(model, "predict")
    assert hasattr(model, "predict_proba")


def test_pipeline_contains_classifier():
    """
    Verify that the saved Pipeline contains a classifier.
    """
    model = joblib.load(MODEL_PATH)

    assert "classifier" in model.named_steps