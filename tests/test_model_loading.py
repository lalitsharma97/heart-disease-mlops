import os
import joblib
import pytest


MODEL_PATH = "artifacts/models/best_model.pkl"


@pytest.mark.skipif(not os.path.exists(MODEL_PATH), reason="Model file not found")
def test_model_file_exists():
    """
    Verify that the trained model file exists.
    """
    assert os.path.exists(MODEL_PATH)


@pytest.mark.skipif(not os.path.exists(MODEL_PATH), reason="Model file not found")
def test_model_loads_successfully():
    """
    Verify that the trained model loads correctly.
    """
    model = joblib.load(MODEL_PATH)

    assert model is not None
    assert hasattr(model, "predict")
    assert hasattr(model, "predict_proba")


@pytest.mark.skipif(not os.path.exists(MODEL_PATH), reason="Model file not found")
def test_pipeline_contains_classifier():
    """
    Verify that the saved Pipeline contains a classifier.
    """
    model = joblib.load(MODEL_PATH)

    assert "classifier" in model.named_steps
