import numpy as np
import pytest
from app.model import MLModel

@pytest.fixture
def model():
    return MLModel(model_path="app/models/model.pkl")

def test_single_prediction(model):
    # Example input for the model (adjust depending on your actual model input)
    input_data = np.array([[3, 34.5, 0, 0, 7.8292, 1]])
    
    prediction = model.predict(input_data)
    
    # Assert that the prediction is not None and has the expected shape
    assert prediction is not None
    assert len(prediction) == 1

def test_batch_prediction(model):
    # Example batch input for the model
    batch_data = np.array([[3, 34.5, 0, 0, 7.8292, 1], [2, 20.4, 1, 0, 2.3471, 0]])
    
    predictions = model.predict_batch(batch_data)
    
    # Assert that predictions are not None and have the expected shape
    assert predictions is not None
    assert len(predictions) == 2

def test_invalid_input(model):
    # Test with invalid input to ensure the model raises an error
    with pytest.raises(ValueError):
        model.predict(np.array([["invalid", "input"]]))
