import numpy as np
import pytest
from app.model import MLModel

@pytest.fixture
def model():
    # Assuming a model is stored in 'path_to_your_model.pkl'
    return MLModel(model_path="path_to_your_model.pkl")

def test_single_prediction(model):
    # Example input for the model (adjust depending on your actual model input)
    input_data = np.array([[5.1, 3.5, 1.4, 0.2]])
    
    prediction = model.predict(input_data)
    
    # Assert that the prediction is not None and has the expected shape
    assert prediction is not None
    assert len(prediction) == 1

def test_batch_prediction(model):
    # Example batch input for the model
    batch_data = np.array([[5.1, 3.5, 1.4, 0.2], [4.9, 3.0, 1.4, 0.2]])
    
    predictions = model.predict_batch(batch_data)
    
    # Assert that predictions are not None and have the expected shape
    assert predictions is not None
    assert len(predictions) == 2

def test_invalid_input(model):
    # Test with invalid input to ensure the model raises an error
    with pytest.raises(ValueError):
        model.predict(np.array([["invalid", "input"]]))
