from fastapi import FastAPI
from pydantic import BaseModel
from .model import MLModel
#from .database import store_prediction
from .schemas import PredictionRequest, BatchPredictionRequest
import numpy as np

# initialize the FastAPI app
app = FastAPI()

# Load the ML model from path
model = MLModel(model_path="app/models/model.pkl")

# Define an endpoint for single predictions
@app.post("/predict/")
async def predict(request: PredictionRequest):
    """
    Handle POST requests for single predictions.
    
    Parameters:
    - request (PredictionRequest): A Pydantic model containing the input data for prediction.

    Returns:
    - dict: A dictionary containing the predicted value.
    """
    # Convert input data to a Numpy array and make the prediction
    prediction = model.predict(np.array([request.data]))
    # Optionally store the prediction in the database
    #store_prediction(prediction)
    # Return the prediction as JSON response 
    return {"prediction": prediction.tolist()}

# Define an endpoint for batch predictions
@app.post("/predict_batch/")
async def predict_batch(request: BatchPredictionRequest):
    """
    Handle POST requests for batch predictions.
    
    Parameters:
    - request (BatchPredictionRequest): A Pydantic model containing a list of input data for batch prediction.

    Returns:
    - dict: A dictionary containing a list of predicted values.
    """
    # Convert the batch input data to a numpy array
    predictions = model.predict_batch(np.array(request.data))
    #store_prediction(predictions)
    # Return the predictions as a JSON response
    return {"predictions": predictions.tolist()}
