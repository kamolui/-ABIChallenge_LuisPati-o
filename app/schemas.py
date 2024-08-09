from pydantic import BaseModel
from typing import List

# Schema for a single prediction request
class PredictionRequest(BaseModel):
    data: List[float]

# Schema for batch prediction requests
class BatchPredictionRequest(BaseModel):
    data: List[List[float]]

# Schema for the prediction response
class PredictionResponse(BaseModel):
    prediction: List[float]

# Schema for batch prediction responses
class BatchPredictionResponse(BaseModel):
    predictions: List[List[float]]
