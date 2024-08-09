from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from .model import MLModel
#from .database import store_prediction
from .schemas import PredictionRequest, BatchPredictionRequest
import numpy as np

app = FastAPI()

model = MLModel(model_path="app/models/model.pkl")

@app.post("/predict/")
async def predict(request: PredictionRequest):
    prediction = model.predict(np.array([request.data]))
    #store_prediction(prediction)
    prediction_label = "survived" if prediction[0] == 1 else "not survived"
    return {"prediction": prediction_label}

@app.post("/predict_batch/")
async def predict_batch(request: BatchPredictionRequest):
    predictions = model.predict_batch(np.array(request.data))
    #store_prediction(predictions)
    return {"predictions": predictions.tolist()}
