import joblib
import numpy as np

class MLModel:
    def __init__(self, model_path: str):
        self.model = joblib.load(model_path)

    def predict(self, input_data: np.ndarray) -> np.ndarray:
        return self.model.predict(input_data)

    def predict_batch(self, batch_data: np.ndarray) -> np.ndarray:
        return self.model.predict(batch_data)
