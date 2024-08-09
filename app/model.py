import joblib
import numpy as np

class MLModel:
    def __init__(self, model_path: str):
        """
        Initialize the MLModel class by loading a sklearn model from the specified path.

        Parameters:
        - model_path (str): The file path to the pre-trained model.
        """
        self.model = joblib.load(model_path)

    def predict(self, input_data: np.ndarray) -> np.ndarray:
        """
        Make a prediction for a single input data instance.

        Parameters:
        - input_data (np.ndarray): A NumPy array containing the input features.

        Returns:
        - np.ndarray: A NumPy array containing the predicted value.
        """
        return self.model.predict(input_data)

    def predict_batch(self, batch_data: np.ndarray) -> np.ndarray:
        """
        Make predictions for a batch of input data instances.

        Parameters:
        - batch_data (np.ndarray): A NumPy array containing multiple input features.

        Returns:
        - np.ndarray: A NumPy array containing the predicted values for the batch.
        """
        return self.model.predict(batch_data)
