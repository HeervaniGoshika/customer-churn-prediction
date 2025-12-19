import joblib
import numpy as np

MODEL_PATH = "models/model.pkl"

def load_model():
    return joblib.load(MODEL_PATH)

def predict_churn(features: list):
    """
    Takes a list of numerical features and returns prediction
    """
    model = load_model()
    data = np.array(features).reshape(1, -1)
    prediction = model.predict(data)
    return int(prediction[0])
