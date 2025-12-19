from fastapi import FastAPI
import joblib
import numpy as np

app = FastAPI()

model = joblib.load("models/model.pkl")

@app.post("/predict")
def predict(features: list):
    data = np.array(features).reshape(1, -1)
    prediction = model.predict(data)
    return {"churn_prediction": int(prediction[0])}
