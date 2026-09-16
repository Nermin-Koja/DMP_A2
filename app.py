from fastapi import FastAPI
from pydantic import BaseModel
import joblib

app = FastAPI()

model = joblib.load("model.pkl")
tfidf = joblib.load("tfidf.pkl")


class PredictionRequest(BaseModel):
    text: str


@app.post("/predict")
def predict(request: PredictionRequest):
    X = tfidf.transform([request.text])
    prediction = model.predict(X)[0]
    return {"prediction": prediction}