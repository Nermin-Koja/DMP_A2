from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel
import joblib

app = FastAPI()

model = joblib.load("model.pkl")
tfidf = joblib.load("tfidf.pkl")


class PredictionRequest(BaseModel):
    text: str


@app.get("/")
def home():
    return FileResponse("index.html")


@app.post("/predict")
def predict(request: PredictionRequest):
    X = tfidf.transform([request.text])
    prediction = model.predict(X)[0]
    return {"prediction": prediction}