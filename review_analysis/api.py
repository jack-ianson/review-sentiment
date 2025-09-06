from __future__ import annotations
from fastapi import FastAPI
from pydantic import BaseModel

from .inference import BagOfWordsInference


model = BagOfWordsInference(
    model_path="./model_stores/model.pt",
    tokeniser_path="./model_stores/tokeniser.json",
    device="cpu",
)

app = FastAPI(title="Review Sentiment Analysis API", version="1.0")


class ReviewRequest(BaseModel):
    title: str
    review: str


@app.post("/predict")
def predict_sentiment(request: ReviewRequest):
    predicted_class, probabilities = model.predict(request.title, request.review)

    return {
        "predicted_class": int(predicted_class),
        "probabilities": {
            "negative": float(probabilities[0]),
            "positive": float(probabilities[1]),
        },
    }
