from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "model" / "ufo_model_bundle.pkl"


class PredictionRequest(BaseModel):
    seconds: float = Field(..., ge=1, le=60, description="UFO sighting duration in seconds")
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)


class PredictionResponse(BaseModel):
    country: str
    country_code: int
    model_accuracy: float | None = None


def load_model_bundle() -> dict:
    if not MODEL_PATH.exists():
        raise RuntimeError(
            f"Model file not found at {MODEL_PATH}. Run `python model/train_model.py` first."
        )

    with MODEL_PATH.open("rb") as f:
        return pickle.load(f)


app = FastAPI(
    title="UFO Country Prediction API",
    description="Local FastAPI backend for the Task 1 UFO machine learning web app.",
    version="1.0.0",
)

model_bundle = load_model_bundle()


@app.get("/")
def root() -> dict[str, str]:
    return {
        "message": "UFO prediction API is running.",
        "docs": "Open /docs to test the API.",
    }


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: PredictionRequest) -> PredictionResponse:
    try:
        features = np.array([[payload.seconds, payload.latitude, payload.longitude]])
        prediction = model_bundle["model"].predict(features)[0]
        country = model_bundle["label_encoder"].inverse_transform([prediction])[0]
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return PredictionResponse(
        country=country.upper(),
        country_code=int(prediction),
        model_accuracy=model_bundle.get("accuracy"),
    )

