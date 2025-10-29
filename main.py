# main.py
from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, ValidationError

APP_TITLE = "Manufacturing Efficiency Prediction API"
ART_MODEL = Path("model.pkl")
ART_SCALER = Path("scaler.pkl")
ART_FEATURES = Path("features.json")

# ------------- FastAPI app -------------
app = FastAPI(title=APP_TITLE, version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten for prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------- Load artifacts (if available) -------------
model = None
scaler = None
feature_columns: list[str] = []

try:
    if ART_MODEL.exists() and ART_SCALER.exists() and ART_FEATURES.exists():
        with open(ART_MODEL, "rb") as f:
            model = pickle.load(f)
        with open(ART_SCALER, "rb") as f:
            scaler = pickle.load(f)
        with open(ART_FEATURES, "r") as f:
            feature_columns = json.load(f)
        print("✅ Loaded trained model, scaler, and features.")
    else:
        print("⚠️ Trained artifacts not found. Running in heuristic fallback mode.")
except Exception as e:
    print("⚠️ Failed to load artifacts:", e)

# ------------- Schemas -------------
class PredictIn(BaseModel):
    # Keep these 9 fields exactly (they match your Streamlit form)
    Injection_Temperature: float = Field(..., ge=0, le=500)
    Injection_Pressure:   float = Field(..., ge=0, le=500)
    Cycle_Time:           float = Field(..., ge=0, le=600)
    Cooling_Time:         float = Field(..., ge=0, le=600)
    Material_Viscosity:   float = Field(..., ge=0, le=1000)
    Ambient_Temperature:  float = Field(..., ge=-20, le=80)
    Operator_Experience:  float = Field(..., ge=0, le=50)
    Machine_Utilization:  float = Field(..., ge=0, le=100)
    Parts_Per_Hour:       float = Field(..., ge=0, le=10000)

class PredictOut(BaseModel):
    prediction: float  # Efficiency_Score
    risk_level: Literal["Low", "Moderate", "High"]
    model_used: Literal["trained-ML-model", "heuristic-fallback"]
    details: dict

# ------------- Helpers -------------
def _bucket(score: float) -> str:
    if score >= 75:
        return "High"
    if score >= 50:
        return "Moderate"
    return "Low"

def _heuristic(row: PredictIn) -> float:
    """Transparent fallback producing a 0..100 score."""
    s = 50.0
    s += 0.05 * (row.Machine_Utilization - 70)
    s += 0.03 * (row.Operator_Experience - 5)
    s += 0.012 * (row.Parts_Per_Hour - 200)
    s -= 0.04 * (row.Cycle_Time - 60)
    s -= 0.03 * (row.Cooling_Time - 30)
    s += 0.01 * (row.Injection_Temperature - 200)
    s += 0.008 * (row.Injection_Pressure - 100)
    s -= 0.01 * (row.Material_Viscosity - 50)
    s -= 0.02 * (row.Ambient_Temperature - 25)
    return float(max(0.0, min(100.0, s)))

# ------------- Routes -------------
@app.get("/")
def home():
    return {"message": "OK", "docs": "/docs", "health": "/health"}

@app.get("/health")
def health():
    return {
        "status": "ok",
        "artifacts": {
            "model": ART_MODEL.exists(),
            "scaler": ART_SCALER.exists(),
            "features": ART_FEATURES.exists(),
        }
    }

@app.post("/predict", response_model=PredictOut)
def predict(payload: PredictIn):
    # Use trained model if artifacts are present
    if model is not None and scaler is not None and feature_columns:
        # Create DataFrame with exact feature order expected by the scaler/model
        row_dict = payload.model_dump()
        # Safety: if features.json differs, reindex to that order; missing -> error
        try:
            df = pd.DataFrame([row_dict], columns=feature_columns)
        except Exception as e:
            raise ValidationError([f"Feature mismatch: {e}"], PredictIn)
        X_scaled = scaler.transform(df.values)
        yhat = float(model.predict(X_scaled)[0])
        return PredictOut(
            prediction=round(yhat, 3),
            risk_level=_bucket(yhat),
            model_used="trained-ML-model",
            details=row_dict,
        )

    # Fallback if not trained yet
    yhat = _heuristic(payload)
    return PredictOut(
        prediction=round(yhat, 3),
        risk_level=_bucket(yhat),
        model_used="heuristic-fallback",
        details=payload.model_dump(),
    )

# Run (dev): uvicorn main:app --reload --port 8000
