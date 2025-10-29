import json
import pickle
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import streamlit as st
import requests
import threading
import uvicorn

# ======================================================
#                 FASTAPI BACKEND
# ======================================================

ART_MODEL = Path("model.pkl")
ART_SCALER = Path("scaler.pkl")
ART_FEATURES = Path("features.json")

app = FastAPI(title="Manufacturing Efficiency Prediction API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model, scaler, feature_columns = None, None, []

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
        print("⚠️ Model artifacts not found. Using fallback logic.")
except Exception as e:
    print("⚠️ Failed to load artifacts:", e)


class PredictIn(BaseModel):
    Injection_Temperature: float
    Injection_Pressure: float
    Cycle_Time: float
    Cooling_Time: float
    Material_Viscosity: float
    Ambient_Temperature: float
    Operator_Experience: float
    Machine_Utilization: float
    Parts_Per_Hour: float


class PredictOut(BaseModel):
    prediction: float
    risk_level: Literal["Low", "Moderate", "High"]
    model_used: Literal["trained-ML-model", "heuristic-fallback"]
    details: dict


def _bucket(score: float) -> str:
    if score >= 75:
        return "High"
    if score >= 50:
        return "Moderate"
    return "Low"


def _heuristic(row: PredictIn) -> float:
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


@app.get("/health")
def health():
    return {
        "status": "ok",
        "artifacts": {
            "model": ART_MODEL.exists(),
            "scaler": ART_SCALER.exists(),
            "features": ART_FEATURES.exists(),
        },
    }


@app.post("/predict", response_model=PredictOut)
def predict(payload: PredictIn):
    if model is not None and scaler is not None and feature_columns:
        df = pd.DataFrame([payload.model_dump()], columns=feature_columns)
        X_scaled = scaler.transform(df.values)
        yhat = float(model.predict(X_scaled)[0])
        return PredictOut(
            prediction=round(yhat, 3),
            risk_level=_bucket(yhat),
            model_used="trained-ML-model",
            details=payload.model_dump(),
        )
    yhat = _heuristic(payload)
    return PredictOut(
        prediction=round(yhat, 3),
        risk_level=_bucket(yhat),
        model_used="heuristic-fallback",
        details=payload.model_dump(),
    )


# ======================================================
#                 STREAMLIT FRONTEND
# ======================================================

def run_streamlit():
    st.set_page_config(page_title="🏭 Efficiency Prediction", page_icon="🏭", layout="centered")

    with st.sidebar:
        st.header("Settings")
        API_URL = st.text_input("FastAPI URL", "http://127.0.0.1:8000")
        if st.button("Ping /health"):
            try:
                r = requests.get(f"{API_URL}/health", timeout=5)
                st.success(r.json())
            except Exception as e:
                st.error(f"Failed: {e}")

    st.markdown(
        """
        <h1>🏭 Manufacturing Efficiency Prediction</h1>
        <p>Enter parameters to estimate the <b>Efficiency Score</b>.</p>
        """,
        unsafe_allow_html=True,
    )

    with st.form("form"):
        c1, c2 = st.columns(2)
        with c1:
            Injection_Temperature = st.number_input("Injection Temperature (°C)", 0.0, 500.0, 200.0)
            Injection_Pressure = st.number_input("Injection Pressure (bar)", 0.0, 500.0, 100.0)
            Cycle_Time = st.number_input("Cycle Time (sec)", 0.0, 600.0, 60.0)
            Cooling_Time = st.number_input("Cooling Time (sec)", 0.0, 600.0, 30.0)
            Material_Viscosity = st.number_input("Material Viscosity", 0.0, 1000.0, 50.0)
        with c2:
            Ambient_Temperature = st.number_input("Ambient Temperature (°C)", -20.0, 80.0, 25.0)
            Operator_Experience = st.number_input("Operator Experience (years)", 0.0, 50.0, 5.0)
            Machine_Utilization = st.number_input("Machine Utilization (%)", 0.0, 100.0, 70.0)
            Parts_Per_Hour = st.number_input("Parts Per Hour", 0.0, 10000.0, 200.0)

        submitted = st.form_submit_button("Predict")

    if submitted:
        payload = {
            "Injection_Temperature": Injection_Temperature,
            "Injection_Pressure": Injection_Pressure,
            "Cycle_Time": Cycle_Time,
            "Cooling_Time": Cooling_Time,
            "Material_Viscosity": Material_Viscosity,
            "Ambient_Temperature": Ambient_Temperature,
            "Operator_Experience": Operator_Experience,
            "Machine_Utilization": Machine_Utilization,
            "Parts_Per_Hour": Parts_Per_Hour,
        }

        try:
            r = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
            res = r.json()
            st.subheader("Results")
            st.write(f"**Efficiency_Score:** `{res['prediction']}`")

            risk = res["risk_level"]
            if risk == "High":
                st.success("🔥 Risk Level: High – Excellent performance.")
            elif risk == "Moderate":
                st.warning("⚙️ Risk Level: Moderate – Can be improved.")
            else:
                st.info("🟢 Risk Level: Low – Focus on optimization.")

            st.caption(f"Model used: **{res['model_used']}**")

        except Exception as e:
            st.error(f"Failed to call API: {e}")

    st.caption("FastAPI + Streamlit • Combined single app.")


# ======================================================
#                 RUN BOTH TOGETHER
# ======================================================

def start_api():
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")

if __name__ == "__main__":
    t = threading.Thread(target=start_api, daemon=True)
    t.start()
    run_streamlit()
