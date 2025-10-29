# app.py — Combined FastAPI (backend) + Streamlit (frontend)
# Run locally with:   streamlit run app.py

import os
import json
import time
import pickle
import threading
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import requests
import streamlit as st

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# ============================================================
#                 FASTAPI BACKEND
# ============================================================

ART_MODEL = Path("model.pkl")
ART_SCALER = Path("scaler.pkl")
ART_FEATURES = Path("features.json")
API_HOST = "127.0.0.1"
API_PORT = 8000
API_URL_DEFAULT = f"http://{API_HOST}:{API_PORT}"

app = FastAPI(title="Manufacturing Efficiency Prediction API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten for prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Try to load trained artifacts if present
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
        print("⚠️ Model artifacts not found. Using heuristic fallback.")
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
    efficiency_score: float
    risk_level: Literal["Low", "Moderate", "High"]
    model_used: Literal["trained-ML-model", "heuristic-fallback"]

def _risk(score: float) -> str:
    if score >= 75:
        return "High"
    if score >= 50:
        return "Moderate"
    return "Low"

def _heuristic(p: PredictIn) -> float:
    # Transparent 0..100 scoring if trained model missing
    s = 50.0
    s += 0.05 * (p.Machine_Utilization - 70)
    s += 0.03 * (p.Operator_Experience - 5)
    s += 0.012 * (p.Parts_Per_Hour - 200)
    s -= 0.04 * (p.Cycle_Time - 60)
    s -= 0.03 * (p.Cooling_Time - 30)
    s += 0.01 * (p.Injection_Temperature - 200)
    s += 0.008 * (p.Injection_Pressure - 100)
    s -= 0.01 * (p.Material_Viscosity - 50)
    s -= 0.02 * (p.Ambient_Temperature - 25)
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
    # Use trained model if available
    if model is not None and scaler is not None and feature_columns:
        row = pd.DataFrame([payload.model_dump()], columns=feature_columns)
        Xs = scaler.transform(row.values)
        score = float(model.predict(Xs)[0])
        return PredictOut(
            efficiency_score=round(score, 2),
            risk_level=_risk(score),
            model_used="trained-ML-model",
        )
    # Fallback
    score = _heuristic(payload)
    return PredictOut(
        efficiency_score=round(score, 2),
        risk_level=_risk(score),
        model_used="heuristic-fallback",
    )

def run_api():
    """Run FastAPI (Uvicorn) in a background thread."""
    uvicorn.run(app, host=API_HOST, port=API_PORT, log_level="info")

# ============================================================
#                 STREAMLIT FRONTEND
# ============================================================

def badge(text: str, bg: str):
    st.markdown(
        f"""
        <div style="
           display:inline-block;padding:10px 12px;border-radius:10px;
           background:{bg};border:1px solid rgba(0,0,0,.06);font-weight:600;">
           {text}
        </div>
        """,
        unsafe_allow_html=True,
    )

def run_ui():
    st.set_page_config(page_title="🏭 Efficiency Prediction", page_icon="🏭", layout="centered")

    # ---- Sidebar ----
    with st.sidebar:
        st.header("Settings")
        api_url = st.text_input("FastAPI URL", API_URL_DEFAULT)
        if st.button("Ping /health"):
            try:
                r = requests.get(f"{api_url}/health", timeout=5)
                st.success(r.json())
            except Exception as e:
                st.error(f"Failed: {e}")

    # ---- Header ----
    st.markdown(
        """
        <h1 style="margin-bottom:0">🏭 Manufacturing Efficiency Prediction</h1>
        <p style="color:#6c757d;margin-top:0">
            Fill in the parameters to estimate <b>Efficiency_Score</b>.
        </p>
        """,
        unsafe_allow_html=True,
    )

    # ---- Form ----
    with st.form("eff_form"):
        c1, c2 = st.columns(2)
        with c1:
            Injection_Temperature = st.number_input("Injection Temperature (°C)", 0.0, 500.0, 200.0, 1.0)
            Injection_Pressure   = st.number_input("Injection Pressure (bar)", 0.0, 500.0, 100.0, 1.0)
            Cycle_Time           = st.number_input("Cycle Time (sec)", 0.0, 600.0, 60.0, 1.0)
            Cooling_Time         = st.number_input("Cooling Time (sec)", 0.0, 600.0, 30.0, 1.0)
            Material_Viscosity   = st.number_input("Material Viscosity", 0.0, 1000.0, 50.0, 1.0)
        with c2:
            Ambient_Temperature  = st.number_input("Ambient Temperature (°C)", -20.0, 80.0, 25.0, 1.0)
            Operator_Experience  = st.number_input("Operator Experience (years)", 0.0, 50.0, 5.0, 1.0)
            Machine_Utilization  = st.number_input("Machine Utilization (%)", 0.0, 100.0, 70.0, 1.0)
            Parts_Per_Hour       = st.number_input("Parts Per Hour", 0.0, 10000.0, 200.0, 1.0)

        submitted = st.form_submit_button("Predict", use_container_width=True)

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
            r = requests.post(f"{API_URL_DEFAULT}/predict", json=payload, timeout=10)
            if r.status_code != 200:
                st.error(f"API error {r.status_code}: {r.text}")
                return
            res = r.json()

            st.subheader("Results")
            st.write(f"**Efficiency_Score:** `{res['efficiency_score']}`")

            risk = res["risk_level"]
            if risk == "High":
                badge("🔥 Risk Level: High – Excellent performance.", "#ffd6a5")
            elif risk == "Moderate":
                badge("⚙️ Risk Level: Moderate – Can be improved.", "#fff3bf")
            else:
                badge("🟢 Risk Level: Low – Focus on optimization.", "#e7f5ff")

            st.caption(f"Model used: **{res['model_used']}**")
            with st.expander("Details sent to API"):
                st.json(payload)

        except Exception as e:
            st.error(f"Failed to call API: {e}")

    st.caption("FastAPI + Streamlit (single file). If trained artifacts exist, uses ML model; otherwise uses a safe heuristic.")
    

# ============================================================
#                 ENTRY POINT
# ============================================================

def start_everything():
    # Start FastAPI in the background, then draw the Streamlit UI.
    threading.Thread(target=run_api, daemon=True).start()
    time.sleep(1.5)   # small delay so the API is ready
    run_ui()

# When run with `streamlit run app.py`, Streamlit executes this file.
# We call our starter to spin up the API thread and then render the UI.
start_everything()
