import streamlit as st
import numpy as np
import pandas as pd
import pickle, json
from pathlib import Path

st.set_page_config(page_title="🏭 Efficiency Prediction", page_icon="⚙️", layout="centered")

MODEL_FILE = Path("model.pkl")
SCALER_FILE = Path("scaler.pkl")
FEATURES_FILE = Path("features.json")

model = scaler = None
features = []
try:
    if MODEL_FILE.exists():
        model = pickle.load(open(MODEL_FILE, "rb"))
    if SCALER_FILE.exists():
        scaler = pickle.load(open(SCALER_FILE, "rb"))
    if FEATURES_FILE.exists():
        features = json.load(open(FEATURES_FILE))
except Exception as e:
    st.warning(f"⚠️ Could not load model files: {e}")

st.title("🏭 Manufacturing Efficiency Prediction System")
st.write("Use this interface to predict **Efficiency_Score** based on input manufacturing parameters.")

col1, col2 = st.columns(2)
with col1:
    Injection_Temperature = st.number_input("Injection Temperature (°C)", 0.0, 500.0, 200.0)
    Injection_Pressure    = st.number_input("Injection Pressure (bar)", 0.0, 500.0, 100.0)
    Cycle_Time            = st.number_input("Cycle Time (sec)", 0.0, 600.0, 60.0)
    Cooling_Time          = st.number_input("Cooling Time (sec)", 0.0, 600.0, 30.0)
    Material_Viscosity    = st.number_input("Material Viscosity", 0.0, 1000.0, 50.0)
with col2:
    Ambient_Temperature   = st.number_input("Ambient Temperature (°C)", -20.0, 80.0, 25.0)
    Operator_Experience   = st.number_input("Operator Experience (years)", 0.0, 50.0, 5.0)
    Machine_Utilization   = st.number_input("Machine Utilization (%)", 0.0, 100.0, 70.0)
    Parts_Per_Hour        = st.number_input("Parts Per Hour", 0.0, 10000.0, 200.0)

if st.button("Predict"):
    row = {
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
    df = pd.DataFrame([row])

    if model is not None and scaler is not None and features:
        try:
            df = df[features]
        except Exception:
            st.error("Features.json does not match the inputs. Re-train or update feature list.")
            st.stop()
        Xs = scaler.transform(df.values)
        score = float(model.predict(Xs)[0])
        used = "trained-ML-model"
    else:
        # fallback heuristic (0..100)
        score = (
            50
            + 0.05 * (Machine_Utilization - 70)
            + 0.03 * (Operator_Experience - 5)
            + 0.012 * (Parts_Per_Hour - 200)
            - 0.04 * (Cycle_Time - 60)
            - 0.03 * (Cooling_Time - 30)
            + 0.01 * (Injection_Temperature - 200)
            + 0.008 * (Injection_Pressure - 100)
            - 0.01 * (Material_Viscosity - 50)
            - 0.02 * (Ambient_Temperature - 25)
        )
        score = float(max(0, min(100, score)))
        used = "heuristic-fallback"

    st.subheader("Results")
    st.metric("Efficiency_Score", f"{score:.2f}")
    if score >= 75:
        st.success("🔥 Risk Level: High – Excellent performance.")
    elif score >= 50:
        st.warning("⚙️ Risk Level: Moderate – Can be improved.")
    else:
        st.info("🟢 Risk Level: Low – Focus on optimization.")
    st.caption(f"Model used: {used}")
