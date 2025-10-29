# app.py
import os
import requests
import streamlit as st

st.set_page_config(page_title="🏭 Efficiency Prediction", page_icon="🏭", layout="centered")

# ---------- Sidebar ----------
with st.sidebar:
    st.header("Settings")
    API_URL = st.text_input("FastAPI URL", os.getenv("API_URL", "http://127.0.0.1:8000"))
    if st.button("Ping /health"):
        try:
            r = requests.get(f"{API_URL}/health", timeout=5)
            st.success(r.json())
        except Exception as e:
            st.error(f"Failed: {e}")

# ---------- Header ----------
st.markdown(
    """
    <h1 style="margin-bottom:0">🏭 Manufacturing Efficiency Prediction</h1>
    <p style="color:#6c757d;margin-top:0">Fill in the process parameters to estimate <b>Efficiency_Score</b>.</p>
    """,
    unsafe_allow_html=True,
)

def badge(text: str, bg: str):
    st.markdown(
        f"""
        <div style="
           display:inline-block;padding:10px 12px;border-radius:10px;
           background:{bg};border:1px solid rgba(0,0,0,.06);font-weight:600;
           ">
           {text}
        </div>
        """,
        unsafe_allow_html=True,
    )

# ---------- Form (two columns like the reference) ----------
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
        "Injection_Temperature": float(Injection_Temperature),
        "Injection_Pressure": float(Injection_Pressure),
        "Cycle_Time": float(Cycle_Time),
        "Cooling_Time": float(Cooling_Time),
        "Material_Viscosity": float(Material_Viscosity),
        "Ambient_Temperature": float(Ambient_Temperature),
        "Operator_Experience": float(Operator_Experience),
        "Machine_Utilization": float(Machine_Utilization),
        "Parts_Per_Hour": float(Parts_Per_Hour),
    }

    try:
        r = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
        if r.status_code != 200:
            st.error(f"API error {r.status_code}: {r.text}")
        else:
            res = r.json()

            st.subheader("Results")
            st.write(f"**Efficiency_Score:** `{res['prediction']}`")

            # Risk chips like the screenshot
            risk = res["risk_level"]
            if risk == "High":
                badge("🔥 Risk Level: High – Excellent performance.", "#ffd6a5")
            elif risk == "Moderate":
                badge("⚙️ Risk Level: Moderate – Can be improved.", "#fff3bf")
            else:
                badge("🟢 Risk Level: Low – Focus on optimization.", "#e7f5ff")

            st.caption(f"Model used: **{res['model_used']}**")

            with st.expander("Details sent to API"):
                st.json(res["details"])

    except Exception as e:
        st.error(f"Failed to call API: {e}")

st.caption("FastAPI + Streamlit • If model files exist, ML model is used; otherwise safe fallback is applied.")
