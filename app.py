# app.py
import os
import requests
import streamlit as st

st.set_page_config(page_title="Streamlit ↔ FastAPI Demo", page_icon="🧪", layout="centered")

# --- Config ---
DEFAULT_API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")
with st.sidebar:
    st.header("Settings")
    api_url = st.text_input("FastAPI URL", value=DEFAULT_API_URL, help="E.g., http://127.0.0.1:8000")
    if st.button("Ping /health"):
        try:
            r = requests.get(f"{api_url}/health", timeout=10)
            st.success(f"Health: {r.json()}")
        except Exception as e:
            st.error(f"Failed to reach API: {e}")

st.title("Streamlit Frontend for FastAPI")
st.caption("Starter UI with numeric prediction and CSV analysis")

tab1, tab2 = st.tabs(["🔢 Predict", "📄 CSV Upload"])

# --- Predict tab ---
with tab1:
    st.subheader("Numeric Features → Prediction")
    st.write("Enter comma-separated numbers (e.g., `1.2, 3, 4.5`)")

    raw = st.text_input("Features", value="1.0, 2.0, 3.0")
    if st.button("Predict"):
        try:
            features = [float(x.strip()) for x in raw.split(",") if x.strip() != ""]
            payload = {"features": features}
            r = requests.post(f"{api_url}/predict", json=payload, timeout=20)
            if r.status_code == 200:
                data = r.json()
                st.success(f"Prediction: **{data['prediction']:.4f}**")
                with st.expander("Details"):
                    st.json(data["details"])
            else:
                st.error(f"API error {r.status_code}: {r.text}")
        except ValueError:
            st.error("Please provide valid numbers separated by commas.")
        except Exception as e:
            st.error(f"Request failed: {e}")

# --- CSV tab ---
with tab2:
    st.subheader("Upload a CSV")
    up = st.file_uploader("Choose a .csv file", type=["csv"])
    if up and st.button("Analyze CSV"):
        try:
            files = {"file": (up.name, up.getvalue(), "text/csv")}
            r = requests.post(f"{api_url}/upload-csv", files=files, timeout=60)
            if r.status_code == 200:
                info = r.json()
                st.success(f"Parsed **{info['filename']}** — Rows: {info['rows']}, Cols: {info['cols']}")
                st.write("Columns:", ", ".join(info["columns"]))
                st.write("Preview (first 5 rows):")
                st.json(info["preview"])
            else:
                st.error(f"API error {r.status_code}: {r.text}")
        except Exception as e:
            st.error(f"Request failed: {e}")

st.markdown("---")
st.caption("Tip: set an environment variable `API_URL` for deployment, or edit it in the sidebar.")
