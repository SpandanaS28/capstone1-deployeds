# main.py
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, conlist
from typing import List
import pandas as pd
import io

app = FastAPI(
    title="Demo Inference API",
    description="FastAPI backend used by a Streamlit frontend.",
    version="1.0.0",
)

# --- CORS (adjust origins as needed) ---
origins = [
    "http://localhost",
    "http://127.0.0.1",
    "http://localhost:8501",
    "http://127.0.0.1:8501",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,      # or ["*"] during local dev
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Schemas ---
class PredictRequest(BaseModel):
    features: list[float] = Field(..., description="Numeric features", min_length=1)



class PredictResponse(BaseModel):
    prediction: float
    details: dict

# --- Routes ---
@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest):
    # Dummy logic: mean of features (replace with your ML model)
    feats = payload.features
    mean_val = sum(feats) / len(feats)
    # Example “prediction”: a simple function of mean
    pred = mean_val * 1.23
    return PredictResponse(
        prediction=pred,
        details={
            "n_features": len(feats),
            "mean": mean_val,
            "sum": sum(feats),
        },
    )

@app.post("/upload-csv")
async def upload_csv(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Please upload a .csv file")
    try:
        content = await file.read()
        df = pd.read_csv(io.BytesIO(content))
        return {
            "filename": file.filename,
            "rows": int(df.shape[0]),
            "cols": int(df.shape[1]),
            "columns": list(df.columns),
            "preview": df.head(5).to_dict(orient="records"),
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse CSV: {e}")

# --- Local dev entrypoint ---
# Run: uvicorn main:app --reload --port 8000
