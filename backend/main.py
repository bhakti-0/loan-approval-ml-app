from fastapi import FastAPI, UploadFile, File
import pandas as pd
import os

from backend.preprocessing import preprocess_train, preprocess_predict
from backend.model import train_model, save_model, load_model

app = FastAPI(title="Loan Approval Prediction API")

MODEL_PATH = "models/loan_model.pkl"


@app.post("/train")
async def train(file: UploadFile = File(...)):
    df = pd.read_csv(file.file)

    X, y, scaler, columns = preprocess_train(df)
    model = train_model(X, y)

    save_model(model, scaler, columns)

    return {
        "message": "Model trained successfully",
        "features_used": len(columns)
    }


@app.post("/predict")
async def predict(data: dict):
    if not os.path.exists(MODEL_PATH):
        return {"error": "Model not trained"}

    bundle = load_model()
    model = bundle["model"]
    scaler = bundle["scaler"]
    columns = bundle["columns"]

    df = pd.DataFrame([data])
    X = preprocess_predict(df, scaler, columns)

    prediction = model.predict(X)[0]
    probability = model.predict_proba(X)[0][1]

    return {
    "loan_status": "Approved" if prediction == 1 else "Rejected",
    "approval_probability": float(probability)  # NO round here
}
