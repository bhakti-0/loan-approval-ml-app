from fastapi import FastAPI, UploadFile, File
import pandas as pd
import os

from backend.preprocessing import preprocess_train, preprocess_predict
from backend.model import train_model, save_model, load_model

from sklearn.metrics import accuracy_score, classification_report


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

@app.post("/test")
async def test(file: UploadFile = File(...)):
    if not os.path.exists(MODEL_PATH):
        return {"error": "Model not trained yet"}

    # Load test data
    df = pd.read_csv(file.file)

    # Load trained model bundle
    bundle = load_model()
    model = bundle["model"]
    scaler = bundle["scaler"]
    columns = bundle["columns"]

    # Separate target
    if "Loan_Status" not in df.columns:
        return {"error": "Loan_Status column not found in test data"}

    y_true = df["Loan_Status"].map({"Y": 1, "N": 0})
    X_df = df.drop(columns=["Loan_Status"])

    # Preprocess test data
    X_test = preprocess_predict(X_df, scaler, columns)

    # Predict
    y_pred = model.predict(X_test)

    # Metrics
    accuracy = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, output_dict=True)

    return {
        "accuracy": round(float(accuracy), 3),
        "classification_report": report
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
