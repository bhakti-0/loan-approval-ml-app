from sklearn.linear_model import LogisticRegression
import joblib

def train_model(X, y):
    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)
    return model

def save_model(model, scaler, columns, path="models/loan_model.pkl"):
    joblib.dump(
        {
            "model": model,
            "scaler": scaler,
            "columns": columns
        },
        path
    )

def load_model(path="models/loan_model.pkl"):
    return joblib.load(path)
