import pandas as pd
from sklearn.preprocessing import StandardScaler

# ================= TRAIN PREPROCESS =================
def preprocess_train(df: pd.DataFrame):
    df = df.copy()

    # Drop ID column
    if "Loan_ID" in df.columns:
        df.drop(columns=["Loan_ID"], inplace=True)

    # Target
    y = df["Loan_Status"].map({"Y": 1, "N": 0})
    X = df.drop(columns=["Loan_Status"])

    # Handle Dependents
    if "Dependents" in X.columns:
        X["Dependents"] = X["Dependents"].replace("3+", 3)
        X["Dependents"] = pd.to_numeric(X["Dependents"], errors="coerce")

    # Fill missing values
    for col in X.select_dtypes(include="object").columns:
        X[col] = X[col].fillna(X[col].mode()[0])

    for col in X.select_dtypes(include="number").columns:
        X[col] = X[col].fillna(X[col].median())

    # One-hot encoding
    X = pd.get_dummies(X, drop_first=True)

    # Final NaN safety
    X = X.fillna(0)

    # Scaling
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X),
        columns=X.columns
    )

    return X_scaled, y, scaler, X.columns.tolist()


# ================= PREDICT PREPROCESS =================
def preprocess_predict(df: pd.DataFrame, scaler, train_columns):
    df = df.copy()

    if "Loan_ID" in df.columns:
        df.drop(columns=["Loan_ID"], inplace=True)

    if "Dependents" in df.columns:
        df["Dependents"] = df["Dependents"].replace("3+", 3)
        df["Dependents"] = pd.to_numeric(df["Dependents"], errors="coerce")

    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].fillna(df[col].mode()[0])

    for col in df.select_dtypes(include="number").columns:
        df[col] = df[col].fillna(df[col].median())

    df = pd.get_dummies(df, drop_first=True)

    # Align columns with training
    for col in train_columns:
        if col not in df.columns:
            df[col] = 0

    df = df[train_columns]

    df = pd.DataFrame(
        scaler.transform(df),
        columns=train_columns
    )

    return df
