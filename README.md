# Loan Approval Prediction System

An end-to-end loan approval prediction application built using Machine Learning, FastAPI, and Streamlit.  
This project demonstrates a complete ML workflow including data preprocessing, model training, inference, and a user-facing web interface.

---

## Overview

This project implements a loan approval prediction system that enables:

- Training a machine learning model using a loan dataset
- Predicting loan approval for individual applicants
- Returning both approval decision and probability
- Providing user-facing explanations for possible rejection reasons

The system is designed to resemble a real-world loan eligibility assessment pipeline.

---

## Features

- Dataset upload and model training
- REST API for prediction using FastAPI
- Interactive web interface built with Streamlit
- Probability-based prediction using Logistic Regression
- Robust preprocessing pipeline:
  - Missing value handling
  - One-hot encoding
  - Feature scaling
  - Feature alignment between training and inference
- Rule-based explanation for loan rejection

---

## Machine Learning Details

- Algorithm: Logistic Regression
- Target Variable: Loan_Status
- Model Output:
  - Binary decision (Approved / Rejected)
  - Approval probability using `predict_proba`
- Preprocessing Techniques:
  - Handling missing values
  - Encoding categorical variables
  - Standardization using StandardScaler
  - Ensuring identical feature sets during training and prediction

---

## Tech Stack

| Layer | Technology |
|------|------------|
| Programming Language | Python |
| Backend Framework | FastAPI |
| Frontend Framework | Streamlit |
| Machine Learning | scikit-learn |
| Data Processing | Pandas, NumPy |
| Model Persistence | Joblib |
| Version Control | Git, GitHub |

---

## Project Structure

loan-approval-ml-app/
│
├── backend/
│ ├── main.py # FastAPI application
│ ├── preprocessing.py # Data preprocessing logic
│ ├── model.py # Model training and loading
│ └── init.py
│
├── frontend/
│ └── app.py # Streamlit web interface
│
├── models/
│ └── loan_model.pkl # Saved model file (gitignored)
│
├── dataset/
│ └── train.csv # Dataset (optional, gitignored)
│
├── requirements.txt
├── README.md
└── .gitignore
