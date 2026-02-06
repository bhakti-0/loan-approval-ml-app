import streamlit as st
import requests

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Loan Approval System",
    layout="wide"
)

BACKEND_URL = "http://127.0.0.1:8000"

# ---------------- HEADER ----------------
st.markdown(
    """
    <h1 style='text-align: center;'>Loan Approval Prediction System</h1>
    <p style='text-align: center; color: gray;'>
        AI-powered system to evaluate loan eligibility using machine learning
    </p>
    <hr>
    """,
    unsafe_allow_html=True
)

# ================= TRAIN MODEL =================
st.subheader("Train Model")

train_file = st.file_uploader(
    "Upload Training Dataset (CSV)",
    type=["csv"]
)

if st.button("Train Model"):
    if train_file:
        with st.spinner("Training model..."):
            response = requests.post(
                f"{BACKEND_URL}/train",
                files={"file": train_file}
            )

        if response.status_code == 200:
            st.success("Model trained successfully")
            st.json(response.json())
        else:
            st.error("Training failed")
            st.code(response.text)
    else:
        st.warning("Please upload a training CSV file")

st.divider()

# ================= TEST MODEL =================
st.subheader("Test Model")

test_file = st.file_uploader(
    "Upload Test Dataset (CSV with Loan_Status)",
    type=["csv"],
    key="test_file"
)

if st.button("Test Model"):
    if test_file is None:
        st.warning("Please upload a test dataset.")
    else:
        with st.spinner("Evaluating model..."):
            response = requests.post(
                f"{BACKEND_URL}/test",
                files={"file": test_file}
            )

        if response.status_code == 200:
            result = response.json()

            st.success("Model evaluated successfully")

            st.metric(
                label="Accuracy",
                value=f"{result['accuracy'] * 100:.2f} %"
            )

            st.markdown("### Classification Report")

            report = result["classification_report"]

            for label in ["0", "1"]:
                st.markdown(f"**Class {label}**")
                st.write(
                    f"Precision: {report[label]['precision']:.2f}  |  "
                    f"Recall: {report[label]['recall']:.2f}  |  "
                    f"F1-score: {report[label]['f1-score']:.2f}"
                )

        else:
            st.error("Model evaluation failed")
            st.code(response.text)

st.divider()


# ================= PREDICTION FORM =================
st.subheader("Predict Loan Approval")

col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["Male", "Female"])
    married = st.selectbox("Marital Status", ["Yes", "No"])
    dependents = st.number_input(
        "Number of Dependents",
        min_value=0,
        max_value=5,
        step=1
    )
    education = st.selectbox(
        "Education Level",
        ["Graduate", "Not Graduate"]
    )

with col2:
    employment_type = st.selectbox(
        "Employment Type",
        ["Salaried", "Self-Employed"],
        help="Salaried: fixed regular income | Self-Employed: business or freelance income"
    )

st.markdown("### Income Details (Yearly, INR)")

applicant_income = st.number_input(
    "Applicant Income (Yearly)",
    min_value=0,
    step=10_000
)

coapplicant_income = st.number_input(
    "Coapplicant Income (Yearly)",
    min_value=0,
    step=10_000
)

st.markdown("### Loan Details")

loan_amount = st.number_input(
    "Loan Amount (INR)",
    min_value=0,
    step=10_000
)

loan_term = st.number_input(
    "Loan Term (Months)",
    min_value=1,
    step=1
)

credit_history = st.selectbox(
    "Credit History",
    ["Good", "Bad"],
    help="Good credit history means past loans were repaid on time"
)

property_area = st.selectbox(
    "Property Area",
    ["Urban", "Semiurban", "Rural"]
)

# ================= PREDICT BUTTON =================
if st.button("Predict Loan Approval"):
    payload = {
        "Gender": 1 if gender == "Male" else 0,
        "Married": 1 if married == "Yes" else 0,
        "Dependents": dependents,
        "Education": 1 if education == "Graduate" else 0,
        "Self_Employed": 1 if employment_type == "Self-Employed" else 0,
        "ApplicantIncome": applicant_income,
        "CoapplicantIncome": coapplicant_income,
        "LoanAmount": loan_amount,
        "Loan_Amount_Term": loan_term,
        "Credit_History": 1 if credit_history == "Good" else 0,
        "Property_Area": property_area
    }

    with st.spinner("Predicting..."):
        response = requests.post(
            f"{BACKEND_URL}/predict",
            json=payload
        )

    if response.status_code == 200:
        result = response.json()

        st.divider()
        st.subheader("Prediction Result")

        if result["loan_status"] == "Approved":
            st.success("Loan Approved")
        else:
            st.error("Loan Rejected")

        st.info(
            f"Approval Probability: {result['approval_probability'] * 100:.2f} %"
        )

        # -------- RULE-BASED EXPLANATION --------
        st.markdown("### Explanation")

        reasons = []

        if credit_history == "Bad":
            reasons.append("Poor credit history negatively impacts approval.")
        if applicant_income + coapplicant_income < loan_amount:
            reasons.append("Total income is low compared to requested loan amount.")
        if loan_term < 120:
            reasons.append("Short loan tenure increases repayment burden.")
        if employment_type == "Self-Employed":
            reasons.append("Self-employed applicants are considered higher risk due to income variability.")

        if reasons:
            for r in reasons:
                st.write("- ", r)
        else:
            st.write("- Applicant satisfies most eligibility criteria.")

    else:
        st.error("Internal Server Error")
        st.code(response.text)

# ================= INFORMATION SECTION =================
st.divider()
st.subheader("Loan Information and Helpful Tips")

st.markdown(
    """
    **How loan approval works**

    Loan approval is primarily based on repayment capacity, credit history,
    income stability, and loan amount requested.

    **Tips to improve approval chances**
    - Maintain a good credit history by paying EMIs on time
    - Keep loan amount reasonable relative to income
    - Opt for longer loan tenures to reduce EMI burden
    - Maintain stable employment and income records

    **Disclaimer**
    This system provides AI-based predictions for informational purposes only.
    Final approval decisions depend on the lender's internal policies.
    """
)
