import streamlit as st
import requests

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Loan Approval System",
    page_icon="🏦",
    layout="wide"
)

BACKEND_URL = "http://127.0.0.1:8000"

# ---------------- HEADER ----------------
st.markdown(
    """
    <h1 style='text-align: center;'>🏦 Loan Approval Prediction System</h1>
    <p style='text-align: center; color: gray;'>
        AI-powered system to evaluate loan eligibility using machine learning
    </p>
    <hr>
    """,
    unsafe_allow_html=True
)

# ================= TRAIN MODEL =================
st.subheader("📊 Train Model")

with st.container():
    train_file = st.file_uploader("Upload Training Dataset (CSV)", type=["csv"])

    if st.button("🚀 Train Model"):
        if train_file:
            with st.spinner("Training model..."):
                response = requests.post(
                    f"{BACKEND_URL}/train",
                    files={"file": train_file}
                )

            if response.status_code == 200:
                result = response.json()
                st.success("✅ Model trained successfully")

                st.write("**Model Summary:**")
                st.json(result)
            else:
                st.error("❌ Training failed")
                st.code(response.text)
        else:
            st.warning("Please upload a training CSV file")

st.divider()

# ================= PREDICTION FORM =================
st.subheader("🔮 Predict Loan Approval")

col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["Male", "Female"])
    married = st.selectbox("Married", ["Yes", "No"])
    dependents = st.number_input(
        "Dependents",
        min_value=0,
        max_value=5,
        step=1
    )
    education = st.selectbox("Education", ["Graduate", "Not Graduate"])

with col2:
    self_employed = st.selectbox(
        "Self Employed",
        ["No", "Yes"],
        help="Self-employed means you do not receive a fixed monthly salary and work independently (freelancer, business owner, contractor, etc.)"
    )
    applicant_income = st.number_input(
        "Applicant Income (₹)",
        min_value=0,
        step=10_000
    )
    coapplicant_income = st.number_input(
        "Coapplicant Income (₹)",
        min_value=0,
        step=10_000
    )

loan_amount = st.number_input(
    "Loan Amount (₹ in thousands)",
    min_value=0,
    step=1
)

loan_term = st.number_input(
    "Loan Amount Term (months)",
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
if st.button("🔍 Predict Loan Approval"):
    payload = {
        "Gender": gender,
        "Married": married,
        "Dependents": dependents,
        "Education": education,
        "Self_Employed": self_employed,
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
        st.subheader("📌 Prediction Result")

        if result["loan_status"] == "Approved":
            st.success("✅ Loan Approved")
        else:
            st.error("❌ Loan Rejected")

        st.info(
            f"Approval Probability: {result['approval_probability'] * 100:.2f} %"
        )

        # -------- REASONING (RULE-BASED EXPLANATION) --------
        st.markdown("### 🧠 Why this decision?")
        reasons = []

        if credit_history == "Bad":
            reasons.append("Poor credit history significantly reduces approval chances.")
        if applicant_income + coapplicant_income < loan_amount * 10:
            reasons.append("Income is low compared to the requested loan amount.")
        if loan_term < 120:
            reasons.append("Short loan term increases EMI burden.")
        if self_employed == "Yes":
            reasons.append("Self-employed applicants are considered higher risk.")

        if reasons:
            for r in reasons:
                st.write("•", r)
        else:
            st.write("• Applicant meets most eligibility criteria.")

    else:
        st.error("Internal Server Error")
        st.code(response.text)

# ================= PROJECT INFO =================
st.divider()
st.divider()
st.subheader("ℹ️ Loan Information & Helpful Tips")

st.markdown(
    """
    ### 🏦 Understanding Loan Approval

    Loan approval is based on multiple financial and personal factors. Banks and financial institutions
    assess your ability to **repay the loan consistently and on time**.

    ---

    ### ✅ Tips to Improve Loan Approval Chances

    **1. Maintain a Good Credit History**
    - Pay EMIs and credit card bills on time
    - Avoid frequent loan or credit card applications
    - A good credit score builds trust with lenders

    **2. Keep a Healthy Income-to-Loan Ratio**
    - Your total income should comfortably support the loan EMI
    - Lower loan amounts relative to income improve approval chances

    **3. Choose a Suitable Loan Term**
    - Longer loan tenure reduces monthly EMI
    - Very short tenures increase repayment pressure

    **4. Stable Employment Matters**
    - Salaried individuals with steady income are considered lower risk
    - Self-employed applicants may face stricter checks due to income variability

    **5. Property Location Can Influence Approval**
    - Urban and semi-urban properties generally have higher approval rates
    - Rural properties may require additional verification

    ---

    ### ⚠️ Important Note
    This prediction is **AI-based** and meant for **informational purposes only**.
    Final loan approval decisions depend on the bank’s internal policies and document verification.
    """
)
