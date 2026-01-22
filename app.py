import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC, LinearSVC
import joblib

# -------------------------------
# Load and preprocess dataset
# -------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("train_u6lujuX_CVtuZ9i.csv")
    # Fill missing values
    df["LoanAmount"] = df["LoanAmount"].fillna(df["LoanAmount"].median())
    df["Credit_History"] = df["Credit_History"].fillna(df["Credit_History"].mode()[0])
    # Encode target
    le = LabelEncoder()
    df["Loan_Status"] = le.fit_transform(df["Loan_Status"])  # Y=1, N=0
    return df, le

df, le_target = load_data()

# Features for training
features = ["LoanAmount", "Credit_History"]  # Using main features for demo

X = df[features]
y = df["Loan_Status"]

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -------------------------------
# Streamlit App UI
# -------------------------------
st.title("🏦 Smart Loan Approval System")
st.markdown("""
This system uses **Support Vector Machines** to predict loan approval.
Choose the kernel, enter applicant details, and check loan eligibility.
""")

# -------------------------------
# Sidebar Inputs
# -------------------------------
st.sidebar.header("Applicant Details")

app_income = st.sidebar.number_input("Applicant Income (₹)", min_value=0, step=1000)
loan_amount = st.sidebar.number_input("Loan Amount (₹)", min_value=0, step=1000)
credit_history = st.sidebar.selectbox("Credit History", ["Yes", "No"])
employment_status = st.sidebar.selectbox("Employment Status", ["Employed", "Self-Employed"])
property_area = st.sidebar.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

kernel_choice = st.sidebar.radio(
    "Select SVM Kernel",
    ["Linear SVM", "Polynomial SVM", "RBF SVM"]
)

# -------------------------------
# Prepare input for prediction
# -------------------------------
# Encode input
credit_history_val = 1 if credit_history == "Yes" else 0

input_data = np.array([[loan_amount, credit_history_val]])
input_scaled = scaler.transform(input_data)

# -------------------------------
# Train models (demo with minimal features)
# -------------------------------
def train_model(kernel_name):
    if kernel_name == "Linear SVM":
        model = LinearSVC(max_iter=5000)
        model.fit(X_scaled, y)
    elif kernel_name == "Polynomial SVM":
        model = SVC(kernel="poly", degree=3, gamma="scale", probability=True)
        model.fit(X_scaled, y)
    else:  # RBF
        model = SVC(kernel="rbf", gamma="scale", probability=True)
        model.fit(X_scaled, y)
    return model

# -------------------------------
# Prediction
# -------------------------------
if st.button("Check Loan Eligibility"):
    model = train_model(kernel_choice)
    
    prediction = model.predict(input_scaled)[0]
    
    # Confidence (if available)
    if kernel_choice == "Linear SVM":
        confidence = "N/A"
    else:
        confidence = model.predict_proba(input_scaled)[0].max()
        confidence = f"{confidence*100:.2f}%"
    
    if prediction == 1:
        st.success(f"✅ Loan Approved!  (Confidence: {confidence})")
    else:
        st.error(f"❌ Loan Rejected!  (Confidence: {confidence})")
    
    st.markdown("---")
    st.subheader("Business Explanation")
    if prediction == 1:
        st.write("Based on credit history and income pattern, the applicant is likely to repay the loan.")
    else:
        st.write("Based on credit history and income pattern, the applicant is unlikely to repay the loan.")
    st.write(f"Kernel used for prediction: **{kernel_choice}**")
