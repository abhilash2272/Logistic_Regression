import streamlit as st
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(page_title="Customer Churn Analysis", layout="centered")
st.title("📊 Customer Churn Prediction Dashboard")

# -----------------------------
# Load Dataset (ROBUST)
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv(
        "WA_Fn-UseC_-Telco-Customer-Churn (1).csv",
        engine="python",
        encoding="utf-8",
        on_bad_lines="skip"
    )
    df.columns = df.columns.str.strip()   # remove hidden spaces
    return df

df = load_data()

# -----------------------------
# SAFETY CHECK: Find Churn Column
# -----------------------------
possible_targets = ["Churn", "churn", "CHURN"]

target_col = None
for col in df.columns:
    if col.strip().lower() == "churn":
        target_col = col
        break

if target_col is None:
    st.error("❌ Target column 'Churn' not found in dataset.")
    st.write("Available columns:", df.columns.tolist())
    st.stop()

# -----------------------------
# Preprocessing
# -----------------------------
df[target_col] = df[target_col].astype(str).str.strip()
df[target_col] = df[target_col].map({"Yes": 1, "No": 0})

df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df.dropna(inplace=True)

X = df[["tenure", "MonthlyCharges", "TotalCharges"]]
y = df[target_col]

# -----------------------------
# Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# Scaling
# -----------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -----------------------------
# Model Training
# -----------------------------
model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)

# -----------------------------
# Evaluation
# -----------------------------
y_pred = model.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=["No Churn", "Churn"])

# -----------------------------
# Predict for Entire Dataset
# -----------------------------
df["Predicted_Churn"] = model.predict(scaler.transform(X))

stay_count = (df["Predicted_Churn"] == 0).sum()
leave_count = (df["Predicted_Churn"] == 1).sum()

# -----------------------------
# Display Results
# -----------------------------
st.subheader("📌 Model Performance")
st.metric("Accuracy", f"{accuracy:.2f}")

st.subheader("📄 Classification Report")
st.text(report)

st.subheader("📊 Confusion Matrix")
cm_df = pd.DataFrame(
    cm,
    index=["Actual No Churn", "Actual Churn"],
    columns=["Predicted No Churn", "Predicted Churn"]
)
st.dataframe(cm_df)

# -----------------------------
# Churn Summary
# -----------------------------
st.subheader("🚨 Churn Prediction Summary")
st.metric("Total Customers", len(df))
st.metric("Customers Likely to Leave", leave_count)
st.metric("Customers Likely to Stay", stay_count)

# -----------------------------
# Sample Predictions
# -----------------------------
st.subheader("📋 Sample Predictions")
st.dataframe(
    df[["tenure", "MonthlyCharges", "TotalCharges", "Predicted_Churn"]]
    .replace({"Predicted_Churn": {0: "No", 1: "Yes"}})
    .head(10)
)

# -----------------------------
# User Prediction
# -----------------------------
st.markdown("---")
st.subheader("🔮 Predict Churn for New Customer")

tenure = st.slider("Tenure (months)", 0, 72, 12)
monthly = st.slider("Monthly Charges", 0.0, 150.0, 70.0)
total = st.slider("Total Charges", 0.0, 10000.0, 2000.0)

if st.button("Predict Churn"):
    input_df = pd.DataFrame(
        [[tenure, monthly, total]],
        columns=["tenure", "MonthlyCharges", "TotalCharges"]
    )
    input_scaled = scaler.transform(input_df)
    prob = model.predict_proba(input_scaled)[0][1]

    if prob >= 0.5:
        st.error(f"⚠️ Likely to Leave (Probability: {prob:.2f})")
    else:
        st.success(f"✅ Likely to Stay (Probability: {prob:.2f})")

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.caption("Logistic Regression | Streamlit | Customer Churn Analysis")
