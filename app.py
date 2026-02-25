import streamlit as st
import pandas as pd
import joblib

# =========================
# Load model and features
# =========================

model = joblib.load("final_model_xgb.pkl")
feature_columns = joblib.load("feature_columns.pkl")

# =========================
# Page config
# =========================

st.set_page_config(
    page_title="Portuguese Bank Term Deposit Predictor",
    page_icon="🏦",
    layout="centered"
)

st.title("🏦 Portuguese Bank Term Deposit Predictor")
st.write("Predict whether customer will subscribe to term deposit")

# =========================
# User Inputs
# =========================

age = st.number_input("Age", 18, 100, 30)

campaign = st.number_input("Campaign Contacts", 1, 50, 1)

pdays = st.number_input("Days since last contact (-1 means never)", -1, 999, -1)

previous = st.number_input("Previous Contacts", 0, 50, 0)

job = st.selectbox("Job", [
    "admin.", "technician", "services", "management",
    "retired", "blue-collar", "unemployed",
    "entrepreneur", "housemaid", "student",
    "self-employed", "unknown"
])

marital = st.selectbox("Marital Status", [
    "married", "single", "divorced"
])

education = st.selectbox("Education", [
    "basic.6y",
    "basic.9y",
    "high.school",
    "professional.course",
    "university.degree",
    "unknown"
])

housing = st.selectbox("Housing Loan", [
    "yes", "no"
])

loan = st.selectbox("Personal Loan", [
    "yes", "no"
])

month = st.selectbox("Last Contact Month", [
    "jan","feb","mar","apr","may","jun",
    "jul","aug","sep","oct","nov","dec"
])

day = st.selectbox("Day of Week", [
    "mon","tue","wed","thu","fri"
])

poutcome = st.selectbox("Previous Campaign Outcome", [
    "success","failure","nonexistent"
])

# =========================
# Prediction
# =========================

if st.button("Predict"):

    # Create empty dataframe with all columns
    input_df = pd.DataFrame(columns=feature_columns)
    input_df.loc[0] = 0

    # Fill numeric values
    input_df.at[0, "age"] = age
    input_df.at[0, "campaign"] = campaign
    input_df.at[0, "pdays"] = pdays
    input_df.at[0, "previous"] = previous

    # Fill categorical one-hot values safely
    if f"job_{job}" in input_df.columns:
        input_df.at[0, f"job_{job}"] = 1

    if f"marital_{marital}" in input_df.columns:
        input_df.at[0, f"marital_{marital}"] = 1

    if f"education_{education}" in input_df.columns:
        input_df.at[0, f"education_{education}"] = 1

    if housing == "yes" and "housing_yes" in input_df.columns:
        input_df.at[0, "housing_yes"] = 1

    if loan == "yes" and "loan_yes" in input_df.columns:
        input_df.at[0, "loan_yes"] = 1

    if f"month_{month}" in input_df.columns:
        input_df.at[0, f"month_{month}"] = 1

    if f"day_of_week_{day}" in input_df.columns:
        input_df.at[0, f"day_of_week_{day}"] = 1

    if f"poutcome_{poutcome}" in input_df.columns:
        input_df.at[0, f"poutcome_{poutcome}"] = 1

    # Predict
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0]
    yes_prob = probability[1]

    st.divider()

    st.metric("Subscription Probability", f"{yes_prob:.2%}")

    if prediction == 1:
      st.success("✅ Customer WILL subscribe to term deposit")
    else:
      st.error("❌ Customer will NOT subscribe to term deposit")