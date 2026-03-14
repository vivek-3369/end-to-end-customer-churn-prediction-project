import streamlit as st
import pandas as pd

from src.utils import load_obj
from src.pipelines.prediction_pipeline import CustomData, PredictionPipeline


st.set_page_config(page_title="Churn Prediction", page_icon="📈", layout="centered")

st.title("📊 Customer Churn Prediction App")
st.markdown("Enter the customer profile details below to predict their churn likelihood 🔍")

gender = st.selectbox("Gender", ["Male", "Female"])
seniorcitizen = st.selectbox("Senior Citizen", ["Yes", "No"])
partner = st.selectbox("Partner", ["Yes", "No"])
dependents = st.selectbox("Dependents", ["Yes", "No"])
tenure = st.slider("Tenure", 1, 72, 1)
phoneservice = st.selectbox("Phone Service", ["Yes", "No"])
multiplelines = st.selectbox("Multiple Lines", ["Yes", "No"])
internetservice = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
onlinesecurity = st.selectbox("Online Security", ["Yes", "No"])
onlinebackup = st.selectbox("Online Backup", ["Yes", "No"])
deviceprotection = st.selectbox("Device Protection", ["Yes", "No"])
techsupport = st.selectbox("Tech Support", ["Yes", "No"])
streamingtv = st.selectbox("Streaming TV", ["Yes", "No"])
streamingmovies = st.selectbox("Streaming Movies", ["Yes", "No"])
contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
paperlessbilling = st.selectbox("Paperless Billing", ["Yes", "No"])
payment = st.selectbox("Payment Method", ["Electronic check", "Mailed check","Bank transfer (automatic)" , "Credit card (automatic)"])
monthly_charges = st.number_input("Monthly Charges", min_value=10.00, max_value=500.00, value=10.00, format="%.2f")
total_charges = st.number_input("Total Charges", min_value=10.00, max_value=100000.00, value=10.00, format="%.2f")


if st.button("Predict Churn 📉") :
    user_data = CustomData(gender, seniorcitizen, partner, dependents, tenure, phoneservice, 
                      multiplelines, internetservice, onlinesecurity, onlinebackup, deviceprotection, 
                      techsupport, streamingtv, streamingmovies, contract, paperlessbilling, 
                      payment, monthly_charges, total_charges)
    
    data_df = user_data.get_input_as_df()

    prediction_pipeline = PredictionPipeline()
    prob = prediction_pipeline.prediction(data_df)

    # To display the Churn Probabiltity
    st.divider()
    st.markdown("### 📋 Prediction Results")
    st.metric("Churn Probability:", f"{prob * 100:.2f}%")

    if prob > 0.6 :
        st.error("🚨 High Risk of Churn: Customer is likely to leave.")

    elif prob > 0.3 :
        st.warning("⚠️ Moderate Risk of Churn: Consider targeted retention.")
    else :
        st.success("✅ Low Risk of Churn: Customer is likely to stay.")