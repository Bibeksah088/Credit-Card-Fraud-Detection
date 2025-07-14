import streamlit as st
import pandas as pd
import joblib

st.title("Credit Card Fraud Detection")

# Load model
model = joblib.load("fraud_model.pkl")

# User input
st.subheader("Enter Transaction Details:")
v_features = [st.number_input(f"V{i}", value=0.0) for i in range(1, 29)]
amount = st.number_input("Transaction Amount", value=0.0)

# Normalize Amount like training
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaled_amount = scaler.fit_transform([[amount]])[0][0]

# Make prediction
if st.button("Predict Fraud"):
    data = [[*v_features, scaled_amount]]
    prediction = model.predict(data)
    result = "Fraudulent Transaction" if prediction[0] == 1 else "Legit Transaction"
    st.write(f"🔍 Prediction: **{result}**")

