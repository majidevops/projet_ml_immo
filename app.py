import streamlit as st
import joblib
import numpy as np

model = joblib.load("model.pkl")
st.title("🏠 Estimation Immobilière")

# Saisie simplifiée
med_inc = st.number_input("Revenu médian", value=8.3)
if st.button("Estimer"):
    features = np.array([[med_inc, 41.0, 6.98, 1.02, 322.0, 2.55, 37.88, -122.23]])
    st.success(f"Prix estimé : {model.predict(features)[0] * 100000:.2f} $")