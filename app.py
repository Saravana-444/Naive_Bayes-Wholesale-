import streamlit as st
import pickle
import numpy as np

# Load trained model
with open("GB.pkl", "rb") as file:
    GB = pickle.load(file)

st.set_page_config(page_title="Wholesale Customer Prediction", layout="centered")

st.title("🛒 Wholesale Customer Classification")
st.write("Enter customer spending details")

# -------- INPUT FEATURES --------
Fresh = st.number_input("Fresh", min_value=0.0)
Milk = st.number_input("Milk", min_value=0.0)
Grocery = st.number_input("Grocery", min_value=0.0)
Frozen = st.number_input("Frozen", min_value=0.0)
Detergents_Paper = st.number_input("Detergents_Paper", min_value=0.0)
Delicassen = st.number_input("Delicassen", min_value=0.0)

# -------- PREDICTION --------
if st.button("Predict Customer Class"):
    input_data = np.array([[Fresh, Milk, Grocery, Frozen,
                            Detergents_Paper, Delicassen]])

    prediction = model.predict(input_data)
    prediction_proba = model.predict_proba(input_data)

    st.success(f"Predicted Class: {prediction[0]}")
    st.write("Prediction Probability:", prediction_proba)
