import streamlit as st
import pickle
import numpy as np

# App title
st.title("ML Prediction App")

st.write("Enter values to get prediction")

# Load trained model
@st.cache_resource
def load_model():
    with open("knn_model.pkl", "rb") as file:
        model = pickle.load(file)
    return model

model = load_model()

# User inputs (change labels & count as per your model)
feature1 = st.number_input("Feature 1", value=0.0)
feature2 = st.number_input("Feature 2", value=0.0)
feature3 = st.number_input("Feature 3", value=0.0)
feature4 = st.number_input("Feature 4", value=0.0)

# Predict button
if st.button("Predict"):
    input_data = np.array([[feature1, feature2, feature3, feature4]])
    prediction = model.predict(input_data)
    st.success(f"Prediction: {prediction[0]}")
