import streamlit as st
import pickle
import numpy as np
from model import train_model, load_data

# Title and description
st.title("Boston Housing Price Prediction")
st.write("This app predicts house prices based on input features.")

# Train the model
model, columns = train_model()

# Create input fields for the features
st.sidebar.header("Input Features")
user_input = {}
for column in columns:
    user_input[column] = st.sidebar.number_input(f"{column}", 0.0, 100.0, 0.0)

# Prediction
if st.sidebar.button("Predict"):
    input_array = np.array([list(user_input.values())]).reshape(1, -1)
    prediction = model.predict(input_array)
    st.write(f"Predicted House Price: ${prediction[0]:.2f}")
