import streamlit as st
import numpy as np
import pickle

# Load the model
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

# Streamlit app
st.title("Iris Flower Classification")
st.write("Enter the flower's measurements to classify its species:")

# Input features
sepal_length = st.number_input("Sepal Length (cm)", min_value=0.0)
sepal_width = st.number_input("Sepal Width (cm)", min_value=0.0)
petal_length = st.number_input("Petal Length (cm)", min_value=0.0)
petal_width = st.number_input("Petal Width (cm)", min_value=0.0)

# Prediction button
if st.button("Predict"):
    features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = model.predict(features)
    st.write(f"The predicted species is: **{prediction[0]}**")
