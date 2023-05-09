import streamlit as st
import numpy as np
import pickle
import pandas as pd

scaler = pickle.load(open('scaler.pkl', 'rb'))
model = pickle.load(open('svm_model.pkl', 'rb'))

def predict_purchase(Age, EstimatedSalary):
    # Create a DataFrame from the inputs
    input_features_df = pd.DataFrame({
        'Age': [Age],
        'EstimatedSalary': [EstimatedSalary]
    })
    # Scale the inputs using the StandardScaler
    scaled_features = scaler.transform(input_features_df)

    # Make a prediction using the model
    prediction = model.predict(scaled_features)
    return prediction

def main():
    st.title("Purchase Prediction")

    # Create input fields for user input
    Age = st.slider("Age", 18, 60, 37)
    EstimatedSalary = st.slider("Estimated Salary", 15000, 150000, 50000)
   
    # When the user clicks the "Predict" button, make a prediction
    if st.button("Predict"):
        prediction = predict_purchase(Age, EstimatedSalary)
        if prediction == 1:
            st.write("The customer is likely to make a purchase!")
        else:
            st.write("The customer is unlikely to make a purchase!")
        
if __name__ == '__main__':
    main()
