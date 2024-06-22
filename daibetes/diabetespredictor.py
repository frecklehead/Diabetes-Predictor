#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 22 08:00:37 2024

@author: frecklehead
"""

import numpy as np
import pickle
import streamlit as st
import os
import sklearn  # Ensure sklearn is imported

# Define the model path
model_path = os.path.join(os.path.dirname(__file__), 'diabetes', 'model', 'trained_model.sav')

# Load the model
try:
    if os.path.exists(model_path):
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
        st.success("Model loaded successfully!")
    else:
        st.error(f"Model file not found at {model_path}")
        st.stop()
except ModuleNotFoundError as e:
    st.error(f"Required module not found: {e}")
    st.stop()
except FileNotFoundError as e:
    st.error(f"File not found: {e}")
    st.stop()
except Exception as e:
    st.error(f"An error occurred: {e}")
    st.stop()

# Prediction function
def prediction(input):
    input_data_as_numpy_array = np.asarray(input, dtype=float)
    # Reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
    
    # Make prediction
    prediction = model.predict(input_data_reshaped)
    if prediction[0] == 0:
        return 'The person is not diabetic'
    else:
        return 'The person is diabetic'

# Main function to define Streamlit app
def main():
    st.title('Diabetes Predictor')
    
    # Input fields
    Pregnancy = st.text_input('Pregnancies')
    Glucose = st.text_input('Glucose')
    BloodPressure = st.text_input('Blood Pressure')
    SkinThickness = st.text_input('Skin Thickness')
    Insulin = st.text_input('Insulin')
    BMI = st.text_input('BMI')
    DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function')
    Age = st.text_input('Age')
    
    diagnosis = ''
    
    # Button for Prediction
    if st.button('Diabetes Prediction Result'):
        try:
            inputs = [Pregnancy, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]
            diagnosis = prediction(inputs)
        except ValueError as e:
            st.error(f"Invalid input: {e}")
        except Exception as e:
            st.error(f"An error occurred: {e}")
            
    st.success(diagnosis)

if __name__ == '__main__':
    main()
