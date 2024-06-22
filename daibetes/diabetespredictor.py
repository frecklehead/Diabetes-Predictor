#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 22 08:00:37 2024

@author: frecklehead
"""
import numpy as np
import pickle
import streamlit as st

model=pickle.load(open('/daibetes/trained_model.sav','rb'))


def prediction(input):
    imput_data_as_numpy_array=np.asarray(input)
    # reshape the array as we are predicting for one instance
    input_data_reshaped=imput_data_as_numpy_array.reshape(1,-1)
    #standardize the input


    prediction=model.predict(input_data_reshaped)
    print(prediction)
    if(prediction[0]==0):
     return('the person is not diabetic')
    else:
     return('the person is diabetic')
 
def main():
    st.title('Diabetes PRedictor')
    
    Pregnancy=st.text_input('Pregnancies')
    Glucose=st.text_input('Glucose')
    BloodPressure=st.text_input('Blood Pressure')
    SkinThickness=st.text_input('Skin Thickness')
    Insulin=st.text_input('Insulin')
    BMI=st.text_input('BMI')
    DiabetesPedigreeFunction=st.text_input('Diabetes Pedigree Function')
    Age=st.text_input('Age')
    
    diagnosis=''
    
    if st.button('Diabetes Prediction Result'):
        
        diagnosis=prediction([Pregnancy,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age])
    st.success(diagnosis)
    
    
    
if __name__ =='__main__':
    main()
