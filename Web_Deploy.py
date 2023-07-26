import numpy as np
import pickle
import streamlit as slt
import pandas as pd
from sklearn.preprocessing import StandardScaler

loaded_model = pickle.load(open('trained_model_diabetes.sav', 'rb'))

def predicting_diabetes(input_array):
    scaler = StandardScaler()

    #Converting to numpy_array

    arr = np.array(input_array).reshape(1, -1)
    array_used = pd.DataFrame(scaler.fit_transform(arr))
    #predicting

    prediction = (loaded_model.predict(array_used)>0.5).astype(int)
    print(prediction)

    if prediction[0] == 0 :
        return "Person Does Not Have Diabetes"
    else:
        return "Person Has Diabetes"

def main():

    #Giving Title to Web Page
    slt.title("Diabetes Predictor Page")

    #Taking User Input

    Pregnancies = slt.text_input("Enter Number of Pregnancies (Enter 0 if candidate is Male) ")
    Glucose = slt.text_input("Enter Glucose Level")
    BloodPressure = slt.text_input("Enter Blood Pressure Value")
    SkinThickness = slt.text_input("Enter Skin Thickness Value")
    Insulin = slt.text_input("Enter Insulin Level")
    BMIs = slt.text_input("Enter BMI(Body Mass Index)")
    DiabetesPedigreeFunction = slt.text_input("Enter DiabetesPedigreeFunction")
    Age = slt.text_input("Enter Age")

    #Creating the array with the inputs & Creating Variable to store the output
    #inp_data = [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]
    output = ""

    #Creating a 'GO' Button
    if slt.button("Prediction"):
        output = predicting_diabetes([Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMIs, DiabetesPedigreeFunction, Age])
    
    slt.success(output)


if __name__ == '__main__':
    main()
