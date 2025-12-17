import numpy as np
import streamlit as st
import pickle

model=pickle.load(open('linear_regression_model.pkl','rb'))
st.title('Linear Regression Salary Prediction Model')
st.write("This app predicts the salary based on years of experience using a pre-trained Linear Regression model.")
years_experience = st.number_input('Enter Years of Experience:', min_value=0.0, max_value=50.0, step=0.1)
if st.button('Predict Salary'):
    predicted_salary = model.predict(np.array([[years_experience]]))
    st.write(f'Predicted Salary: ${predicted_salary[0]:.2f}')
    st.success('Prediction Successful!')
    st.write('Developed by Sravanthi')
    print("Model hasbeen Pickled and saved as linear_regression_model.pkl")
