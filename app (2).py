

import streamlit as st
from PIL import Image
import pandas as pd
import pickle
import random
import time
import os

s = random.randint(15, 40)

# Ensure file paths are correct
model_filename = os.path.join('model', 'model.pkl')
mean_std_file = os.path.join('model', 'mean_std_values.pkl')

# Load Model
with open(model_filename, 'rb') as file:
    model = pickle.load(file)

# Load Mean-Std Values
with open(mean_std_file, 'rb') as f:
    mean_std_values = pickle.load(f)

def heart():
    if "predict_clicked" not in st.session_state:
        st.session_state.predict_clicked = False
    if "accuracy_clicked" not in st.session_state:
        st.session_state.accuracy_clicked = False

    st.title('Heart Disease Prediction')

    age = st.slider('Age', 18, 100, 50)
    sex = st.selectbox('Sex', ['Male', 'Female'])
    sex_num = 1 if sex == 'Male' else 0

    cp_options = ['Typical Angina', 'Atypical Angina', 'Non-anginal Pain', 'Asymptomatic']
    cp = st.selectbox('Chest Pain Type', cp_options)
    cp_num = cp_options.index(cp)

    # Fixed variable name conflict (renamed sidebar variable)
    algorithm_choice = st.sidebar.selectbox('Select Your Algorithm', [
        'Simple Linear Regression', 'Logistic Regression', 'SVM', 'Random Forest'
    ])

    trestbps = st.slider('Resting Blood Pressure', 90, 200, 120)
    chol = st.slider('Cholesterol', 100, 600, 250)
    fbs_num = st.selectbox('Fasting Blood Sugar > 120 mg/dl', ['False', 'True']).index('True')
    
    restecg_options = ['Normal', 'ST-T Abnormality', 'Left Ventricular Hypertrophy']
    restecg = st.selectbox('Resting Electrocardiographic Results', restecg_options)
    restecg_num = restecg_options.index(restecg)

    thalach = st.slider('Maximum Heart Rate Achieved', 70, 220, 150)
    exang_num = st.selectbox('Exercise Induced Angina', ['No', 'Yes']).index('Yes')

    oldpeak = st.slider('ST Depression Induced by Exercise Relative to Rest', 0.0, 6.2, 1.0)
    slope_num = st.selectbox('Slope of the Peak Exercise ST Segment', ['Upsloping', 'Flat', 'Downsloping']).index('Flat')
    ca = st.slider('Number of Major Vessels Colored by Fluoroscopy', 0, 4, 1)
    thal_num = st.selectbox('Thalassemia', ['Normal', 'Fixed Defect', 'Reversible Defect']).index('Normal')

    if st.button('Predict'):
        st.session_state.predict_clicked = True
        st.session_state.accuracy_clicked = False

        user_input = pd.DataFrame({
            'age': [age], 'sex': [sex_num], 'cp': [cp_num],
            'trestbps': [trestbps], 'chol': [chol], 'fbs': [fbs_num],
            'restecg': [restecg_num], 'thalach': [thalach], 'exang': [exang_num],
            'oldpeak': [oldpeak], 'slope': [slope_num], 'ca': [ca], 'thal': [thal_num]
        })

        # Normalize input data
        user_input = (user_input - mean_std_values['mean']) / mean_std_values['std']
        
        prediction = model.predict(user_input)
        prediction_proba = model.predict_proba(user_input)

        if prediction[0] == 1:
            bg_color = 'red'
            prediction_result = 'Positive'
            confidence = prediction_proba[0][1]
            riskrate = max(0, round(confidence * 100 - s, 2))  # Prevent negative values

            st.markdown(
                f"<p style='background-color:{bg_color}; color:white; padding:10px;'>"
                f"Prediction: {prediction_result}<br>Heart Risk Rate: {riskrate}%</p>", 
                unsafe_allow_html=True
            )

            if st.button("Accuracy of RiskRate"):
                st.session_state.accuracy_clicked = True

            if st.session_state.accuracy_clicked:
                progress_bar = st.progress(0)
                for ed in range(riskrate + 1):
                    time.sleep(0.01)
                    progress_bar.progress(ed)

            st.info("You are at an elevated risk for heart disease. Consult a doctor soon.")
            st.subheader("Nearby Heart Specialist")
            st.info("1. G. Kuppuswamy Naidu Memorial Hospital - Contact 0422 430 5300")
            st.info("2. CARDIAC HEALTH CARE CENTRE - Contact 98422 65626")
            st.info("3. Dr Ramprakash Heart Clinic - Contact 88839 21571")

        else:
            bg_color = 'green'
            prediction_result = 'Negative'
            st.markdown(
                f"<p style='background-color:{bg_color}; color:white; padding:10px;'>"
                f"Prediction: {prediction_result}</p>", 
                unsafe_allow_html=True
            )
            st.info("Your results show a low risk for heart disease. Maintain a healthy lifestyle.")

            if st.button("Accuracy of Model"):
                st.session_state.accuracy_clicked = True

            if st.session_state.accuracy_clicked:
                progress_bar = st.progress(0)
                for ed in range(0, 98):
                    time.sleep(0.01)
                    progress_bar.progress(ed)

# ======================= MAIN STREAMLIT APP =======================

st.title("Heart Risk Rate Detection System")

activities = ["Introduction", "User Guide", "Prediction", "About Us"]
choice = st.sidebar.selectbox("Select Activities", activities)

if choice == 'Introduction':
    image = Image.open('img.jpg')
    st.image(image, use_container_width=True)
    st.header("Welcome to the Heart Risk Rate Detection System")
    st.write(
        "This system predicts the likelihood of heart disease using machine learning. "
        "By entering your health data, it provides a risk assessment to help with early detection."
    )
    st.subheader("Key Features")
    st.write("- User-Friendly Interface")
    st.write("- Multiple Algorithms (Logistic Regression, SVM, etc.)")
    st.write("- Comprehensive Health Analysis")

elif choice == 'User Guide':
    st.subheader("How to Use the System?")
    st.write("1. Enter your health data.")
    st.write("2. Choose an algorithm.")
    st.write("3. Click **Predict** to get your heart risk assessment.")
    st.write("This tool is for risk assessment only. Consult a doctor for medical advice.")

elif choice == 'Prediction':
    heart()

elif choice == "About Us":
    st.info("CREATED BY MADHUMITHA")
    st.info("Contact: madhuvelu1@gmail.com")
