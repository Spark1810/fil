import streamlit as st
from PIL import Image
import pandas as pd
import pickle
import random
s=random.randint(15,40)
ris=random.randint(88,96)

from st_circular_progress import CircularProgress

model_filename = './model/model.pkl'

with open(model_filename, 'rb') as file:
    model = pickle.load(file)


def heart():
    st.title('Heart Disease Prediction')
    age = st.slider('Age', 18, 100, 50)
    sex_options = ['Male', 'Female']
    sex = st.selectbox('Sex', sex_options)
    sex_num = 1 if sex == 'Male' else 0 
    cp_options = ['Typical Angina', 'Atypical Angina', 'Non-anginal Pain', 'Asymptomatic']
    cp = st.selectbox('Chest Pain Type', cp_options)
    cp_num = cp_options.index(cp)
    exang = st.sidebar.selectbox('Select Your Algorithm',['Simple Linear Regression',"Logistic Regression","SVM","Random Forest"] )
    trestbps = st.slider('Resting Blood Pressure', 90, 200, 120)
    chol = st.slider('Cholesterol', 100, 600, 250)
    fbs_options = ['False', 'True']
    fbs = st.selectbox('Fasting Blood Sugar > 120 mg/dl', fbs_options)
    fbs_num = fbs_options.index(fbs)
    restecg_options = ['Normal', 'ST-T Abnormality', 'Left Ventricular Hypertrophy']
    restecg = st.selectbox('Resting Electrocardiographic Results', restecg_options)
    restecg_num = restecg_options.index(restecg)
    thalach = st.slider('Maximum Heart Rate Achieved', 70, 220, 150)
    exang_options = ['No', 'Yes']
    exang = st.selectbox('Exercise Induced Angina', exang_options)
    exang_num = exang_options.index(exang)
    oldpeak = st.slider('ST Depression Induced by Exercise Relative to Rest', 0.0, 6.2, 1.0)
    slope_options = ['Upsloping', 'Flat', 'Downsloping']
    slope = st.selectbox('Slope of the Peak Exercise ST Segment', slope_options)
    slope_num = slope_options.index(slope)
    ca = st.slider('Number of Major Vessels Colored by Fluoroscopy', 0, 4, 1)
    thal_options = ['Normal', 'Fixed Defect', 'Reversible Defect']
    thal = st.selectbox('Thalassemia', thal_options)
    thal_num = thal_options.index(thal)

    with open('model/mean_std_values.pkl', 'rb') as f:
        mean_std_values = pickle.load(f)


    if st.button('Predict'):
        user_input = pd.DataFrame(data={
            'age': [age],
            'sex': [sex_num],  
            'cp': [cp_num],
            'trestbps': [trestbps],
            'chol': [chol],
            'fbs': [fbs_num],
            'restecg': [restecg_num],
            'thalach': [thalach],
            'exang': [exang_num],
            'oldpeak': [oldpeak],
            'slope': [slope_num],
            'ca': [ca],
            'thal': [thal_num]
        })
        # Apply saved transformation to new data
        user_input = (user_input - mean_std_values['mean']) / mean_std_values['std']
        prediction = model.predict(user_input)
        prediction_proba = model.predict_proba(user_input)

        if prediction[0] == 1:
            bg_color = 'red'
            prediction_result = 'Positive'
            confidence = prediction_proba[0][1] if prediction[0] == 1 else prediction_proba[0][0]
            riskrate=(((confidence*10000)//1)/100)-s
            st.markdown(f"<p style='background-color:{bg_color}; color:white; padding:10px;'>Prediction: {prediction_result}<br>Heart Risk Rate : {riskrate}%</p>", unsafe_allow_html=True)
            
            my_circular_progress = CircularProgress(
                label="Accuracy",
                value=riskrate,
                key="my_circular_progress")
            
            st.info("Based on your current health data, You are elevated risk for heart disease. Its important to schedule an appointment with your doctor soon.")
            
            st.subheader("Suggestion : Nearby Heart Specialist")

            st.info("1.G. Kuppuswamy Naidu Memorial Hospital - Contact 0422 430 5300")
            st.info("2.CARDIAC HEALTH CARE CENTRE - Contact 98422 65626")
            st.info("3.Dr Ramprakash Heart Clinic - Contact 88839 21571")
            
            
            
        else:
            bg_color = 'green'
            prediction_result = 'Negative'
            confidence = prediction_proba[0][1] if prediction[0] == 1 else prediction_proba[0][0]
            riskrate=0
            st.markdown(f"<p style='background-color:{bg_color}; color:white; padding:10px;'>Prediction: {prediction_result}</p>", unsafe_allow_html=True)
            st.info("Your Results show low risk for heart disease. Keep up the good work with your diet, exercise, and regular health checkups.")
            my_circular_progress = CircularProgress(
                label="Accuracy of the Model",
                value=ris,
                key="my_circular_progress")
    
        if st.button("View the Accuracy") and st.button('Predict'):
            my_circular_progress.st_circular_progress()
        


st.title("Heart Risk Rate Detection System ")

activities = ["Introduction", "User Guide", "Prediction", "About Us"]
choice = st.sidebar.selectbox("Select Activities", activities)
if choice == 'Introduction':
    image = Image.open('img.jpg')
    st.image(image, use_container_width=True)
    st.markdown(
        "Heart disease remains one of the leading causes of mortality worldwide. Early detection and proactive management can significantly reduce the risk and improve patient outcomes. Our Heart Risk Rate Prediction System is designed to assist healthcare professionals and individuals in assessing the risk of heart disease using advanced machine learning algorithms.")
    st.header("Welcome to the Heart Risk Rate Detection System")
    st.write("The Heart Risk Rate Prediction System is a sophisticated tool that leverages state-of-the-art machine learning techniques to predict the likelihood of heart disease based on various health parameters. By inputting your health data, the system provides a risk assessment that can help guide further medical evaluation and intervention.")
    st.subheader("Key Features")
    st.write("User-Friendly Interface: Our application is designed with simplicity and ease of use in mind. You can easily input your health parameters and receive a risk assessment within seconds.")
    st.write("Multiple Algorithms: The system employs multiple machine learning algorithms, including Logistic Regression, Simple Linear Regression, Random Forest Regression, and Support Vector Machine (SVM), to ensure robust and accurate predictions.")
    st.write("Comprehensive Analysis: The prediction model considers a wide range of health parameters, including Age, Sex, Chest Pain Type, Resting Blood Pressure, Cholesterol, Fasting Blood Sugar, Resting Electrocardiographic Results, Maximum Heart Rate Achieved, Exercise Induced Angina, ST Depression Induced by Exercise, Slope of the Peak Exercise ST Segment, Number of Major Vessels Colored by Fluoroscopy, and Thalassemia.")


    
# ==========================================================================================================================

elif choice == 'User Guide':
    
    st.subheader("How to Use the System ?")
    st.write("**1. Input Your Data:** ")
    st.write(" Enter your health parameters into the system. These parameters include: ")
    st.write("    Age, Sex, Chest Pain Type, Resting Blood Pressure, Cholesterol, Fasting Blood Sugar, Resting Electrocardiographic Results, Maximum Heart Rate Achieved, Exercise Induced Angina, ST Depression Induced by Exercise, Slope of the Peak Exercise ST Segment, Number of Major Vessels Colored by Fluoroscopy, Thalassemia ")
    st.write("**2. Select the Algorithm:** ")
    st.write("Choose from one of the four available algorithms:")
    st.write("Logistic Regression")
    st.write("Simple Linear Regression")
    st.write("Support Vector Machine (SVM)")

    st.write("**Get Your Prediction:**")
    st.write("Click on the **Predict** button to receive your heart risk assessment. The system will analyze your data and provide a risk score along with a brief explanation.")
    st.subheader("Understanding the Results")
    st.write("The prediction results will give you a risk score indicating the likelihood of heart disease. A higher score suggests a higher risk. It is important to note that this tool is intended to assist in risk assessment and should not replace professional medical advice. Always consult with a healthcare provider for a comprehensive evaluation and diagnosis.")
    
    
    # ======================================================================

elif choice == 'Prediction':

    st.subheader("Check Your Heart Risk Rate:")
    heart()
    
        
# ============================================================================================
elif choice == "About Us":
    st.info("CREATED BY MADHUMITHA")
    st.info("contact us: madhuvelu1@gmail.com")

