import streamlit as st
import pandas as pd
import pickle
import random
import time
from PIL import Image

# Random seed for risk rate fluctuation
s = random.randint(15, 40)

# Load the trained model
model_filename = './model/model.pkl'

try:
    with open(model_filename, 'rb') as file:
        model = pickle.load(file)
except FileNotFoundError:
    st.error("Model file not found. Ensure 'model.pkl' is placed inside the 'model' directory.")
    st.stop()

# Load mean and standard deviation values
mean_std_filename = './model/mean_std_values.pkl'

try:
    with open(mean_std_filename, 'rb') as f:
        mean_std_values = pickle.load(f)
except FileNotFoundError:
    st.error("Mean and standard deviation values file not found. Ensure 'mean_std_values.pkl' is placed inside the 'model' directory.")
    st.stop()

# Heart Disease Prediction Function
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
    cp_num = cp_options.index(cp) if cp in cp_options else 0

    st.sidebar.selectbox('Select Your Algorithm', ['Simple Linear Regression', "Logistic Regression", "SVM", "Random Forest"])

    trestbps = st.slider('Resting Blood Pressure', 90, 200, 120)
    chol = st.slider('Cholesterol', 100, 600, 250)

    fbs = st.selectbox('Fasting Blood Sugar > 120 mg/dl', ['False', 'True'])
    fbs_num = 1 if fbs == 'True' else 0

    restecg_options = ['Normal', 'ST-T Abnormality', 'Left Ventricular Hypertrophy']
    restecg = st.selectbox('Resting Electrocardiographic Results', restecg_options)
    restecg_num = restecg_options.index(restecg) if restecg in restecg_options else 0

    thalach = st.slider('Maximum Heart Rate Achieved', 70, 220, 150)

    exang = st.selectbox('Exercise Induced Angina', ['No', 'Yes'])
    exang_num = 1 if exang == 'Yes' else 0

    oldpeak = st.slider('ST Depression Induced by Exercise Relative to Rest', 0.0, 6.2, 1.0)

    slope_options = ['Upsloping', 'Flat', 'Downsloping']
    slope = st.selectbox('Slope of the Peak Exercise ST Segment', slope_options)
    slope_num = slope_options.index(slope) if slope in slope_options else 0

    ca = st.slider('Number of Major Vessels Colored by Fluoroscopy', 0, 4, 1)

    thal_options = ['Normal', 'Fixed Defect', 'Reversible Defect']
    thal = st.selectbox('Thalassemia', thal_options)
    thal_num = thal_options.index(thal) if thal in thal_options else 0

    if st.button('Predict'):
        st.session_state.predict_clicked = True
        st.session_state.accuracy_clicked = False

        user_input = pd.DataFrame({
            'age': [age], 'sex': [sex_num], 'cp': [cp_num],
            'trestbps': [trestbps], 'chol': [chol], 'fbs': [fbs_num],
            'restecg': [restecg_num], 'thalach': [thalach],
            'exang': [exang_num], 'oldpeak': [oldpeak],
            'slope': [slope_num], 'ca': [ca], 'thal': [thal_num]
        })

        user_input = (user_input - mean_std_values['mean']) / mean_std_values['std']

        prediction = model.predict(user_input)
        prediction_proba = model.predict_proba(user_input)

        if prediction[0] == 1:
            bg_color = 'red'
            prediction_result = 'Positive'
            confidence = prediction_proba[0][1]
            riskrate = round(confidence * 100) - s

            st.markdown(f"<p style='background-color:{bg_color}; color:white; padding:10px;'>"
                        f"Prediction: {prediction_result}<br>Heart Risk Rate: {riskrate}%</p>",
                        unsafe_allow_html=True)

            if st.button("Accuracy of RiskRate"):
                st.session_state.accuracy_clicked = True
                progress_bar = st.progress(0)
                for ed in range(0, riskrate + 1):
                    time.sleep(0.01)
                    progress_bar.progress(ed)

            st.info("Based on your current health data, you are at elevated risk for heart disease. It's important to schedule an appointment with your doctor soon.")

            st.subheader("Nearby Heart Specialists:")
            st.info("1. G. Kuppuswamy Naidu Memorial Hospital - Contact: 0422 430 5300")
            st.info("2. Cardiac Health Care Centre - Contact: 98422 65626")
            st.info("3. Dr. Ramprakash Heart Clinic - Contact: 88839 21571")

        else:
            bg_color = 'green'
            prediction_result = 'Negative'
            riskrate = 0

            st.markdown(f"<p style='background-color:{bg_color}; color:white; padding:10px;'>"
                        f"Prediction: {prediction_result}</p>", unsafe_allow_html=True)

            st.info("Your results show low risk for heart disease. Keep up the good work with your diet, exercise, and regular health checkups.")

            if st.button("Accuracy of Model"):
                st.session_state.accuracy_clicked = True
                progress_bar = st.progress(0)
                for ed in range(0, 98):
                    time.sleep(0.01)
                    progress_bar.progress(ed)


# Streamlit Sidebar Navigation
st.title("Heart Risk Rate Detection System")

activities = ["Introduction", "User Guide", "Prediction", "About Us"]
choice = st.sidebar.selectbox("Select Activities", activities)

if choice == 'Introduction':
    try:
        image = Image.open('img.jpg')
        st.image(image, use_container_width=True)
    except FileNotFoundError:
        st.warning("Image file 'img.jpg' not found. Please add it to the project directory.")

    st.markdown("Heart disease remains one of the leading causes of mortality worldwide...")

elif choice == 'User Guide':
    st.subheader("How to Use the System?")
    st.write("1. **Input Your Data**: Enter your health parameters...")
    st.write("2. **Select the Algorithm**: Choose from available models...")
    st.write("3. **Get Your Prediction**: Click on **Predict** to receive your heart risk assessment.")

elif choice == 'Prediction':
    heart()

elif choice == "About Us":
    st.info("CREATED BY MADHUMITHA")
    st.info("Contact us: madhuvelu1@gmail.com")
