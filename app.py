import streamlit as st
import pickle
import numpy as np

# Load trained model & scaler
try:
    with open("student_performance_svm.sav", "rb") as file:
        model = pickle.load(file)

    with open("scaler.sav", "rb") as file:
        scaler = pickle.load(file)
except FileNotFoundError:
    st.error(" Model or scaler file not found! Please check the files and restart the app.")
    st.stop()

# Apply simple styling
st.markdown("""
    <style>
        .stApp {
            text-align: center;
        }
        .main-container {
            max-width: 500px;
            margin: auto;
            padding: 2rem;
            border-radius: 10px;
            background: #f4f4f4;
            box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.1);
        }
        .stButton>button {
            width: 100%;
            background-color: #007BFF;
            color: white;
            border-radius: 5px;
            font-size: 16px;
            padding: 10px;
        }
    </style>
""", unsafe_allow_html=True)

# Page selection
st.sidebar.title(" Navigation")
page = st.sidebar.radio("Go to:", ["Overview", "Predictor", "Extra Info"])

if page == "Overview":
    st.title("Student Performance Predictor")
    st.markdown("""
    ## Problem Statement
    Many students struggle with academic performance due to various factors such as attendance, study habits, distractions, and anxiety levels. 
    Identifying at-risk students early can help provide necessary interventions to improve their outcomes.
    
    ## Solution
    This app uses a machine learning model to analyze key factors influencing student performance and predict whether a student is likely to pass or fail.
    By providing insights based on data, educators and students can take proactive steps to enhance learning strategies.
    """)

elif page == "Predictor":
    st.title(" Student Performance Predictor")
    with st.form("prediction_form"):
        st.subheader("Enter Student Details")
        attendance = st.slider("Attendance (%)", 0, 100, 90)
        study_hours = st.slider("Study Hours per Week", 0, 50, 20)
        previous_exam = st.slider("Previous Exam Score (%)", 0, 100, 75)
        sleep_hours = st.slider("Sleep Hours per Day", 0.0, 12.0, 7.0)
        distractions = st.slider("Number of Distractions", 0, 10, 2)
        extracurricular = st.radio("Extracurricular Activities", ["No", "Yes"], index=1)
        anxiety = st.slider("Test Anxiety Level (1-10)", 1, 10, 3)
        
        submit_button = st.form_submit_button("Predict")

    if submit_button:
        # Placeholder for missing features (assuming 6 missing features)
        missing_features = [0] * 6  
        
        input_data = np.array([[attendance, study_hours, previous_exam, sleep_hours, distractions, 
                                1 if extracurricular == "Yes" else 0, anxiety] + missing_features])

        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)
        result = " Pass" if prediction[0] == 1 else " Fail"
        st.markdown(f"### Prediction: {result}")

elif page == "Extra Info":
    st.title("ℹ Extra Information")
    st.markdown("""
    This application predicts student performance based on several factors like attendance, study habits, and anxiety levels.
    
    **Key Features Considered:**
    - Attendance percentage
    - Weekly study hours
    - Previous exam score
    - Sleep hours per day
    - Number of distractions
    - Extracurricular activities
    - Test anxiety level
    
    **How It Works:**
    The model uses machine learning techniques to analyze student data and determine whether they are likely to pass or fail.
    
    **Disclaimer:**
    This prediction is based on past data and should not be taken as a definitive result.
    """)



