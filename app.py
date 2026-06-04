import streamlit as st
import pickle

# Load trained model
with open("student_pass_fail_model.pkl", "rb") as file:
    model = pickle.load(file)

# App Title
st.set_page_config(page_title="Student Pass/Fail Predictor", page_icon="🎓")

st.title("🎓 Student Pass/Fail Prediction System")
st.write("Predict whether a student will Pass or Fail based on Study Hours, Attendance, and Previous Score.")

# User Inputs
study_hours = st.number_input(
    "📚 Study Hours per Day",
    min_value=0.0,
    max_value=15.0,
    step=0.5
)

attendance = st.number_input(
    "📅 Attendance Percentage",
    min_value=0,
    max_value=100
)

previous_score = st.number_input(
    "📝 Previous Exam Score",
    min_value=0,
    max_value=100
)

# Prediction Button
if st.button("Predict Result"):

    input_data = [[study_hours, attendance, previous_score]]

    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.success("✅ Student is likely to PASS")
        st.balloons()
    else:
        st.error("❌ Student is likely to FAIL")