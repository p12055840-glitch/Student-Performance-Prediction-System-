import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

# Page Configuration
st.set_page_config(page_title="Student Pass/Fail Predictor", page_icon="🎓")

# Header Image
st.image("student_banner.jpg", use_container_width=True)

st.title("🎓 Student Pass/Fail Prediction System")
st.write("Predict whether a student will pass or fail based on Study Hours, Attendance, and Previous Score.")

# Sample Dataset
data = {
    "Study_Hours": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "Attendance": [50, 55, 60, 65, 70, 75, 80, 85, 90, 95],
    "Previous_Score": [30, 35, 40, 45, 50, 55, 60, 70, 80, 90],
    "Result": [0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
}

df = pd.DataFrame(data)

# Features and Target
X = df[["Study_Hours", "Attendance", "Previous_Score"]]
y = df["Result"]

# Train Model
model = DecisionTreeClassifier()
model.fit(X, y)

# Inputs
study_hours = st.number_input(
    "📚 Study Hours per Day",
    min_value=0.0,
    max_value=15.0,
    step=0.5
)

attendance = st.number_input(
    "📅 Attendance (%)",
    min_value=0,
    max_value=100
)

previous_score = st.number_input(
    "📝 Previous Exam Score",
    min_value=0,
    max_value=100
)

# Prediction
if st.button("Predict Result"):

    input_data = [[study_hours, attendance, previous_score]]
    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.success("✅ Prediction: PASS")
        st.image("pass.jpg", width=300)
    else:
        st.error("❌ Prediction: FAIL")
        st.image("fail.jpg", width=300)

# Display Dataset
st.subheader("Sample Training Dataset")
st.dataframe(df)