import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import pickle

# Load dataset
df = pd.read_excel("student_performance_dataset.xlsx")

# Convert Pass/Fail to numeric
df["Result"] = df["Result"].map({"Fail": 0, "Pass": 1})

# Features and Target
X = df[["Study_Hours", "Attendance", "Previous_Score"]]
y = df["Result"]

# Train Model
model = DecisionTreeClassifier()
model.fit(X, y)

# Save model
with open("student_pass_fail_model.pkl", "wb") as file:
    pickle.dump(model, file)

print("Model saved successfully!") 
