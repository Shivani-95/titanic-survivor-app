import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load("titanic_model.pkl")

st.set_page_config(page_title="Titanic Survivor Predictor", layout="centered")
st.title("🚢 Titanic Survival Prediction App")
st.write("Enter passenger details to predict survival:")

# Input fields
pclass = st.selectbox("Passenger Class (1 = 1st, 2 = 2nd, 3 = 3rd)", [1, 2, 3])
sex = st.selectbox("Sex", ["male", "female"])
age = st.number_input("Age", min_value=1, max_value=100, value=25)
sibsp = st.number_input("No. of Siblings/Spouses Aboard", min_value=0, value=0)
parch = st.number_input("No. of Parents/Children Aboard", min_value=0, value=0)
fare = st.number_input("Fare Paid", min_value=0.0, value=50.0)

# Convert sex to numeric
sex_encoded = 1 if sex == "male" else 0

# Prepare input
features = pd.DataFrame([[pclass, sex_encoded, age, sibsp, parch, fare]],
                        columns=["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare"])

# Predict
if st.button("Predict Survival"):
    prediction = model.predict(features)[0]
    if prediction == 1:
        st.success("🎉 This passenger **would survive**!")
    else:
        st.error("💀 Unfortunately, this passenger **would not survive**.")
