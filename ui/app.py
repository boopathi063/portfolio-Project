# app.py
import streamlit as st
import requests
import os

API_URL = os.environ.get("API_URL", "http://localhost:8000/predict")

st.title("AI Food & Health Advisor")
st.write("Upload a food image and enter your age, height (m) and weight (kg).")

uploaded_file = st.file_uploader("Choose an image", type=['jpg','jpeg','png'])
age = st.number_input("Age", min_value=1, max_value=120, value=25)
height = st.number_input("Height (meters)", min_value=0.5, max_value=2.5, value=1.7, format="%.2f")
weight = st.number_input("Weight (kg)", min_value=10.0, max_value=300.0, value=70.0, format="%.1f")

if st.button("Analyze"):
    if uploaded_file is None:
        st.error("Please upload an image first.")
    else:
        files = {"file": ("image.jpg", uploaded_file.getvalue(), "image/jpeg")}
        data = {"age": str(int(age)), "height_m": str(float(height)), "weight_kg": str(float(weight))}
        with st.spinner("Analyzing..."):
            try:
                r = requests.post(API_URL, files=files, data=data, timeout=60)
                r.raise_for_status()
                res = r.json()
                st.subheader("Results")
                st.write(f"**Predicted food:** {res.get('predicted_class')}")
                st.write(f"**Calories (est.):** {res.get('calories')}")
                st.write(f"**BMI:** {res.get('bmi')} ({res.get('bmi_category')})")
                st.write(f"**Advice:** {res.get('verdict')}")
                st.write(res.get('explanation'))
            except Exception as e:
                st.error(f"Error: {e}")
