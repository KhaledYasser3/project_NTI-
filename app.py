import streamlit as st
import numpy as np
import joblib

# Load model and scaler
model = joblib.load("best_model.pkl")
scaler = joblib.load("scaler.pkl")

# Page configuration
st.set_page_config(
    page_title="📱 Phone Addiction Classifier",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom styling
st.markdown("""
    <style>
        .main {
            background-color: #f7f7f7;
        }
        h1 {
            color: #333333;
            font-size: 36px;
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            border-radius: 8px;
            padding: 0.5em 1em;
        }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown("<h1 style='text-align: center;'>📱 Phone Addiction Classifier</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Enter your daily phone usage details to check if you're addicted!</p>", unsafe_allow_html=True)
st.markdown("---")

# Input fields in two columns
col1, col2 = st.columns(2)

with col1:
    daily_usage = st.number_input("Daily Usage (hours)", min_value=0.0, max_value=24.0, value=4.0, step=0.1)
    apps_used = st.number_input("Apps Used Per Day", min_value=0, max_value=100, value=10)
    social_time = st.number_input("Time on Social Media (hours)", min_value=0.0, max_value=24.0, value=2.0, step=0.1)

with col2:
    phone_checks = st.number_input("Phone Checks Per Day", min_value=0, max_value=500, value=100)
    gaming_time = st.number_input("Time on Gaming (hours)", min_value=0.0, max_value=24.0, value=1.0, step=0.1)
    social_interactions = st.number_input("Social Interactions Per Day", min_value=0, max_value=100, value=5)

# Predict button
if st.button("Check Addiction Level"):
    user_input = np.array([[daily_usage, phone_checks, apps_used, social_time, gaming_time, social_interactions]])
    user_scaled = scaler.transform(user_input)
    prediction = model.predict(user_scaled)[0]
    probability = model.predict_proba(user_scaled)[0][1]

    st.markdown("---")
    st.subheader("🔍 Prediction Result:")

    if prediction == 1:
        st.error(f"High Risk of Phone Addiction ⚠️\n\n**Probability: {probability:.2f}**")
        st.image("https://sl.bing.net/gDGifAhOusK", width=250, caption="Phone Addiction Detected")
    else:
        st.success(f"Low Risk of Phone Addiction ✅\n\n**Probability: {probability:.2f}**")
        st.image("https://png.pngtree.com/thumb_back/fh260/background/20220208/pngtree-man-smiling-and-throwing-up-arms-healthy-male-cheerful-photo-image_29207588.jpg", width=250, caption="Healthy Phone Usage")



        
