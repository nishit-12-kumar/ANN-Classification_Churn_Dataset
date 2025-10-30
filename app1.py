import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
import pickle
import streamlit as st
import os

# ------------------ Load the trained model ------------------
if os.path.exists("model.keras"):
    model = tf.keras.models.load_model("model.keras")
elif os.path.exists("model.h5"):
    model = tf.keras.models.load_model("model.h5", compile=False)
else:
    st.error("Model file not found! Please place model.h5 or model.keras in the same folder.")
    st.stop()

# ------------------ Load encoders and scaler ------------------
with open('ohe_geography.pkl', 'rb') as file:
    ohe_geography = pickle.load(file)

with open('label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# ------------------ Streamlit App ------------------
st.title("💳 Customer Churn Prediction")

# User Inputs
geography = st.selectbox('🌍 Geography', ohe_geography.categories_[0])
gender = st.selectbox('👩‍🦰 Gender', label_encoder_gender.classes_)
age = st.slider('🎂 Age', 18, 92, 30)
balance = st.number_input('🏦 Balance', min_value=0.0)
credit_score = st.number_input('💳 Credit Score', min_value=0)
estimated_salary = st.number_input('💰 Estimated Salary', min_value=0.0)
tenure = st.slider('⌛ Tenure', 0, 10, 5)
num_of_products = st.slider('🛍️ Number of Products', 1, 4, 1)
has_cr_card = st.selectbox('💳 Has Credit Card', [0, 1])
is_active_member = st.selectbox('✅ Is Active Member', [0, 1])

# ------------------ Encoding Geography safely ------------------
try:
    geo_encoded = ohe_geography.transform([[geography]])
except ValueError:
    st.error(f"Invalid geography '{geography}'. Valid options: {list(ohe_geography.categories_[0])}")
    st.stop()

geo_encoded_df = pd.DataFrame(
    geo_encoded,
    columns=ohe_geography.get_feature_names_out(['Geography'])
)

# ------------------ Prepare Input Data ------------------
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

# Combine encoded columns
input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

# ------------------ Scaling ------------------
input_data_scaled = scaler.transform(input_data)

# ------------------ Prediction ------------------
prediction = model.predict(input_data_scaled)
prediction_proba = prediction[0][0]

# ------------------ Display Result ------------------
st.subheader("📊 Prediction Result")
st.write(f"**Churn Probability:** {(prediction_proba*100):.2f}")

if prediction_proba > 0.5:
    st.error('⚠️ The customer is likely to churn.')
else:
    st.success('✅ The customer is not likely to churn.')
