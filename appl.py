import streamlit as st
import joblib
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model # type: ignore
import tensorflow as tf
import keras
#############################
@keras.saving.register_keras_serializable()
def r2_keras(y_true, y_pred):
    ss_res = tf.reduce_sum(tf.square(y_true - y_pred))
    ss_tot = tf.reduce_sum(tf.square(y_true - tf.reduce_mean(y_true)))
    return 1 - ss_res / (ss_tot + tf.keras.backend.epsilon())
# Load model
model = load_model("ann.keras")

# Load dropdown values (for UI only)
states = joblib.load("state_names.joblib")   # list of state names
cities = joblib.load("city_names.joblib")    # list of city names

# Load encoders and scaler
city_encoder = joblib.load("city_label_ecoding.joblib")   # should be LabelEncoder
state_encoder = joblib.load("state_label_ecoding.joblib") # should be LabelEncoder
scaler = joblib.load("standard_scaling.joblib")      # should be StandardScaler

# UI
st.title("🏠 Real Estate Price Prediction")
st.header("📋 Enter Property Details:")

bed = st.number_input("Number of Bedrooms", min_value=0, step=1)
bath = st.number_input("Number of Bathrooms", min_value=0.0, step=0.5)
acre_lot = st.number_input("Acre Lot Size", min_value=0.0, format="%.2f")
house_size = st.number_input("House Size (in sq ft)", min_value=0)

street = st.text_input("Street")  # optional, not used in prediction
city = st.selectbox("City", cities)
state = st.selectbox("State", states)

if st.button("Predict"):
    try:
        # Encode categorical features
        encoded_city = city_encoder.transform([city])[0]
        encoded_state = state_encoder.transform([state])[0]

        # Combine all features
        raw_input = np.array([bed,bath,acre_lot,house_size,street,encoded_city,encoded_state])
        # Apply standard scaling to numeric columns (assuming these 4 were scaled during training)
        
        scaled_data = scaler.transform([raw_input])
        

        # Predict
        prediction = model.predict(scaled_data)[0][0]  # get float from array
        st.success(f"🏡 Predicted Price: ${prediction:,.2f}")
    except Exception as e:
        st.error(f"❌ Error during prediction: {e}")
