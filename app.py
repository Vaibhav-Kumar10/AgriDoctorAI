# app.py - Streamlit App for AgriDoctorAI with Styling, Spinner, and Confidence Progress Bar

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# Load model
model_path = 'model/plant_disease_model.h5'
if not os.path.exists(model_path):
    st.error("Model file not found. Please upload or train the model.")
    st.stop()

model = tf.keras.models.load_model(model_path)

# Define class names (update as per your model)
class_names = [
    'Tomato_Bacterial_spot',
    'Tomato_Early_blight',
    'Tomato_Late_blight',
    'Tomato_Leaf_Mold',
    'Tomato_Septoria_leaf_spot',
    'Tomato_Spider_mites',
    'Tomato_Target_Spot',
    'Tomato_Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato_Tomato_mosaic_virus',
    'Tomato_healthy'
]

# Prediction function
def predict(image):
    image = image.resize((224, 224)).convert('RGB')
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)
    class_idx = np.argmax(prediction)
    confidence = float(prediction[0][class_idx])
    return class_names[class_idx], confidence

# Streamlit UI with Bootstrap styling
st.set_page_config(page_title="AgriDoctorAI", layout="centered")
st.markdown("""
    <style>
        .main { background-color: #f4f4f4; padding: 20px; border-radius: 10px; }
        .title { color: #4CAF50; font-size: 2.5em; margin-bottom: 10px; }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main">', unsafe_allow_html=True)

st.markdown('<div class="title">🌿 AgriDoctorAI</div>', unsafe_allow_html=True)
st.write("Upload a leaf image to detect potential plant disease")

uploaded_file = st.file_uploader("Choose a leaf image", type=['jpg', 'png', 'jpeg'])

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        with st.spinner('🔍 Detecting disease, please wait...'):
            label, confidence = predict(image)
        st.success(f"Prediction: {label}")
        st.progress(confidence)
        st.info(f"Confidence: {confidence*100:.2f}%")
    except Exception as e:
        st.error(f"Error processing the image: {str(e)}")

st.markdown('</div>', unsafe_allow_html=True)
