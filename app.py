# app.py - Streamlit App for AgriDoctorAI (Render-ready)

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# Define model path
model_path = "model/plant_disease_model.keras"

# Ensure model directory exists
if not os.path.exists("model"):
    st.error(
        "Model directory not found. Please upload your trained model under /model."
    )
    st.stop()

# Ensure model file exists
if not os.path.exists(model_path):
    st.error("Model file not found at 'model/plant_disease_model.keras'.")
    st.stop()

# Load the model
try:
    model = tf.keras.models.load_model(model_path)
except Exception as e:
    st.error(f"Error loading the model: {e}")
    st.stop()

# Define class names (based on training)
class_names = [
    "Tomato_Bacterial_spot",
    "Tomato_Early_blight",
    "Tomato_Late_blight",
    "Tomato_Leaf_Mold",
    "Tomato_Septoria_leaf_spot",
    "Tomato_Spider_mites",
    "Tomato_Target_Spot",
    "Tomato_Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato_Tomato_mosaic_virus",
    "Tomato_healthy",
]


# Prediction function
def predict(image):
    image = image.resize((224, 224)).convert("RGB")
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)
    class_idx = np.argmax(prediction)
    confidence = float(prediction[0][class_idx])
    return class_names[class_idx], confidence


# Streamlit UI setup
st.set_page_config(page_title="AgriDoctorAI", layout="centered")

# Styling
st.markdown(
    """
    <style>
        .main { background-color: #f4f4f4; padding: 20px; border-radius: 10px; }
        .title { color: #4CAF50; font-size: 2.5em; margin-bottom: 10px; }
    </style>
""",
    unsafe_allow_html=True,
)

st.markdown('<div class="main">', unsafe_allow_html=True)
st.markdown('<div class="title">🌿 AgriDoctorAI</div>', unsafe_allow_html=True)
st.write("Upload a leaf image to detect potential plant disease")

# Upload form
uploaded_file = st.file_uploader("Choose a leaf image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file)
# app.py - Streamlit App for AgriDoctorAI (Render-ready)

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# Define model path
# model_path = "model/plant_disease_model.h5"
# model_path = "plant_disease_model.keras"

# # Ensure model directory exists
# if not os.path.exists("model"):
#     st.error(
#         "Model directory not found. Please upload your trained model under /model."
#     )
#     st.stop()

# # Ensure model file exists
# if not os.path.exists(model_path):
#     st.error("Model file not found at 'plant_disease_model.keras'.")
#     st.stop()

# Load the model
try:
    # model = tf.keras.models.load_model(model_path)
    model = tf.keras.models.load_model("plant_disease_model.keras")

except Exception as e:
    st.error(f"Error loading the model: {e}")
    st.stop()

# Define class names (based on training)
class_names = [
    "Potato___Late_blight",
    "Tomato_healthy",
    "Potato___healthy",
    "Tomato__Tomato_YellowLeaf__Curl_Virus",
    "Tomato_Leaf_Mold",
    "Tomato_Late_blight",
    "Potato___Early_blight",
    "Tomato_Early_blight",
    "Tomato__Tomato_mosaic_virus",
    "Tomato_Spider_mites_Two_spotted_spider_mite",
    "Pepper__bell___Bacterial_spot",
    "Tomato__Target_Spot",
    "Pepper__bell___healthy",
    "Tomato_Bacterial_spot",
    "Tomato_Septoria_leaf_spot",
]


# Prediction function
def predict(image):
    image = image.resize((224, 224)).convert("RGB")
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)
    class_idx = np.argmax(prediction)
    confidence = float(prediction[0][class_idx])
    return class_names[class_idx], confidence


# Streamlit UI setup
st.set_page_config(page_title="AgriDoctorAI", layout="centered")

# Styling
st.markdown(
    """
    <style>
        .main { background-color: #f4f4f4; padding: 20px; border-radius: 10px; }
        .title { color: #4CAF50; font-size: 2.5em; margin-bottom: 10px; }
    </style>
""",
    unsafe_allow_html=True,
)

st.markdown('<div class="main">', unsafe_allow_html=True)
st.markdown('<div class="title">🌿 AgriDoctorAI</div>', unsafe_allow_html=True)
st.write("Upload a leaf image to detect potential plant disease")

# Upload form
uploaded_file = st.file_uploader("Choose a leaf image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file)
        st.image(image, use_container_width=True)

        with st.spinner("🔍 Detecting disease, please wait..."):
            label, confidence = predict(image)
        st.success(f"Prediction: {label}")
        st.progress(confidence)
        st.info(f"Confidence: {confidence*100:.2f}%")
    except Exception as e:
        st.error(f"Error processing the image: {str(e)}")

st.markdown("</div>", unsafe_allow_html=True)
        with st.spinner("🔍 Detecting disease, please wait..."):
            label, confidence = predict(image)
        st.success(f"Prediction: {label}")
        st.progress(confidence)
        st.info(f"Confidence: {confidence*100:.2f}%")
    except Exception as e:
        st.error(f"Error processing the image: {str(e)}")

st.markdown("</div>", unsafe_allow_html=True)
