import os
import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

# Force TensorFlow to use CPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Define class labels
CLASS_NAMES = ["melanocytic nevi", "melanoma", "benign keratosis-like lesions", 
               "basal cell carcinoma", "pyogenic granulomas and hemorrhage", 
               "Actinic keratoses and intraepithelial carcinomae", "dermatofibroma"]

# Cache the model loading to avoid reloading it multiple times
@st.cache_resource
def load_vit_model():
    model_path = "ArgusDerma_ViT.keras"
    if not os.path.exists(model_path):
        st.error("Model file not found! Ensure 'ArgusDerma_ViT.keras' is in the same directory.")
        return None
    model = load_model(model_path)  # Load full model
    return model

# Image preprocessing function
def preprocess_image(img):
    img = img.convert("RGB")  # Ensure RGB format
    img = img.resize((32, 32))  # Resize image to match model input size (32x32)
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img.astype(np.float32) / 255.0  # Normalize
    return img

# Streamlit UI
st.title("🩺 Skin Cancer Detection")
st.write("Upload an image to analyze.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    try:
        # Display the uploaded image
        image_data = Image.open(uploaded_file)
        st.image(image_data, caption="Uploaded Image", use_column_width=True)

        if st.button("Analyze Image"):
            with st.spinner("Analyzing..."):
                # Load model
                vit_classifier = load_vit_model()
                if vit_classifier is None:
                    st.error("Model could not be loaded. Check the file path.")
                else:
                    # Preprocess the image
                    img = preprocess_image(image_data)
                    
                    # Make predictions
                    predictions = vit_classifier.predict(img)[0]

                    # Display the results
                    st.subheader("Prediction Results:")
                    for class_name, confidence in zip(CLASS_NAMES, predictions):
                        st.write(f"**{class_name}:** {confidence * 100:.2f}%")

                    # Display the most confident diagnosis
                    max_index = np.argmax(predictions)
                    st.success(f"**Diagnosis:** {CLASS_NAMES[max_index]} ({predictions[max_index] * 100:.2f}%)")

    except Exception as e:
        st.error(f"Error processing image: {e}")
