import os
import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
from tensorflow.keras.utils import get_custom_objects
from tensorflow.keras.layers import Lambda, LayerNormalization, MultiHeadAttention, Input, Conv2D, Reshape, Embedding, Add, Dense, Flatten, Dropout

# Register custom layers
get_custom_objects().update({
    'Lambda': Lambda,
    'MultiHeadAttention': MultiHeadAttention,
    'LayerNormalization': LayerNormalization,
})

# Force TensorFlow to use CPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Define the mlp function
def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = Dense(units, activation=tf.nn.gelu)(x)
        x = Dropout(dropout_rate)(x)
    return x

# Load model function (with model architecture first)
def create_vit_classifier(
    input_shape=(32, 32, 3),
    patch_size=4,
    num_patches=64,
    projection_dim=64,
    num_heads=4,
    transformer_layers=8,
    mlp_head_units=[2048, 1024],
    num_classes=7,
):
    inputs = Input(shape=input_shape)
    
    # Patch embedding
    patches = Conv2D(projection_dim, kernel_size=patch_size, strides=patch_size, padding="VALID")(inputs)
    patches = Reshape((num_patches, projection_dim))(patches)
    
    # Position embedding
    positions = tf.range(start=0, limit=num_patches, delta=1)
    position_embedding = Embedding(input_dim=num_patches, output_dim=projection_dim)(positions)
    encoded_patches = patches + position_embedding
    
    # Transformer blocks
    for _ in range(transformer_layers):
        x1 = LayerNormalization(epsilon=1e-6)(encoded_patches)
        attention_output = MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, dropout=0.1
        )(x1, x1)
        x2 = Add()([attention_output, encoded_patches])
        x3 = LayerNormalization(epsilon=1e-6)(x2)
        x3 = mlp(x3, hidden_units=[projection_dim * 2, projection_dim], dropout_rate=0.1)
        encoded_patches = Add()([x3, x2])
    
    # MLP head
    representation = LayerNormalization(epsilon=1e-6)(encoded_patches)
    representation = Flatten()(representation)
    representation = Dropout(0.5)(representation)
    
    features = mlp(representation, hidden_units=mlp_head_units, dropout_rate=0.5)
    
    logits = Dense(num_classes)(features)
    outputs = tf.keras.layers.Activation("softmax")(logits)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

# Define class labels
CLASS_NAMES = ["melanocytic nevi", "melanoma", "benign keratosis-like lesions", "basal cell carcinoma", "pyogenic granulomas and hemorrhage", "Actinic keratoses and intraepithelial carcinomae", "dermatofibroma"]

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
                # Load model architecture and weights
                vit_classifier = create_vit_classifier()
                vit_classifier.load_weights('ArgusDerma_ViT.h5')  # Load the weights

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
