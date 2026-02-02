import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D


# ================================
# App Configuration
# ================================
st.set_page_config(
    page_title="PSC Detection",
    layout="centered"
)

st.title("Posterior Subcapsular Cataract (PSC) Detection")
st.write("EfficientNetB0-based deep learning system for PSC classification")

# ================================
# Load Model (EfficientNetB0 ONLY)
# ================================
@st.cache_resource
def load_psc_model():
    base = EfficientNetB0(
        weights=None,
        include_top=False,
        input_shape=(224, 224, 3)
    )

    x = base.output
    x = GlobalAveragePooling2D()(x)
    output = Dense(1, activation="sigmoid")(x)

    model = Model(inputs=base.input, outputs=output)
    model.load_weights("psc_efficientnetb0.weights.h5")

    return model

model = load_psc_model()


# ================================
# Image Preprocessing
# ================================
IMG_SIZE = 224

def preprocess_image(image: Image.Image):
    image = image.convert("RGB")
    image = image.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(image)
    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# ================================
# Grad-CAM (EfficientNetB0)
# ================================
def make_gradcam_heatmap(img_array, model, last_conv_layer_name="top_conv"):
    grad_model = tf.keras.models.Model(
        model.inputs,
        [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, 0]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

# ================================
# Manual Patient Inputs
# ================================
st.subheader("Patient Details")
patient_id = st.text_input("Patient ID")
patient_name = st.text_input("Patient Name")
patient_age = st.number_input("Age", min_value=0, max_value=120)
patient_gender = st.selectbox("Gender", ["M", "F", "Other"])

# ================================
# File Upload
# ================================
uploaded_file = st.file_uploader("Upload a retinal image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display patient details
    st.subheader("Entered Patient Details")
    st.write({
        "Patient ID": patient_id,
        "Name": patient_name,
        "Age": patient_age,
        "Gender": patient_gender
    })

    # Display uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess
    img_array = preprocess_image(image)

    # Prediction
    prediction = model.predict(img_array)[0][0]

    st.subheader("Prediction Result")
    if prediction >= 0.5:
        st.error(f"PSC Detected (Confidence: {prediction:.2f})")
    else:
        st.success(f"No PSC Detected (Confidence: {1 - prediction:.2f})")

    # ================================
    # Grad-CAM Visualization
    # ================================
    st.subheader("Grad-CAM Visualization")
    heatmap = make_gradcam_heatmap(img_array, model)
    img_np = np.array(image.resize((IMG_SIZE, IMG_SIZE)))

    fig, ax = plt.subplots()
    ax.imshow(img_np)
    ax.imshow(heatmap, cmap="jet", alpha=0.4)
    ax.axis("off")
    st.pyplot(fig)

# ================================
# Disclaimer
# ================================
st.markdown(
    """
    ---
    **Disclaimer:**  
    This system is intended for academic and research purposes only.  
    It is not a substitute for professional medical diagnosis.
    """
)

#streamlit run psc_app.py