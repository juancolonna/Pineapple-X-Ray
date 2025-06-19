import streamlit as st
import os
import numpy as np
from tensorflow.keras.models import load_model
from lime import lime_image
from PIL import Image
import io
import sys
import random
import time
import tensorflow as tf

# 1. Defina funções nomeadas para substituir lambdas
@tf.keras.utils.register_keras_serializable()
def reduce_mean_spatial(x):
    return tf.reduce_mean(x, axis=-1, keepdims=True)

@tf.keras.utils.register_keras_serializable()
def reduce_max_spatial(x):
    return tf.reduce_max(x, axis=-1, keepdims=True)

tf.keras.config.enable_unsafe_deserialization()

# --- Configuration and Constants ---
st.set_page_config(layout="wide")

IMG_HEIGHT = 224
IMG_WIDTH = int(1.1 * IMG_HEIGHT)
NUM_FEATURES_LIME = 5
NUM_SAMPLES_LIME = 1000
LIME_RANDOM_STATE = 42  # for reproducibility

MODEL_PATH = 'Models/best.keras'
# MODEL_PATH = 'Models/MULTILABEL_model_2025-06-14_20-44-02.keras'
# MODEL_PATH = 'Models/MULTILABEL_model_2025-06-14_22-09-33.keras'
# MODEL_PATH = 'Models/MULTILABEL_model_2025-06-15_14-42-39.keras' # oversampling

TRANS_THRESHOLD = 0.57
BROWN_THRESHOLD = 0.47

@st.cache_resource
def load_keras_model(path):
    if not os.path.exists(path):
        st.error(f"Model file not found at '{path}'. Please ensure the file exists.")
        return None
    try:
        model = load_model(path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

@st.cache_resource
def get_lime_explainer():
    return lime_image.LimeImageExplainer()

def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    img_resized = img.resize((IMG_WIDTH, IMG_HEIGHT))
    img_array = np.array(img_resized) / 255.0
    return img_array, np.expand_dims(img_array, axis=0)

def generate_lime_explanation(_explainer, model, image_np, label_to_explain):
    np.random.seed(LIME_RANDOM_STATE)
    random.seed(LIME_RANDOM_STATE)
    sys.stdout = open(os.devnull, 'w')
    explanation = _explainer.explain_instance(
        image=image_np,
        classifier_fn=model.predict,
        top_labels=2,
        hide_color=0,
        num_samples=NUM_SAMPLES_LIME,
    )
    sys.stdout = sys.__stdout__
    temp, _ = explanation.get_image_and_mask(
        label=label_to_explain,
        positive_only=False,
        num_features=NUM_FEATURES_LIME,
        hide_rest=False
    )
    return (temp * 255).astype(np.uint8)

def main():
    st.title("🍍 Pineapple X-Ray Defect Classifier")
    st.markdown("Upload an X-ray image of a pineapple to classify it for **translucency** and **browning** defects.")

    # === Example Download Link in Sidebar ===
    st.sidebar.markdown("---")
    st.sidebar.markdown("#### 📥 Examples of Images for Testing")
    GOOGLE_DRIVE_URL = "https://drive.google.com/file/d/13NCIhPE-vuo9ghfTDi7-ZE8ozDX8zAYk/view?usp=sharing"
    st.sidebar.markdown(f"[Click here to download a .zip file with test images]({GOOGLE_DRIVE_URL})")
    st.sidebar.markdown("---")

    model = load_keras_model(MODEL_PATH)
    if model is None:
        st.warning("Please add the keras model to the 'Models' folder and refresh.")
        return

    st.sidebar.header("⚙️ Controls")
    uploaded_file = st.sidebar.file_uploader(
        "Upload your image",
        type=["jpg", "jpeg", "png"]
    )
    run_lime = st.sidebar.checkbox("Generate LIME Explanations", value=False)
    st.sidebar.info(
        "LIME generates a visual explanation of the model's decision. "
        "Green areas increased the probability for the class, while red areas decreased it."
    )

    if uploaded_file is None:
        st.info("""
        👈 **Upload an image using the sidebar to get started.**

        - Your image must be in **JPG**, **JPEG**, or **PNG** format.
        - Any size is accepted, but it will be **resized to 224×246 pixels** for processing.
        - For best results, use a **black background** (avoid white or salt-and-pepper background noise).
        - Make sure the image is **sharp, centered**, and contains as **few edges and black areas as possible** for accurate predictions.
        """)
        return

    col1, col2 = st.columns([0.5, 0.5])
    image_bytes = uploaded_file.getvalue()
    image_array_norm, image_for_prediction = preprocess_image(image_bytes)

    with col1:
        st.subheader("Uploaded Image")
        img_name = uploaded_file.name if hasattr(uploaded_file, "name") else "Image"
        caption_text = f"{img_name} — Your uploaded pineapple X-ray."
        st.image(image_for_prediction, caption=caption_text, width=470)

    with col2:
        st.subheader("🤖 Model Predictions")
        with st.spinner("Analyzing image..."):
            preds = model.predict(image_for_prediction)
            time.sleep(2)  # <-- Add this line to force spinner visibility
            prob_browning = preds[0][1]
            prob_translucency = preds[0][0]
            mcol1, mcol2 = st.columns(2)
            with mcol1:
                st.metric(label="Translucency (Confidence)", 
                          value=f"{'yes' if prob_translucency > TRANS_THRESHOLD else 'no'} ({prob_translucency:.2%})")
            with mcol2:
                st.metric(label="Browning (Confidence)", 
                          value=f"{'yes' if prob_browning > BROWN_THRESHOLD else 'no'} ({prob_browning:.2%})")

        if run_lime:
            st.subheader("🔬 LIME Explanations")
            st.markdown("Red areas decrease the probability of each class, while green areas increase it.")
            lime_explainer = get_lime_explainer()
            with st.spinner("Generating LIME explanations..."):
                lime_translucency_img = generate_lime_explanation(
                    lime_explainer, model, image_array_norm, 0
                )
                lime_browning_img = generate_lime_explanation(
                    lime_explainer, model, image_array_norm, 1
                )
            e_col1, e_col2 = st.columns(2)
            with e_col1:
                st.image(
                    lime_translucency_img,
                    caption=f"Explanation for Translucency (Prob: {prob_translucency:.2f})",
                    use_container_width=False, width=300
                )
            with e_col2:
                st.image(
                    lime_browning_img,
                    caption=f"Explanation for Browning (Prob: {prob_browning:.2f})",
                    use_container_width=False, width=300
                )

if __name__ == "__main__":
    # run using: streamlit run streamlitApp.py
    main()
