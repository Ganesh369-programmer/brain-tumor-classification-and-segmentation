"""Streamlit app for Brain Tumor Classification (Phase 1: classification only).

Run:  streamlit run app.py
Requires: models/brain_tumor_cnn.keras and models/class_names.json
(created by brain_tumor_classification.ipynb).

Educational project only. Not a medical diagnosis tool.
"""
import io
import json
import os

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from PIL import Image

MODEL_PATH = os.path.join("models", "brain_tumor_cnn.keras")
CLASS_NAMES_PATH = os.path.join("models", "class_names.json")

st.set_page_config(page_title="Brain Tumor Classification", page_icon="🧠", layout="centered")


@st.cache_resource
def load_model_and_meta():
    """Load the trained CNN and class metadata once."""
    if not os.path.isfile(MODEL_PATH) or not os.path.isfile(CLASS_NAMES_PATH):
        return None, None
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(CLASS_NAMES_PATH) as f:
        meta = json.load(f)
    return model, meta


def preprocess(image_bytes, img_size):
    """Same preprocessing as the notebook: RGB -> resize -> float -> /255."""
    image = tf.io.decode_image(image_bytes, channels=3, expand_animations=False)
    image = tf.image.resize(image, img_size)
    image = tf.cast(image, tf.float32) / 255.0
    return tf.expand_dims(image, axis=0)  # add batch dimension


def predict(model, meta, image_bytes):
    class_names = meta["class_names"]
    display = meta["display_names"]
    batch = preprocess(image_bytes, tuple(meta["img_size"]))
    probs = model.predict(batch, verbose=0)[0]

    idx = int(np.argmax(probs))
    pred_class = class_names[idx]
    detected = pred_class != "no_tumor"
    return {
        "detected": detected,
        "status": "Tumor Detected" if detected else "No Tumor Detected",
        "tumor_type": display[pred_class] if detected else None,
        "prediction": display[pred_class],
        "confidence": float(probs[idx]) * 100,
        "probabilities": {display[c]: float(p) * 100 for c, p in zip(class_names, probs)},
    }


# ----------------------------- UI -----------------------------
st.title("🧠 Brain Tumor Classification")
st.caption("Upload a brain MRI image to detect and classify a tumor (Glioma, Meningioma, Pituitary, or No Tumor).")
st.warning(
    "Educational project only. This is NOT a medical diagnosis and does not replace "
    "a qualified doctor or radiologist."
)

model, meta = load_model_and_meta()
if model is None:
    st.error(
        f"Model files not found. Run `brain_tumor_classification.ipynb` first so that "
        f"`{MODEL_PATH}` and `{CLASS_NAMES_PATH}` exist."
    )
    st.stop()

uploaded = st.file_uploader("Upload an MRI image", type=["jpg", "jpeg", "png", "bmp"])

if uploaded is not None:
    image_bytes = uploaded.read()
    try:
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        st.error("This file could not be read as an image. Please upload a valid MRI image.")
        st.stop()

    st.image(pil_image, caption="Uploaded MRI", use_container_width=True)

    if st.button("Classify", type="primary"):
        with st.spinner("Analyzing..."):
            result = predict(model, meta, image_bytes)

        st.subheader("Result")
        if result["detected"]:
            st.error(f"Status: {result['status']}")
            st.write(f"**Tumor Type:** {result['tumor_type']}")
        else:
            st.success(f"Status: {result['status']}")
            st.write(f"**Prediction:** {result['prediction']}")
        st.write(f"**Confidence:** {result['confidence']:.2f}%")

        st.subheader("Class Probabilities")
        prob_df = pd.DataFrame(
            {"Class": list(result["probabilities"].keys()),
             "Probability (%)": [round(v, 2) for v in result["probabilities"].values()]}
        )
        st.bar_chart(prob_df.set_index("Class"))
        st.table(prob_df.set_index("Class"))

st.divider()
st.caption("Phase 1: classification only. Segmentation will be added in a later phase.")
