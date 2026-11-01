"""Streamlit app for educational brain MRI classification + segmentation.

Run:  streamlit run app.py

Requires:
    models/brain_tumor_cnn.keras      (classifier, from brain_tumor_classification.ipynb)
    models/class_names.json           (classifier metadata)
Optional:
    models/brain_tumor_unet.keras     (segmentation, from train_segmentation.py)
    If it is missing, the app still works as a classifier only.

This is not a medical diagnostic tool.
"""
import io
import json
import os

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps

MODEL_PATH = os.path.join("models", "brain_tumor_cnn.keras")
CLASS_NAMES_PATH = os.path.join("models", "class_names.json")
SEG_MODEL_PATH = os.path.join("models", "brain_tumor_unet.keras")

st.set_page_config(page_title="Brain Tumor Classification & Segmentation", page_icon="🧠", layout="centered")


# ------------------------- Loading -------------------------
@st.cache_resource
def load_model_and_meta():
    if not os.path.isfile(MODEL_PATH) or not os.path.isfile(CLASS_NAMES_PATH):
        return None, None
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(CLASS_NAMES_PATH, encoding="utf-8") as f:
        meta = json.load(f)
    return model, meta


@st.cache_resource
def load_segmenter():
    """Load the U-Net once. Returns None if it has not been trained yet."""
    if not os.path.isfile(SEG_MODEL_PATH):
        return None
    return tf.keras.models.load_model(SEG_MODEL_PATH, compile=False)


# ------------------------- Classification (modified version) -------------------------
def preprocess(image_bytes, img_size):
    """Match notebook preprocessing: RGB, resize, float32 in [0, 255].

    EfficientNetB0 contains its own input rescaling layer; do not divide by 255.
    """
    with Image.open(io.BytesIO(image_bytes)) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")
        image = np.asarray(im)
    image = tf.convert_to_tensor(image, dtype=tf.float32)
    image = tf.image.resize(image, img_size)
    return tf.expand_dims(image, axis=0)


def predict(model, meta, image_bytes):
    class_names = meta["class_names"]
    display = meta["display_names"]
    batch = preprocess(image_bytes, tuple(meta["img_size"]))
    probs = model.predict(batch, verbose=0)[0]
    order = np.argsort(probs)
    idx = int(order[-1])
    pred_class = class_names[idx]
    confidence = float(probs[idx])
    margin = confidence - float(probs[order[-2]])
    min_confidence = float(meta.get("minimum_confidence", 0.65))
    min_margin = float(meta.get("minimum_top_two_margin", 0.15))
    uncertain = confidence < min_confidence or margin < min_margin

    if uncertain:
        status = "Uncertain — no reliable class assigned"
        detected = None
        prediction = "Uncertain"
        tumor_type = None
    elif pred_class == "no_tumor":
        status = "No Tumor Class Predicted"
        detected = False
        prediction = display[pred_class]
        tumor_type = None
    else:
        status = "Tumor Class Predicted"
        detected = True
        prediction = display[pred_class]
        tumor_type = display[pred_class]

    return {
        "uncertain": uncertain,
        "detected": detected,
        "status": status,
        "tumor_type": tumor_type,
        "prediction": prediction,
        "confidence": confidence * 100,
        "margin": margin * 100,
        "probabilities": {display[c]: float(p) * 100 for c, p in zip(class_names, probs)},
    }


# ------------------------- Segmentation (unchanged logic) -------------------------
def preprocess_seg(image_bytes, img_size):
    """U-Net preprocessing: RGB -> resize -> /255 (this is how the U-Net was trained).

    Kept separate from the classifier's preprocess(), because the classifier now
    expects values in [0, 255] while the U-Net expects values in [0, 1].
    """
    image = tf.io.decode_image(image_bytes, channels=3, expand_animations=False)
    image = tf.image.resize(image, img_size)
    image = tf.cast(image, tf.float32) / 255.0
    return tf.expand_dims(image, axis=0)


def segment(seg_model, image_bytes, threshold):
    """Run the U-Net. Returns (image_uint8, binary_mask) both 224x224."""
    size = tuple(seg_model.input_shape[1:3])            # e.g. (224, 224)
    batch = preprocess_seg(image_bytes, size)
    prob_map = seg_model.predict(batch, verbose=0)[0, :, :, 0]
    mask = (prob_map > threshold).astype(np.uint8)
    image = (batch[0].numpy() * 255).astype(np.uint8)
    return image, mask


def make_overlay(image, mask, alpha=0.4):
    """Tint the tumor area red and draw a yellow outline around it."""
    overlay = image.copy()
    red = np.zeros_like(image)
    red[..., 0] = 255
    m = mask.astype(bool)
    overlay[m] = ((1 - alpha) * image[m] + alpha * red[m]).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (255, 255, 0), 1)
    return overlay


# ----------------------------- UI -----------------------------
st.title("🧠 Brain Tumor Classification & Segmentation")
st.caption(
    "Educational classification of brain MRI images into glioma, meningioma, pituitary, or no-tumor "
    "classes. If a tumor class is predicted, the app also marks where the tumor is."
)
st.warning(
    "Academic project only. This model is not a medical diagnosis and does not replace "
    "a qualified doctor or radiologist. It may misclassify unfamiliar scans and non-MRI images."
)

model, meta = load_model_and_meta()
if model is None:
    st.error(
        "Model files were not found. Install requirements, run all cells in "
        "brain_tumor_classification.ipynb to train and save the model, then restart this app."
    )
    st.stop()

seg_model = load_segmenter()

with st.sidebar:
    st.header("Segmentation settings")
    if seg_model is None:
        st.info("Segmentation model not found. Run `train_segmentation.py` to enable it.")
    threshold = st.slider("Mask threshold", 0.10, 0.90, 0.50, 0.05,
                          help="Lower marks more area as tumor. Higher marks less.")
    force_seg = st.checkbox("Run segmentation even if the result is No Tumor or Uncertain", value=False)

uploaded = st.file_uploader("Upload a brain MRI image", type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"])
if uploaded is not None:
    image_bytes = uploaded.getvalue()
    try:
        pil_image = ImageOps.exif_transpose(Image.open(io.BytesIO(image_bytes))).convert("RGB")
    except Exception:
        st.error("This file could not be read as an image. Please upload a valid image file.")
        st.stop()
    st.image(pil_image, caption="Uploaded image", use_container_width=True)

    if st.button("Classify image", type="primary"):
        with st.spinner("Analyzing image..."):
            try:
                result = predict(model, meta, image_bytes)
            except Exception as exc:
                st.error(f"Could not classify this image: {exc}")
                st.stop()

            seg_out = None
            if seg_model is not None and (result["detected"] is True or force_seg):
                try:
                    seg_out = segment(seg_model, image_bytes, threshold)
                except Exception as exc:
                    st.error(f"Could not segment this image: {exc}")

        # ---- Classification result ----
        st.subheader("Result")
        if result["uncertain"]:
            st.warning(result["status"])
            st.write("Try a clear brain MRI image. This uncertainty check is not a guarantee that non-MRI images will be rejected.")
        elif result["detected"]:
            st.error(result["status"])
            st.write(f"**Predicted class:** {result['tumor_type']}")
        else:
            st.success(result["status"])
            st.write(f"**Predicted class:** {result['prediction']}")

        st.write(f"**Model score:** {result['confidence']:.2f}%")
        st.caption("The score is not a calibrated medical probability. Results require clinical interpretation.")
        st.subheader("Class scores")
        prob_df = pd.DataFrame({
            "Class": list(result["probabilities"].keys()),
            "Score (%)": [round(v, 2) for v in result["probabilities"].values()],
        })
        st.bar_chart(prob_df.set_index("Class"))
        st.table(prob_df.set_index("Class"))

        # ---- Segmentation result ----
        st.subheader("Tumor Segmentation")
        if seg_model is None:
            st.info("Segmentation model not available yet. Train it with `train_segmentation.py`.")
        elif seg_out is None:
            st.write("Skipped because no tumor class was reliably predicted. "
                     "Tick the box in the sidebar to run it anyway.")
        else:
            image, mask = seg_out
            area_pct = 100.0 * mask.sum() / mask.size
            if mask.sum() == 0:
                st.write("The segmentation model found no tumor region at this threshold. "
                         "Try lowering the threshold in the sidebar.")
            else:
                c1, c2, c3 = st.columns(3)
                c1.image(image, caption="MRI (224x224)", use_container_width=True)
                c2.image(mask * 255, caption="Predicted mask", use_container_width=True, clamp=True)
                c3.image(make_overlay(image, mask), caption="Tumor outlined", use_container_width=True)
                st.write(f"**Tumor area:** {area_pct:.1f}% of the image")
                st.caption("Area is a share of image pixels, not a real size in millimetres.")

# ----------------------------- Model performance -----------------------------
# Numbers are read from files made by evaluate_all.py, so nothing is typed in by hand.
RESULTS_DIR = "results"
SUMMARY_PATH = os.path.join(RESULTS_DIR, "summary.json")

st.divider()
with st.expander("📊 Model performance (measured on held-out images)"):
    if not os.path.isfile(SUMMARY_PATH):
        st.info("Scores not found. Run `python evaluate_all.py` once, then restart the app.")
    else:
        with open(SUMMARY_PATH, encoding="utf-8") as f:
            summary = json.load(f)

        cls_sum = summary.get("classification")
        if cls_sum:
            st.subheader("Classification")
            a, b = st.columns(2)
            a.metric("Test accuracy", f"{cls_sum['test_accuracy_percent']:.2f}%")
            b.metric("Macro F1-score", f"{cls_sum['macro_f1']:.3f}")
            st.caption(f"Measured on {cls_sum['test_images']} test images the model never saw during training.")
            cm_path = os.path.join(RESULTS_DIR, "cls_confusion_matrix.png")
            metrics_path = os.path.join(RESULTS_DIR, "cls_metrics.csv")
            if os.path.isfile(metrics_path):
                st.table(pd.read_csv(metrics_path).set_index("class"))
            if os.path.isfile(cm_path):
                st.image(cm_path, caption="Confusion matrix (test set)", use_container_width=True)

        seg_sum = summary.get("segmentation")
        if seg_sum:
            st.subheader("Segmentation")
            a, b = st.columns(2)
            a.metric("Validation Dice", f"{seg_sum['validation_dice']:.3f}")
            b.metric("Validation IoU", f"{seg_sum['validation_iou']:.3f}")
            st.caption(f"Measured on {seg_sum['validation_images']} validation images at mask threshold "
                       f"{seg_sum['threshold']}. Dice and IoU: 1.0 is a perfect match with the expert mask.")
            seg_path = os.path.join(RESULTS_DIR, "seg_metrics.csv")
            if os.path.isfile(seg_path):
                st.table(pd.read_csv(seg_path).set_index("subset"))
            q_path = os.path.join(RESULTS_DIR, "seg_qualitative.png")
            if os.path.isfile(q_path):
                st.image(q_path, caption="Examples: best, median and worst cases", use_container_width=True)

        st.caption("These scores describe this dataset only. They do not show how the model would perform on real patients.")

st.caption("Phase 2: classification + segmentation. Educational use only.")