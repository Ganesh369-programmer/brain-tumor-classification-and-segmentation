# Brain Tumor Detection and Multi-Class Classification Using Deep Learning

**Phase 1: Classification** (segmentation will be added in a later phase)

A CNN built with TensorFlow/Keras that classifies a brain MRI image as **glioma**, **meningioma**, **pituitary**, or **no_tumor**. One 4-class model handles both tumor detection and tumor type classification.

> **Disclaimer:** Academic project for educational purposes only. Not a medical diagnosis tool and not a replacement for a doctor or radiologist.

## Project structure

```
.
├── brain_tumor_classification.ipynb   # main implementation
├── requirements.txt
├── README.md
├── .gitignore
├── app.py                             # Streamlit app
├── models/
│   └── class_names.json
└── results/
    ├── class_distribution.png, accuracy_plot.png, loss_plot.png
    ├── confusion_matrix.png
    ├── classification_report.csv
    └── training_history.csv
```

The MRI dataset, generated `results/sample_images.png` montage, and trained `models/brain_tumor_cnn.keras` file are not included in this repository. The dataset and montage are excluded because the dataset's redistribution terms should be checked separately; the trained model is excluded because it exceeds GitHub's 100 MB per-file limit. The notebook creates the model and result files when run.

## Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook brain_tumor_classification.ipynb
```

Place the dataset in `dataset/` with these exact folder names:

```
dataset/
├── Training/
│   ├── glioma_tumor/
│   ├── meningioma_tumor/
│   ├── no_tumor/
│   └── pituitary_tumor/
└── Testing/
    ├── glioma_tumor/
    ├── meningioma_tumor/
    ├── no_tumor/
    └── pituitary_tumor/
```

## How it works

- **Split:** Training folder is split 80% train / 20% validation (stratified, `SEED = 42`). The Testing folder is only used for final evaluation.
- **Preprocessing:** RGB, resize to 224x224, divide by 255.0. One shared function is used for training, evaluation and prediction.
- **Augmentation (training only):** horizontal flip, small rotation, small zoom.
- **Model:** 3 x (Conv2D + ReLU + MaxPool) -> Flatten -> Dense(128) -> Dropout(0.5) -> Dense(4, softmax).
- **Training:** Adam, sparse categorical cross-entropy, up to 20 epochs with EarlyStopping.
- **Outputs:** test accuracy, classification report, confusion matrix, accuracy/loss plots, single-image prediction with confidence and class probabilities.

Run all cells top to bottom. If `models/brain_tumor_cnn.keras` already exists, the notebook loads it instead of retraining. Set `FORCE_RETRAIN = True` in the configuration cell to retrain.

## Predicting on a single image

Set `IMAGE_PATH` in the "Individual MRI Prediction" section and run the cells, or call:

```python
result = predict_mri("path/to/mri.jpg")
show_prediction(result)
```

## Run the Streamlit app

Train/save the model with the notebook first, then:

```bash
streamlit run app.py
```

Upload an MRI image and click **Classify** to see the status, tumor type, confidence and class probabilities.

## Using the saved model later (Streamlit / segmentation phase)

```python
import json, numpy as np, tensorflow as tf
model = tf.keras.models.load_model("models/brain_tumor_cnn.keras")
meta = json.load(open("models/class_names.json"))
```

Contract:

- **Input:** float32 array `(N, 224, 224, 3)`, RGB, values scaled to **0-1** (the model does not rescale internally).
- **Output:** 4 softmax probabilities in the order of `class_names.json`: `glioma_tumor, meningioma_tumor, no_tumor, pituitary_tumor`.
- No label is "tumor/no tumor" separately: detected = predicted class is not `no_tumor`.

## Future segmentation compatibility

The classifier is independent of any segmentation code. The planned pipeline:

```
MRI -> Preprocessing -> CNN Classifier  -> Detection + Tumor Type
                     -> Segmentation    -> Tumor Mask
                     -> Combined results (Streamlit app)
```

## Limitations

Trained on a single dataset, 2D images only, no tumor localization, softmax confidence can be overconfident, and results may not generalize to other scanners or hospitals.
