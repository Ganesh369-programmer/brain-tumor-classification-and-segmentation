# Brain Tumor Classification Project — Comprehensive Overview

> **Purpose of this document:** Complete technical reference for this project, intended to be handed off to another AI assistant for modifications or enhancements.

---

## 1. Project Summary

| Field | Value |
|---|---|
| **Name** | Brain Tumor Detection and Multi-Class Classification Using Deep Learning |
| **Phase** | Phase 1 — Classification only (segmentation planned for Phase 2) |
| **Goal** | Classify brain MRI images into 4 categories: Glioma, Meningioma, Pituitary Tumor, or No Tumor |
| **Tech Stack** | Python, TensorFlow/Keras, Streamlit, NumPy, Pandas, OpenCV, scikit-learn, Matplotlib, Pillow |
| **Interface** | Streamlit web app (`app.py`) |
| **Disclaimer** | Academic/educational project — NOT a medical diagnosis tool |

---

## 2. Project File Structure

```
brain tumor project/
│
├── brain_tumor_classification.ipynb   ← Main implementation (model training, evaluation, prediction)
├── app.py                             ← Streamlit web application (inference UI)
├── requirements.txt                   ← Python dependencies
├── README.md                          ← Project documentation
├── .gitignore                         ← Excludes dataset/, *.keras, venv/, sample_images.png
│
├── dataset/                           ← NOT in git (excluded by .gitignore)
│   ├── Training/
│   │   ├── glioma_tumor/              ← 826 images
│   │   ├── meningioma_tumor/          ← 822 images
│   │   ├── no_tumor/                  ← 395 images
│   │   └── pituitary_tumor/           ← 827 images
│   └── Testing/
│       ├── glioma_tumor/              ← 100 images
│       ├── meningioma_tumor/          ← 115 images
│       ├── no_tumor/                  ← 105 images
│       └── pituitary_tumor/           ← 74 images
│
├── models/
│   ├── brain_tumor_cnn.keras          ← Trained model (~128 MB, NOT in git)
│   └── class_names.json               ← Model metadata (class names, display names, img_size)
│
├── results/
│   ├── accuracy_plot.png              ← Training vs validation accuracy graph
│   ├── loss_plot.png                  ← Training vs validation loss graph
│   ├── confusion_matrix.png           ← Test set confusion matrix
│   ├── class_distribution.png         ← Bar chart of class distribution
│   ├── sample_images.png              ← Montage of sample MRI images (NOT in git)
│   ├── classification_report.csv      ← Per-class precision/recall/f1
│   └── training_history.csv           ← Epoch-by-epoch train/val accuracy and loss
│
├── venv/                              ← Virtual environment (NOT in git)
└── .venv/                             ← Alternate virtual environment (NOT in git)
```

---

## 3. Dataset Details

- **Source:** A standard brain MRI tumor dataset (redistribution not included in repo)
- **Format:** Image files (JPG/JPEG/PNG) organized in class-named folders
- **Split strategy:** Training folder is split 80% train / 20% validation (stratified, `SEED = 42`). The Testing folder is held out for final evaluation only.

### Class Distribution

| Class | Training Images | Testing Images | Total |
|---|---|---|---|
| `glioma_tumor` | 826 | 100 | 926 |
| `meningioma_tumor` | 822 | 115 | 937 |
| `no_tumor` | 395 | 105 | 500 |
| `pituitary_tumor` | 827 | 74 | 901 |
| **Total** | **2,870** | **394** | **3,264** |

> ⚠️ **Class Imbalance:** `no_tumor` has significantly fewer training samples (~47% of other classes). This likely contributes to model weaknesses.

---

## 4. Model Architecture

**Model file:** `models/brain_tumor_cnn.keras` (TensorFlow/Keras `.keras` format)

**Architecture:** Custom CNN (Convolutional Neural Network)

```
Input: (None, 224, 224, 3)  ← float32, RGB, pre-scaled to [0, 1]
   │
   ├─ Conv2D(filters, 3x3) + ReLU
   ├─ MaxPooling2D
   │
   ├─ Conv2D(filters, 3x3) + ReLU
   ├─ MaxPooling2D
   │
   ├─ Conv2D(filters, 3x3) + ReLU
   ├─ MaxPooling2D
   │
   ├─ Flatten
   ├─ Dense(128) + ReLU
   ├─ Dropout(0.5)
   └─ Dense(4) + Softmax  ← 4 output probabilities
```

**Pattern:** 3 × (Conv2D → ReLU → MaxPool) → Flatten → Dense(128) → Dropout(0.5) → Dense(4, softmax)

### Model Contract (Important for Integration)

| Property | Value |
|---|---|
| **Input dtype** | `float32` |
| **Input shape** | `(N, 224, 224, 3)` — batch of N RGB images |
| **Input range** | `[0.0, 1.0]` — model does NOT rescale internally |
| **Output** | 4 softmax probabilities |
| **Output order** | `glioma_tumor`, `meningioma_tumor`, `no_tumor`, `pituitary_tumor` |
| **Detection logic** | `detected = predicted_class != "no_tumor"` |

---

## 5. Training Configuration

| Hyperparameter | Value |
|---|---|
| **Optimizer** | Adam (default learning rate) |
| **Loss function** | Sparse Categorical Cross-Entropy |
| **Max epochs** | 20 |
| **Early stopping** | Yes (monitors validation loss) |
| **Random seed** | `SEED = 42` |
| **Train/Val split** | 80% / 20% (stratified) |
| **Image size** | 224 × 224 px |
| **Color mode** | RGB |
| **Normalization** | Pixel values / 255.0 |

### Data Augmentation (Training only)

- Horizontal flip
- Small random rotation
- Small random zoom

---

## 6. Training History (20 Epochs)

| Epoch | Train Acc | Train Loss | Val Acc | Val Loss |
|---|---|---|---|---|
| 0 | 46.0% | 1.258 | 61.7% | 0.997 |
| 5 | 70.3% | 0.700 | 75.8% | 0.579 |
| 10 | 77.0% | 0.544 | 72.8% | 0.659 |
| 15 | 80.0% | 0.482 | 84.1% | 0.401 |
| 19 | 82.9% | 0.402 | 83.8% | 0.395 |

**Best validation accuracy achieved:** ~84.1% (epoch 15)  
**Final validation accuracy (epoch 19):** ~83.8%

---

## 7. Model Performance on Test Set

**Overall test accuracy: ~56.1%** ← significantly lower than validation accuracy (possible overfitting or domain gap)

### Per-Class Results

| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| **Glioma** | 1.000 | 0.120 | 0.214 | 100 |
| **Meningioma** | 0.585 | 0.600 | 0.592 | 115 |
| **No Tumor** | 0.454 | 0.943 | 0.613 | 105 |
| **Pituitary** | 0.891 | 0.554 | 0.683 | 74 |
| **Accuracy** | — | — | **0.561** | 394 |
| **Macro avg** | 0.733 | 0.554 | 0.525 | 394 |
| **Weighted avg** | 0.713 | 0.561 | 0.519 | 394 |

### Key Observations

- **Glioma:** Very high precision (1.0) but terrible recall (12%) — the model almost never correctly predicts glioma.
- **No Tumor:** High recall (94%) but low precision (45%) — many tumors misclassified as no_tumor.
- **Pituitary:** Good precision (89%) but mediocre recall (55%).
- **Meningioma:** Balanced but mediocre overall (~59% F1).
- **Gap:** Validation accuracy (~84%) vs Test accuracy (~56%) suggests significant overfitting or distribution mismatch between train-val and test splits.

---

## 8. `models/class_names.json` — Full Contents

```json
{
  "class_names": [
    "glioma_tumor",
    "meningioma_tumor",
    "no_tumor",
    "pituitary_tumor"
  ],
  "display_names": {
    "glioma_tumor": "Glioma",
    "meningioma_tumor": "Meningioma",
    "pituitary_tumor": "Pituitary",
    "no_tumor": "No Tumor"
  },
  "img_size": [224, 224],
  "preprocessing": "RGB, resize to 224x224, divide by 255.0 (float32, range 0-1)"
}
```

---

## 9. Streamlit App (`app.py`) — Full Logic

**Run command:** `streamlit run app.py`

### App Flow

1. Page configured: title "Brain Tumor Classification", icon 🧠, layout centered
2. Load model + metadata once via `@st.cache_resource`
3. Show disclaimer warning
4. File uploader (accepts jpg, jpeg, png, bmp)
5. Display uploaded image
6. "Classify" button → preprocess → `model.predict()` → show results
7. Results shown: detection status, tumor type (if detected), confidence %, bar chart of class probabilities, probability table

### Preprocessing Function (`preprocess`)
```python
def preprocess(image_bytes, img_size):
    image = tf.io.decode_image(image_bytes, channels=3, expand_animations=False)
    image = tf.image.resize(image, img_size)
    image = tf.cast(image, tf.float32) / 255.0
    return tf.expand_dims(image, axis=0)  # add batch dimension
```

### Prediction Output Dictionary
```python
{
    "detected": bool,                  # True if not no_tumor
    "status": "Tumor Detected" | "No Tumor Detected",
    "tumor_type": str | None,          # Display name if detected
    "prediction": str,                 # Display name always
    "confidence": float,               # 0-100 %
    "probabilities": {                 # All 4 class probs (0-100)
        "Glioma": float,
        "Meningioma": float,
        "No Tumor": float,
        "Pituitary": float,
    }
}
```

---

## 10. Dependencies (`requirements.txt`)

```
tensorflow>=2.15
numpy
pandas
matplotlib
scikit-learn
opencv-python
Pillow
jupyter
streamlit
```

---

## 11. Notebook (`brain_tumor_classification.ipynb`)

The notebook is the **main implementation file** and performs all of the following:

1. **Configuration cell:** Sets `SEED`, `IMG_SIZE`, `BATCH_SIZE`, `MODEL_PATH`, `FORCE_RETRAIN`
2. **Data loading:** Reads from `dataset/Training/` and `dataset/Testing/` using `tf.keras.utils.image_dataset_from_directory` or similar
3. **Preprocessing:** Single shared `preprocess_image()` function used for training, evaluation, and prediction
4. **Augmentation:** Applied to training split only
5. **Model definition:** Custom CNN as described above
6. **Training:** Adam optimizer, sparse categorical cross-entropy, EarlyStopping on val_loss
7. **Save/Load logic:** Checks if `models/brain_tumor_cnn.keras` exists; loads it if `FORCE_RETRAIN = False`; saves after training
8. **Evaluation:** Generates confusion matrix, classification report, accuracy/loss plots
9. **Single-image prediction:** `predict_mri(path)` + `show_prediction(result)` helpers
10. **Output files saved:**
    - `models/brain_tumor_cnn.keras`
    - `models/class_names.json`
    - `results/accuracy_plot.png`
    - `results/loss_plot.png`
    - `results/confusion_matrix.png`
    - `results/class_distribution.png`
    - `results/sample_images.png`
    - `results/classification_report.csv`
    - `results/training_history.csv`

---

## 12. Planned Future Work (from README)

### Phase 2: Segmentation

The planned full pipeline:

```
MRI → Preprocessing → CNN Classifier  → Detection + Tumor Type
                    → Segmentation    → Tumor Mask
                    → Combined results (Streamlit app)
```

- Classifier is intentionally designed to be **independent** of segmentation code
- Segmentation will be added as a separate branch/module

---

## 13. Known Issues & Limitations

| Issue | Detail |
|---|---|
| **Low test accuracy** | ~56% on test set vs ~84% validation — major gap, possible overfitting |
| **Glioma recall collapse** | Only 12% recall on glioma test images — model rarely predicts it correctly |
| **No Tumor low precision** | 45% — many tumors misclassified as No Tumor |
| **Class imbalance** | `no_tumor` has ~47% fewer training images than other classes |
| **2D only** | No volumetric (3D) MRI processing |
| **No localization** | Classification only — no bounding box or pixel-level tumor location |
| **Softmax overconfidence** | Softmax can output high confidence even on wrong predictions |
| **Single dataset** | May not generalize to other MRI scanners or hospitals |
| **No explainability** | No GradCAM, LIME, or SHAP — predictions are black-box |

---

## 14. How to Run the Project

### Setup
```bash
python -m venv venv
venv\Scripts\activate         # Windows
pip install -r requirements.txt
```

### Train / Evaluate
```bash
jupyter notebook brain_tumor_classification.ipynb
# Run all cells top to bottom
# Set FORCE_RETRAIN = True to retrain even if model exists
```

### Run Streamlit App
```bash
# Train model first, then:
streamlit run app.py
```

### Use the Saved Model Programmatically
```python
import json, numpy as np, tensorflow as tf

model = tf.keras.models.load_model("models/brain_tumor_cnn.keras")
meta = json.load(open("models/class_names.json"))

# Preprocess your image to shape (1, 224, 224, 3), float32, range [0,1]
probs = model.predict(image_array)[0]
predicted_class = meta["class_names"][np.argmax(probs)]
```

---

## 15. Repository / Git Info

- **GitHub repo:** `Ganesh369-programmer/brain-tumor-classification-and-segmentation`
- **Files excluded from git:** `dataset/`, `*.keras`, `venv/`, `.venv/`, `results/sample_images.png`, `__pycache__/`, `.ipynb_checkpoints/`
- **Files included in git:** `app.py`, `brain_tumor_classification.ipynb`, `requirements.txt`, `README.md`, `.gitignore`, `models/class_names.json`, all result CSVs and PNGs (except sample_images.png)

---

*Document generated: 2026-10-20*
