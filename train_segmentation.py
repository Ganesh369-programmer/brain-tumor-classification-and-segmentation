"""
Brain tumor SEGMENTATION - train a U-Net.

Expected dataset layout (image and mask must have the SAME file name stem):

    data_seg/
        images/  img_001.png, img_002.png, ...
        masks/   img_001.png, img_002.png, ...   (white = tumor, black = background)

Run:
    python train_segmentation.py

Output:
    models/brain_tumor_unet.keras
    models/seg_training_curves.png
    models/seg_sample_predictions.png
    results/seg_metrics.csv       Dice and IoU: overall + per tumor type (needs data_seg/labels.csv)
    results/seg_per_image.csv     Dice and IoU for every validation image
    results/summary.json          main scores, shown in the Streamlit app

NOTE: This is a research/educational project, not a medical device.
"""

import csv
import json
import os
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----------------------------- Settings -----------------------------
SEED = 42
IMG_SIZE = 224          # same size as the classifier
BATCH_SIZE = 16
EPOCHS = 40
LEARNING_RATE = 1e-3
DATA_DIR = "data_seg"
IMAGE_DIR = os.path.join(DATA_DIR, "images")
MASK_DIR = os.path.join(DATA_DIR, "masks")
MODEL_PATH = "models/brain_tumor_unet.keras"
FORCE_RETRAIN = False

os.makedirs("models", exist_ok=True)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


# ------------------------- 1. Pair images and masks -------------------------
def find_pairs():
    exts = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
    masks = {}
    for f in os.listdir(MASK_DIR):
        if f.lower().endswith(exts):
            masks[os.path.splitext(f)[0]] = os.path.join(MASK_DIR, f)
    pairs = []
    for f in sorted(os.listdir(IMAGE_DIR)):
        if f.lower().endswith(exts):
            stem = os.path.splitext(f)[0]
            if stem in masks:
                pairs.append((os.path.join(IMAGE_DIR, f), masks[stem]))
    return pairs


pairs = find_pairs()
if len(pairs) == 0:
    raise SystemExit(
        "No image/mask pairs found. Put images in data_seg/images and masks in "
        "data_seg/masks with matching file names."
    )
print(f"Found {len(pairs)} image/mask pairs")

random.shuffle(pairs)
split = int(0.8 * len(pairs))
train_pairs, val_pairs = pairs[:split], pairs[split:]
print(f"Train: {len(train_pairs)} | Validation: {len(val_pairs)}")


# ------------------------- 2. tf.data pipeline -------------------------
def load_pair(img_path, mask_path):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE))
    img = tf.cast(img, tf.float32) / 255.0            # same preprocessing as classifier

    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_image(mask, channels=1, expand_animations=False)
    mask = tf.image.resize(mask, (IMG_SIZE, IMG_SIZE), method="nearest")
    mask = tf.cast(mask > 127, tf.float32)            # binary 0/1
    img.set_shape((IMG_SIZE, IMG_SIZE, 3))
    mask.set_shape((IMG_SIZE, IMG_SIZE, 1))
    return img, mask


def augment(img, mask):
    # The SAME random flip must be applied to image and mask.
    if tf.random.uniform(()) > 0.5:
        img, mask = tf.image.flip_left_right(img), tf.image.flip_left_right(mask)
    if tf.random.uniform(()) > 0.5:
        img, mask = tf.image.flip_up_down(img), tf.image.flip_up_down(mask)
    return img, mask


def make_ds(pair_list, training):
    img_paths = [p[0] for p in pair_list]
    mask_paths = [p[1] for p in pair_list]
    ds = tf.data.Dataset.from_tensor_slices((img_paths, mask_paths))
    if training:
        ds = ds.shuffle(len(pair_list), seed=SEED)
    ds = ds.map(load_pair, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.ignore_errors()                            # skip corrupt files
    if training:
        ds = ds.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


train_ds = make_ds(train_pairs, training=True)
val_ds = make_ds(val_pairs, training=False)


# ------------------------- 3. Loss and metrics -------------------------
def dice_coef(y_true, y_pred, smooth=1.0):
    y_true = tf.reshape(y_true, [tf.shape(y_true)[0], -1])
    y_pred = tf.reshape(y_pred, [tf.shape(y_pred)[0], -1])
    inter = tf.reduce_sum(y_true * y_pred, axis=1)
    union = tf.reduce_sum(y_true, axis=1) + tf.reduce_sum(y_pred, axis=1)
    return tf.reduce_mean((2.0 * inter + smooth) / (union + smooth))


def dice_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)


def bce_dice_loss(y_true, y_pred):
    bce = tf.reduce_mean(tf.keras.losses.binary_crossentropy(y_true, y_pred))
    return bce + dice_loss(y_true, y_pred)


def iou_metric(y_true, y_pred, smooth=1.0):
    y_pred = tf.cast(y_pred > 0.5, tf.float32)
    y_true = tf.reshape(y_true, [tf.shape(y_true)[0], -1])
    y_pred = tf.reshape(y_pred, [tf.shape(y_pred)[0], -1])
    inter = tf.reduce_sum(y_true * y_pred, axis=1)
    union = tf.reduce_sum(y_true, axis=1) + tf.reduce_sum(y_pred, axis=1) - inter
    return tf.reduce_mean((inter + smooth) / (union + smooth))


# ------------------------- 4. U-Net model -------------------------
def conv_block(x, filters):
    x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    return x


def build_unet(input_shape=(IMG_SIZE, IMG_SIZE, 3), base=32):
    inputs = layers.Input(input_shape)

    # Encoder (goes down, learns "what" is in the image)
    c1 = conv_block(inputs, base)
    p1 = layers.MaxPooling2D()(c1)
    c2 = conv_block(p1, base * 2)
    p2 = layers.MaxPooling2D()(c2)
    c3 = conv_block(p2, base * 4)
    p3 = layers.MaxPooling2D()(c3)
    c4 = conv_block(p3, base * 8)
    p4 = layers.MaxPooling2D()(c4)

    # Bottleneck
    b = conv_block(p4, base * 16)

    # Decoder (goes up, learns "where" it is) + skip connections
    u4 = layers.Conv2DTranspose(base * 8, 2, strides=2, padding="same")(b)
    c5 = conv_block(layers.concatenate([u4, c4]), base * 8)
    u3 = layers.Conv2DTranspose(base * 4, 2, strides=2, padding="same")(c5)
    c6 = conv_block(layers.concatenate([u3, c3]), base * 4)
    u2 = layers.Conv2DTranspose(base * 2, 2, strides=2, padding="same")(c6)
    c7 = conv_block(layers.concatenate([u2, c2]), base * 2)
    u1 = layers.Conv2DTranspose(base, 2, strides=2, padding="same")(c7)
    c8 = conv_block(layers.concatenate([u1, c1]), base)

    outputs = layers.Conv2D(1, 1, activation="sigmoid")(c8)  # 1 channel mask, 0-1
    return models.Model(inputs, outputs, name="unet")


# ------------------------- 5. Train (or load) -------------------------
if os.path.exists(MODEL_PATH) and not FORCE_RETRAIN:
    print(f"Loading existing model: {MODEL_PATH}")
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(LEARNING_RATE),
        loss=bce_dice_loss,
        metrics=[dice_coef, iou_metric],
    )
    history = None
else:
    model = build_unet()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(LEARNING_RATE),
        loss=bce_dice_loss,
        metrics=[dice_coef, iou_metric],
    )
    model.summary()
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            MODEL_PATH, monitor="val_dice_coef", mode="max", save_best_only=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=4, min_lr=1e-6
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_dice_coef", mode="max", patience=10, restore_best_weights=True
        ),
    ]
    history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=callbacks)

    # Training curves
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(history.history["loss"], label="train")
    axes[0].plot(history.history["val_loss"], label="val")
    axes[0].set_title("Loss")
    axes[0].legend()
    axes[1].plot(history.history["dice_coef"], label="train")
    axes[1].plot(history.history["val_dice_coef"], label="val")
    axes[1].set_title("Dice score")
    axes[1].legend()
    plt.tight_layout()
    plt.savefig("models/seg_training_curves.png", dpi=120)
    plt.close()

# ------------------------- 6. Evaluate -------------------------
results = model.evaluate(val_ds, verbose=0, return_dict=True)
print(f"Validation  Dice: {results['dice_coef']:.4f} | IoU: {results['iou_metric']:.4f}")
print("(These two come from Keras and use the soft Dice. The exact paper scores are in 6b below.)")

# ------------------------- 6b. Dice similarity and IoU (hard mask, threshold 0.5) -------------------------
# Dice = 2 * |P and G| / (|P| + |G|)        IoU = |P and G| / |P or G|
# P = predicted mask (probability > THRESHOLD), G = true mask. 1.0 = perfect overlap.
# Scores are worked out for each validation image, then averaged: overall and per tumor type.
THRESHOLD = 0.5
RESULTS_DIR = "results"
LABELS_CSV = os.path.join(DATA_DIR, "labels.csv")   # made by prepare_segmentation_data.py
os.makedirs(RESULTS_DIR, exist_ok=True)


def dice_iou_np(gt, pred, smooth=1.0):
    inter = float((gt * pred).sum())
    g, p = float(gt.sum()), float(pred.sum())
    dice = (2.0 * inter + smooth) / (g + p + smooth)
    iou = (inter + smooth) / (g + p - inter + smooth)
    return dice, iou


# Load the validation images one by one so every score is tied to its file name
val_imgs, val_gts, val_stems = [], [], []
for img_path, mask_path in val_pairs:
    try:
        i_t, m_t = load_pair(img_path, mask_path)
    except Exception:
        continue                                        # skip a corrupt file
    val_imgs.append(i_t.numpy())
    val_gts.append(m_t.numpy()[..., 0])
    val_stems.append(os.path.splitext(os.path.basename(img_path))[0])
val_imgs, val_gts = np.stack(val_imgs), np.stack(val_gts)

val_probs = np.concatenate(
    [model.predict(val_imgs[s:s + BATCH_SIZE], verbose=0) for s in range(0, len(val_imgs), BATCH_SIZE)]
)
val_preds = (val_probs[..., 0] > THRESHOLD).astype(np.float32)

# Tumor type of each image (optional: only if labels.csv exists)
tumor_type = {}
if os.path.isfile(LABELS_CSV):
    with open(LABELS_CSV) as f:
        for row in csv.DictReader(f):
            tumor_type[row["file"]] = row["class"]
else:
    print("labels.csv not found: scores per tumor type will be skipped.")

rows = []
for k, stem in enumerate(val_stems):
    d, u = dice_iou_np(val_gts[k], val_preds[k])
    rows.append((stem, tumor_type.get(stem, ""), d, u))

with open(os.path.join(RESULTS_DIR, "seg_per_image.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["file", "class", "dice", "iou"])
    for r in rows:
        w.writerow([r[0], r[1], f"{r[2]:.4f}", f"{r[3]:.4f}"])

seg_summary = [("Overall", len(rows), float(np.mean([r[2] for r in rows])), float(np.mean([r[3] for r in rows])))]
for c in sorted({r[1] for r in rows if r[1]}):
    sub = [r for r in rows if r[1] == c]
    seg_summary.append((c, len(sub), float(np.mean([r[2] for r in sub])), float(np.mean([r[3] for r in sub]))))

with open(os.path.join(RESULTS_DIR, "seg_metrics.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["subset", "n_images", "dice", "iou"])
    for s in seg_summary:
        w.writerow([s[0], s[1], f"{s[2]:.4f}", f"{s[3]:.4f}"])

# Save the main scores so the Streamlit app can show them
summary_path = os.path.join(RESULTS_DIR, "summary.json")
summary_data = {}
if os.path.isfile(summary_path):
    with open(summary_path, encoding="utf-8") as f:
        summary_data = json.load(f)
summary_data["segmentation"] = {
    "validation_dice": round(seg_summary[0][2], 4),
    "validation_iou": round(seg_summary[0][3], 4),
    "validation_images": seg_summary[0][1],
    "threshold": THRESHOLD,
}
with open(summary_path, "w", encoding="utf-8") as f:
    json.dump(summary_data, f, indent=2)

print(f"\nDice and IoU at threshold {THRESHOLD} (mean over validation images)")
print(f"{'Subset':<20}{'Images':>8}{'Dice':>9}{'IoU':>9}")
for s in seg_summary:
    print(f"{s[0]:<20}{s[1]:>8}{s[2]:>9.4f}{s[3]:>9.4f}")
print("Saved: results/seg_metrics.csv, results/seg_per_image.csv, results/summary.json\n")

# ------------------------- 7. Sample predictions -------------------------
imgs, masks_true = next(iter(val_ds))
preds = model.predict(imgs, verbose=0)
n = min(4, len(imgs))
fig, axes = plt.subplots(n, 3, figsize=(9, 3 * n))
axes = np.atleast_2d(axes)
for i in range(n):
    axes[i, 0].imshow(imgs[i])
    axes[i, 0].set_title("MRI")
    axes[i, 1].imshow(masks_true[i, :, :, 0], cmap="gray")
    axes[i, 1].set_title("True mask")
    axes[i, 2].imshow(preds[i, :, :, 0] > 0.5, cmap="gray")
    axes[i, 2].set_title("Predicted mask")
    for a in axes[i]:
        a.axis("off")
plt.tight_layout()
plt.savefig("models/seg_sample_predictions.png", dpi=120)
plt.close()
print("Saved:", MODEL_PATH)
print("Disclaimer: educational project, not for medical diagnosis.")