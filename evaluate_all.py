"""Make every missing number and figure for the research paper.

Run from the project root (the folder that contains dataset/, models/, data_seg/):
    python evaluate_all.py

Part A (classification, needs models/brain_tumor_cnn.keras + dataset/Testing):
    results/cls_report.txt             full sklearn report
    results/cls_metrics.csv            precision, recall, F1, specificity per class
    results/cls_confusion_matrix.png   confusion matrix figure
    results/cls_confusions.txt         which classes get confused (for the discussion)

Part B (segmentation, needs models/brain_tumor_unet.keras + data_seg):
    results/seg_metrics.csv            Dice and IoU: overall + per tumor type
    results/seg_per_image.csv          Dice and IoU for every validation image
    results/seg_qualitative.png        input / true mask / predicted mask / overlay

Each part is skipped with a message if its files are missing.
Educational project only. Not a medical device.
"""
import csv
import json
import os
import random

import cv2
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

# ----------------------------- Settings -----------------------------
TEST_DIR = os.path.join("dataset", "Testing")
CLS_MODEL = os.path.join("models", "brain_tumor_cnn.keras")
META_PATH = os.path.join("models", "class_names.json")
SEG_MODEL = os.path.join("models", "brain_tumor_unet.keras")
SEG_DIR = "data_seg"
IMAGE_DIR = os.path.join(SEG_DIR, "images")
MASK_DIR = os.path.join(SEG_DIR, "masks")
LABELS_CSV = os.path.join(SEG_DIR, "labels.csv")
OUT = "results"
SEED = 42            # must match train_segmentation.py so we get the SAME validation images
THRESHOLD = 0.5      # mask threshold
IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

os.makedirs(OUT, exist_ok=True)


def update_summary(new):
    """Save the main scores in results/summary.json so the Streamlit app can show them."""
    path = os.path.join(OUT, "summary.json")
    data = {}
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    data.update(new)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


# =====================================================================
# PART A - CLASSIFICATION
# =====================================================================
def load_cls_image(path, size):
    """Same as the app: RGB, exif fix, resize, float32 in [0, 255] (EfficientNetB0 rescales inside)."""
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")
        arr = np.asarray(im)
    t = tf.image.resize(tf.convert_to_tensor(arr, dtype=tf.float32), size)
    return t.numpy()


def part_a():
    if not (os.path.isfile(CLS_MODEL) and os.path.isfile(META_PATH) and os.path.isdir(TEST_DIR)):
        print("Part A skipped: need", CLS_MODEL, META_PATH, "and", TEST_DIR)
        return
    with open(META_PATH, encoding="utf-8") as f:
        meta = json.load(f)
    names = meta["class_names"]
    display = [meta["display_names"][n] for n in names]
    size = tuple(meta["img_size"])
    model = tf.keras.models.load_model(CLS_MODEL)

    paths, y_true = [], []
    for i, cname in enumerate(names):
        folder = os.path.join(TEST_DIR, cname)
        if not os.path.isdir(folder):
            raise SystemExit(f"Test folder not found: {folder}")
        for fn in sorted(os.listdir(folder)):
            if fn.lower().endswith(IMG_EXTS):
                paths.append(os.path.join(folder, fn))
                y_true.append(i)
    print(f"Part A: {len(paths)} test images")

    probs = []
    for s in range(0, len(paths), 32):
        batch = np.stack([load_cls_image(p, size) for p in paths[s:s + 32]])
        probs.append(model.predict(batch, verbose=0))
    probs = np.concatenate(probs)
    y_pred = probs.argmax(axis=1)
    y_true = np.array(y_true)

    acc = float((y_pred == y_true).mean())
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(names))))
    report = classification_report(y_true, y_pred, target_names=display, digits=4, output_dict=True)
    report_txt = classification_report(y_true, y_pred, target_names=display, digits=4)

    # specificity per class = TN / (TN + FP)
    total = cm.sum()
    spec = []
    for i in range(len(names)):
        fp = cm[:, i].sum() - cm[i, i]
        tn = total - cm[i, :].sum() - cm[:, i].sum() + cm[i, i]
        spec.append(tn / (tn + fp) if (tn + fp) else float("nan"))

    # share flagged "uncertain" by the app's rule
    min_conf = float(meta.get("minimum_confidence", 0.65))
    min_margin = float(meta.get("minimum_top_two_margin", 0.15))
    srt = np.sort(probs, axis=1)
    uncertain = (srt[:, -1] < min_conf) | ((srt[:, -1] - srt[:, -2]) < min_margin)

    with open(os.path.join(OUT, "cls_report.txt"), "w") as f:
        f.write(f"Test images: {len(paths)}\nTest accuracy: {acc * 100:.2f}%\n\n{report_txt}\n")
        f.write("Specificity per class:\n")
        for d, s in zip(display, spec):
            f.write(f"  {d}: {s:.4f}\n")
        f.write(f"\nImages flagged uncertain by the app rule "
                f"(top<{min_conf}, gap<{min_margin}): {int(uncertain.sum())} "
                f"({100 * uncertain.mean():.1f}%)\n")

    with open(os.path.join(OUT, "cls_metrics.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["class", "precision", "recall", "f1", "specificity", "support"])
        for d, s in zip(display, spec):
            r = report[d]
            w.writerow([d, f"{r['precision']:.4f}", f"{r['recall']:.4f}", f"{r['f1-score']:.4f}",
                        f"{s:.4f}", int(r["support"])])
        for k in ("macro avg", "weighted avg"):
            r = report[k]
            w.writerow([k, f"{r['precision']:.4f}", f"{r['recall']:.4f}", f"{r['f1-score']:.4f}", "", int(r["support"])])

    # most common confusions, for the discussion section
    pairs = []
    for i in range(len(names)):
        for j in range(len(names)):
            if i != j and cm[i, j] > 0:
                pairs.append((int(cm[i, j]), display[i], display[j], 100.0 * cm[i, j] / cm[i, :].sum()))
    pairs.sort(reverse=True)
    with open(os.path.join(OUT, "cls_confusions.txt"), "w") as f:
        f.write("Most common mistakes (true class -> predicted class):\n")
        for n, a, b, pct in pairs:
            f.write(f"  {n:3d} images ({pct:.1f}% of true {a}): {a} predicted as {b}\n")

    # confusion matrix figure
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(display)))
    ax.set_yticks(range(len(display)))
    ax.set_xticklabels(display, rotation=30, ha="right")
    ax.set_yticklabels(display)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title(f"Confusion matrix (test accuracy {acc * 100:.2f}%)")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "cls_confusion_matrix.png"), dpi=200)
    plt.close()

    update_summary({
        "classification": {
            "test_accuracy_percent": round(acc * 100, 2),
            "test_images": int(len(paths)),
            "macro_f1": round(float(report["macro avg"]["f1-score"]), 4),
            "uncertain_percent": round(float(100 * uncertain.mean()), 1),
        }
    })
    print(f"Test accuracy: {acc * 100:.2f}%")
    print(report_txt)
    print("Saved Part A files in", OUT)


# =====================================================================
# PART B - SEGMENTATION
# =====================================================================
def find_pairs():
    """Same function as train_segmentation.py."""
    masks = {}
    for f in os.listdir(MASK_DIR):
        if f.lower().endswith(IMG_EXTS):
            masks[os.path.splitext(f)[0]] = os.path.join(MASK_DIR, f)
    pairs = []
    for f in sorted(os.listdir(IMAGE_DIR)):
        if f.lower().endswith(IMG_EXTS):
            stem = os.path.splitext(f)[0]
            if stem in masks:
                pairs.append((os.path.join(IMAGE_DIR, f), masks[stem]))
    return pairs


def load_seg_pair(img_path, mask_path, size):
    """Same as train_segmentation.py: image RGB /255, mask resized (nearest) and made binary."""
    img = tf.io.decode_image(tf.io.read_file(img_path), channels=3, expand_animations=False)
    img = tf.cast(tf.image.resize(img, size), tf.float32) / 255.0
    msk = tf.io.decode_image(tf.io.read_file(mask_path), channels=1, expand_animations=False)
    msk = tf.image.resize(msk, size, method="nearest")
    msk = tf.cast(msk > 127, tf.float32)
    return img.numpy(), msk.numpy()[..., 0]


def dice_iou(gt, pred, smooth=1.0):
    inter = float((gt * pred).sum())
    g, p = float(gt.sum()), float(pred.sum())
    dice = (2 * inter + smooth) / (g + p + smooth)
    iou = (inter + smooth) / (g + p - inter + smooth)
    return dice, iou


def overlay(img, gt, pred):
    """Image with GT outline (green) and predicted outline + tint (red)."""
    out = (img * 255).astype(np.uint8).copy()
    m = pred.astype(bool)
    out[m] = (0.65 * out[m] + 0.35 * np.array([255, 0, 0])).astype(np.uint8)
    for mask, color in ((gt, (0, 255, 0)), (pred, (255, 255, 0))):
        cs, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, cs, -1, color, 1)
    return out


def part_b():
    if not (os.path.isfile(SEG_MODEL) and os.path.isdir(IMAGE_DIR) and os.path.isdir(MASK_DIR)):
        print("Part B skipped: need", SEG_MODEL, "and", SEG_DIR, "(images/ and masks/)")
        return
    model = tf.keras.models.load_model(SEG_MODEL, compile=False)
    size = tuple(model.input_shape[1:3])

    # Rebuild the SAME 80/20 split as train_segmentation.py (seed 42, shuffle, first 80% = train)
    random.seed(SEED)
    pairs = find_pairs()
    random.shuffle(pairs)
    val_pairs = pairs[int(0.8 * len(pairs)):]
    print(f"Part B: {len(val_pairs)} validation pairs (of {len(pairs)})")

    classes = {}
    if os.path.isfile(LABELS_CSV):
        with open(LABELS_CSV) as f:
            for row in csv.DictReader(f):
                classes[row["file"]] = row["class"]
    else:
        print("labels.csv not found: per tumor type scores will be skipped.")

    imgs, gts = [], []
    for ip, mp in val_pairs:
        i, m = load_seg_pair(ip, mp, size)
        imgs.append(i)
        gts.append(m)
    imgs = np.stack(imgs)
    gts = np.stack(gts)
    preds = np.concatenate([model.predict(imgs[s:s + 16], verbose=0) for s in range(0, len(imgs), 16)])
    preds = (preds[..., 0] > THRESHOLD).astype(np.float32)

    rows = []
    for k, (ip, _) in enumerate(val_pairs):
        stem = os.path.splitext(os.path.basename(ip))[0]
        d, u = dice_iou(gts[k], preds[k])
        rows.append((stem, classes.get(stem, ""), d, u))

    with open(os.path.join(OUT, "seg_per_image.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file", "class", "dice", "iou"])
        for r in rows:
            w.writerow([r[0], r[1], f"{r[2]:.4f}", f"{r[3]:.4f}"])

    summary = [("Overall", len(rows), np.mean([r[2] for r in rows]), np.mean([r[3] for r in rows]))]
    for c in sorted({r[1] for r in rows if r[1]}):
        sub = [r for r in rows if r[1] == c]
        summary.append((c, len(sub), np.mean([r[2] for r in sub]), np.mean([r[3] for r in sub])))
    with open(os.path.join(OUT, "seg_metrics.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["subset", "n_images", "dice", "iou"])
        for s in summary:
            w.writerow([s[0], s[1], f"{s[2]:.4f}", f"{s[3]:.4f}"])
    update_summary({
        "segmentation": {
            "validation_dice": round(float(summary[0][2]), 4),
            "validation_iou": round(float(summary[0][3]), 4),
            "validation_images": int(summary[0][1]),
            "threshold": THRESHOLD,
        }
    })
    print("subset, n, Dice, IoU")
    for s in summary:
        print(f"  {s[0]}: n={s[1]}  Dice={s[2]:.4f}  IoU={s[3]:.4f}")

    # Qualitative figure: 2 best, 2 median, 2 worst by Dice
    order = np.argsort([r[2] for r in rows])
    mid = len(order) // 2
    chosen = list(order[-2:][::-1]) + list(order[mid - 1:mid + 1]) + list(order[:2])
    titles = ["Best", "Best", "Median", "Median", "Worst", "Worst"]
    fig, axes = plt.subplots(len(chosen), 4, figsize=(10, 2.6 * len(chosen)))
    for r, (k, t) in enumerate(zip(chosen, titles)):
        axes[r, 0].imshow(imgs[k])
        axes[r, 1].imshow(gts[k], cmap="gray")
        axes[r, 2].imshow(preds[k], cmap="gray")
        axes[r, 3].imshow(overlay(imgs[k], gts[k], preds[k]))
        axes[r, 0].set_ylabel(f"{t}\nDice {rows[k][2]:.2f}\n{rows[k][1].replace('_tumor', '')}", fontsize=8)
        for c in range(4):
            axes[r, c].set_xticks([])
            axes[r, c].set_yticks([])
    for c, t in enumerate(["Input MRI", "True mask", "Predicted mask", "Overlay (green=true, yellow=pred)"]):
        axes[0, c].set_title(t, fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "seg_qualitative.png"), dpi=200)
    plt.close()
    print("Saved Part B files in", OUT)


if __name__ == "__main__":
    part_a()
    part_b()
    print("Educational project only. Not for medical diagnosis.")
