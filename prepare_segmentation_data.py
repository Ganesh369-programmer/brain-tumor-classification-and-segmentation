"""Convert the Figshare brain tumor dataset (Jun Cheng) into PNG images + masks.

Why: your classification dataset has NO masks, so it cannot train segmentation.
The Figshare dataset has the same 3 tumor types (glioma, meningioma, pituitary)
with a hand-drawn tumor mask for every image (3064 images, stored as .mat files).

Steps:
  1. Download the dataset from Figshare (search: "brain tumor dataset Jun Cheng").
     It comes as 4 zip files. Extract ALL the .mat files into one folder,
     for example:  raw_mat/1.mat, raw_mat/2.mat, ...
  2. Run:
         python prepare_segmentation_data.py raw_mat data_seg
  3. You get:
         data_seg/images/*.png
         data_seg/masks/*.png     (white = tumor, black = background)
         data_seg/labels.csv
"""
import csv
import glob
import os
import sys

import cv2
import h5py
import numpy as np

LABELS = {1: "meningioma_tumor", 2: "glioma_tumor", 3: "pituitary_tumor"}


def main(src, dst):
    files = sorted(glob.glob(os.path.join(src, "*.mat")))
    if not files:
        raise SystemExit(f"No .mat files found in '{src}'.")

    img_dir = os.path.join(dst, "images")
    mask_dir = os.path.join(dst, "masks")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)

    rows, skipped = [], 0
    for path in files:
        stem = os.path.splitext(os.path.basename(path))[0]
        try:
            with h5py.File(path, "r") as f:
                d = f["cjdata"]
                # .T because MATLAB stores arrays column-major
                img = np.array(d["image"]).T.astype(np.float32)
                mask = np.array(d["tumorMask"]).T
                label = int(np.array(d["label"]).squeeze())
        except Exception as e:
            print(f"Skipping {path}: {e}")
            skipped += 1
            continue

        # scale image to 0-255
        rng = img.max() - img.min()
        img = (img - img.min()) / rng * 255.0 if rng > 0 else np.zeros_like(img)
        cv2.imwrite(os.path.join(img_dir, stem + ".png"), img.astype(np.uint8))
        cv2.imwrite(os.path.join(mask_dir, stem + ".png"), (mask > 0).astype(np.uint8) * 255)
        rows.append((stem, LABELS.get(label, str(label))))

    with open(os.path.join(dst, "labels.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file", "class"])
        w.writerows(rows)

    print(f"Done: {len(rows)} image/mask pairs written to '{dst}' ({skipped} skipped).")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("Usage: python prepare_segmentation_data.py <folder_with_mat_files> <output_folder>")
    main(sys.argv[1], sys.argv[2])
