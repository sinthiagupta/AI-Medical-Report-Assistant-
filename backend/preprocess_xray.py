# backend/augment_map_cluster_fullhand.py
"""
Single-script pipeline:
 - Augment dataset (albumentations) with bbox transforms preserved
 - Apply light CLAHE+denoise (NO rotation) before augment
 - Save augmented bbox CSVs
 - Run KMeans on x_center_norm to derive finger clusters (adaptive)
 - Detect wrist/whole-hand boxes and label them 'wrist'
 - Save final mapping CSVs for training

Outputs:
 data/xray_aug/train/images/   (augmented images)
 data/xray_aug/test/images/
 data/xray_aug/train_labels.csv    (aug bbox bboxes)
 data/xray_aug/test_labels.csv
 data/xray_aug/train_fracture_locations_kmeans.csv
 data/xray_aug/test_fracture_locations_kmeans.csv
 backend/kmeans_xnorm.joblib
 backend/kmeans_centers.csv
 backend/debug_hist_xnorm.png
"""
import os, sys, math, random
from pathlib import Path
import shutil
import pandas as pd
import numpy as np
from PIL import Image
import cv2
import albumentations as A

# clustering & utils
from sklearn.cluster import KMeans
import joblib
import matplotlib.pyplot as plt

# ---------- USER CONFIG (edit if needed) ----------
SRC_TRAIN_CSV = "data/xray/train/train_labels.csv"
SRC_TEST_CSV  = "data/xray/test/test_labels.csv"
SRC_TRAIN_IMG_DIR = "data/xray/train/images"
SRC_TEST_IMG_DIR  = "data/xray/test/images"

OUT_ROOT = "data/xray_aug"
OUT_TRAIN_IMG_DIR = os.path.join(OUT_ROOT, "train", "images")
OUT_TEST_IMG_DIR  = os.path.join(OUT_ROOT, "test",  "images")
OUT_TRAIN_CSV = os.path.join(OUT_ROOT, "train_labels.csv")
OUT_TEST_CSV  = os.path.join(OUT_ROOT, "test_labels.csv")

OUT_TRAIN_MAP = os.path.join(OUT_ROOT, "train_fracture_locations_kmeans.csv")
OUT_TEST_MAP  = os.path.join(OUT_ROOT, "test_fracture_locations_kmeans.csv")

KM_MODEL_PATH = "backend/kmeans_xnorm.joblib"
CENTERS_CSV = "backend/kmeans_centers.csv"
PLOT_PATH = "backend/debug_hist_xnorm.png"

NUM_AUG_PER_IMAGE = 3   # extra augmentations per image (plus original)
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# finger order left->right after sorting cluster centers
FINGER_ORDER = ["thumb","index","middle","ring","little"]
# we'll ask KMeans for 5 clusters (fingers). Wrist detection done by rule.
N_CLUSTERS = 5

# Albumentations pipeline (keep bbox_params = 'pascal_voc')
aug_pipeline = A.Compose([
    A.RandomRotate90(p=0.08),
    A.ShiftScaleRotate(shift_limit=0.06, scale_limit=0.08, rotate_limit=22, border_mode=cv2.BORDER_CONSTANT, p=0.6),
    A.HorizontalFlip(p=0.45),
    A.RandomBrightnessContrast(brightness_limit=0.22, contrast_limit=0.22, p=0.6),
    A.GaussNoise(var_limit=(8.0,45.0), p=0.35),
    A.MotionBlur(blur_limit=5, p=0.18),
], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['category_ids']))

# ---------- helpers ----------
def preprocess_no_rotate_cv2(img_bgr):
    """CLAHE + denoise (no rotation). Input/Output: BGR numpy arrays."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    g = clahe.apply(gray)
    g = cv2.fastNlMeansDenoising(g, None, h=10, templateWindowSize=7, searchWindowSize=21)
    proc = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
    return proc

def apply_aug_and_save(img_path, bboxes, out_dir, base_name, aug_index):
    """Apply preprocessing+augmentation and save image & bboxes. Returns saved_filename and list of bbox dicts."""
    img = cv2.imread(str(img_path))
    if img is None:
        return None, []
    img = preprocess_no_rotate_cv2(img)   # important: no rotation here
    h, w = img.shape[:2]

    alb_bboxes = []
    labels = []
    for bb in bboxes:
        try:
            xmin = float(bb['xmin']); ymin = float(bb['ymin'])
            xmax = float(bb['xmax']); ymax = float(bb['ymax'])
        except Exception:
            continue
        # clip
        xmin = max(0.0, min(xmin, w-1)); ymin = max(0.0, min(ymin, h-1))
        xmax = max(0.0, min(xmax, w-1)); ymax = max(0.0, min(ymax, h-1))
        if xmax <= xmin or ymax <= ymin:
            continue
        alb_bboxes.append([xmin, ymin, xmax, ymax])
        labels.append("fracture")

    if len(alb_bboxes) == 0:
        return None, []

    if aug_index == 0:
        saved_img = img
        saved_bboxes = alb_bboxes
    else:
        try:
            transformed = aug_pipeline(image=img, bboxes=alb_bboxes, category_ids=labels)
            saved_img = transformed['image']
            saved_bboxes = transformed['bboxes']
        except Exception as e:
            print("Augmentation failed for", img_path, "err:", e)
            saved_img = img
            saved_bboxes = alb_bboxes

    if len(saved_bboxes) == 0:
        return None, []

    ext = ".jpg"
    saved_name = f"{base_name}_aug{aug_index}{ext}"
    saved_path = os.path.join(out_dir, saved_name)
    cv2.imwrite(saved_path, saved_img)

    out_bbs = []
    for bb in saved_bboxes:
        xmin, ymin, xmax, ymax = bb
        xmin = int(max(0, min(xmin, saved_img.shape[1]-1)))
        ymin = int(max(0, min(ymin, saved_img.shape[0]-1)))
        xmax = int(max(0, min(xmax, saved_img.shape[1]-1)))
        ymax = int(max(0, min(ymax, saved_img.shape[0]-1)))
        if xmax <= xmin or ymax <= ymin:
            continue
        out_bbs.append({"filename": saved_name, "width": saved_img.shape[1], "height": saved_img.shape[0],
                        "xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax})
    return saved_name, out_bbs

def augment_split(src_csv, src_img_dir, out_img_dir, out_csv, num_aug_per_image=3):
    df = pd.read_csv(src_csv)
    rows_out = []
    grouped = df.groupby('filename')
    processed = 0
    for fname, group in grouped:
        src_path = os.path.join(src_img_dir, fname)
        if not os.path.exists(src_path):
            alt = os.path.basename(fname)
            alt_path = os.path.join(src_img_dir, alt)
            if os.path.exists(alt_path):
                src_path = alt_path
                fname = alt
            else:
                print("Missing image:", fname); continue

        bboxes = []
        for _, r in group.iterrows():
            try:
                bboxes.append({"xmin": float(r['xmin']), "ymin": float(r['ymin']),
                               "xmax": float(r['xmax']), "ymax": float(r['ymax'])})
            except Exception:
                continue
        base_name = Path(fname).stem
        # original (aug_index=0)
        saved_name, out_bbs = apply_aug_and_save(src_path, bboxes, out_img_dir, base_name, 0)
        if saved_name is not None:
            for b in out_bbs: rows_out.append(b)
            processed += 1
        # augment
        for ai in range(1, num_aug_per_image+1):
            saved_name, out_bbs = apply_aug_and_save(src_path, bboxes, out_img_dir, base_name, ai)
            if saved_name is None:
                continue
            for b in out_bbs: rows_out.append(b)
    out_df = pd.DataFrame(rows_out)
    out_df.to_csv(out_csv, index=False)
    print(f"Augmented {processed} source images -> saved {len(out_df)} bbox rows to {out_csv}")
    return out_df

# ---------- clustering & mapping ----------
def compute_centers_and_clusters(train_labels_csv, n_clusters=N_CLUSTERS):
    df = pd.read_csv(train_labels_csv)
    # compute x_center_norm, y_center_norm
    def safe_vals(r):
        try:
            W = float(r['width']); H = float(r['height'])
            xmin = float(r['xmin']); ymin = float(r['ymin']); xmax = float(r['xmax']); ymax = float(r['ymax'])
            if W<=0 or H<=0:
                return np.nan, np.nan
            xc = (xmin + xmax) / 2.0; yc = (ymin + ymax) / 2.0
            return float(xc / W), float(yc / H)
        except Exception:
            return np.nan, np.nan
    vals = df.apply(lambda r: pd.Series(safe_vals(r), index=['x_center_norm','y_center_norm']), axis=1)
    df[['x_center_norm','y_center_norm']] = vals
    df = df.dropna(subset=['x_center_norm']).reset_index(drop=True)
    X = df['x_center_norm'].values.reshape(-1,1)
    if len(X) < n_clusters:
        raise RuntimeError(f"Not enough valid samples ({len(X)}) for k={n_clusters}")
    k = KMeans(n_clusters=n_clusters, random_state=SEED, n_init=10)
    k.fit(X)
    joblib.dump(k, KM_MODEL_PATH)
    centers = k.cluster_centers_.reshape(-1)
    centers_df = pd.DataFrame({'cluster': np.arange(len(centers)), 'center': centers})
    centers_df = centers_df.sort_values('center').reset_index(drop=True)
    centers_df.to_csv(CENTERS_CSV, index=False)
    # histogram plot
    try:
        labels = k.predict(X)
        plt.figure(figsize=(8,4))
        for cl in np.unique(labels):
            plt.hist(df['x_center_norm'][labels==cl], bins=40, alpha=0.6, label=f"c{cl}")
        plt.legend(); plt.title("x_center_norm by kmeans cluster"); plt.xlabel("x_center_norm")
        plt.savefig(PLOT_PATH); plt.close()
    except Exception:
        pass
    return k, df

def map_using_kmeans(label_csv, kmodel, out_map_csv):
    df = pd.read_csv(label_csv)
    # compute centers
    def safe_vals(r):
        try:
            W = float(r['width']); H = float(r['height'])
            xmin = float(r['xmin']); ymin = float(r['ymin']); xmax = float(r['xmax']); ymax = float(r['ymax'])
            if W<=0 or H<=0:
                return np.nan, np.nan, np.nan
            xc = (xmin + xmax) / 2.0; yc = (ymin + ymax) / 2.0
            w_box = (xmax - xmin) / W
            h_box = (ymax - ymin) / H
            return float(xc / W), float(yc / H), float(h_box)
        except Exception:
            return np.nan, np.nan, np.nan
    vals = df.apply(lambda r: pd.Series(safe_vals(r), index=['x_center_norm','y_center_norm','box_h_norm']), axis=1)
    df[['x_center_norm','y_center_norm','box_h_norm']] = vals

    valid_mask = ~df['x_center_norm'].isna()
    if valid_mask.sum() > 0:
        preds = kmodel.predict(df.loc[valid_mask,'x_center_norm'].values.reshape(-1,1))
        # map cluster -> finger by sorted center order
        centers = kmodel.cluster_centers_.reshape(-1)
        order = np.argsort(centers)
        cl_to_finger = {}
        for i, cl in enumerate(order):
            if i < len(FINGER_ORDER):
                cl_to_finger[cl] = FINGER_ORDER[i]
            else:
                cl_to_finger[cl] = f"c{cl}"
        # assign
        fingers = np.array(["unknown"]*len(df))
        fingers[valid_mask.values] = [cl_to_finger[c] for c in preds]
        df['finger'] = fingers
    else:
        df['finger'] = "unknown"

    # detect wrist/whole-hand using heuristic:
    # if box height covers large fraction of image (box_h_norm > 0.55) OR y_center_norm > 0.88 -> wrist/hand
    def wrist_rule(r):
        try:
            if np.isnan(r['box_h_norm']) or np.isnan(r['y_center_norm']):
                return False
            if r['box_h_norm'] > 0.55:
                return True
            if r['y_center_norm'] > 0.88:
                return True
            return False
        except Exception:
            return False

    notes = []
    for idx, row in df.iterrows():
        note = ""
        if wrist_rule(row):
            df.at[idx,'finger'] = "wrist"
            note = "wrist_by_rule"
        # region mapping (top->distal, bottom->proximal)
        y = row['y_center_norm']
        if np.isnan(y):
            region = "unknown"
        else:
            if y < 0.33:
                region = "distal"
            elif y < 0.66:
                region = "middle"
            else:
                region = "proximal"
        df.at[idx,'region'] = region
        df.at[idx,'note'] = note
    out = df[['filename','region','finger','x_center_norm','y_center_norm','note']].copy()
    out.to_csv(out_map_csv, index=False)
    print("Saved mapping:", out_map_csv, "rows:", len(out))
    return out

# ---------- main ----------
def main():
    # create dirs
    Path(OUT_TRAIN_IMG_DIR).mkdir(parents=True, exist_ok=True)
    Path(OUT_TEST_IMG_DIR).mkdir(parents=True, exist_ok=True)

    print("Augmenting TRAIN...")
    train_out_df = augment_split(SRC_TRAIN_CSV, SRC_TRAIN_IMG_DIR, OUT_TRAIN_IMG_DIR, OUT_TRAIN_CSV, NUM_AUG_PER_IMAGE)
    print("Augmenting TEST...")
    test_out_df = augment_split(SRC_TEST_CSV, SRC_TEST_IMG_DIR, OUT_TEST_IMG_DIR, OUT_TEST_CSV, NUM_AUG_PER_IMAGE)

    print("Running KMeans on augmented train x_center_norm ...")
    kmodel, df_with_x = compute_centers_and_clusters(OUT_TRAIN_CSV, n_clusters=N_CLUSTERS)
    print("KMeans centers (unsorted):", kmodel.cluster_centers_.reshape(-1))
    print("Saved kmeans model ->", KM_MODEL_PATH)

    # create mapped CSVs using kmeans + wrist rule
    map_using_kmeans(OUT_TRAIN_CSV, kmodel, OUT_TRAIN_MAP)
    map_using_kmeans(OUT_TEST_CSV, kmodel, OUT_TEST_MAP)
    print("All done. Augmented images in:", OUT_ROOT)
    print("Train mapping:", OUT_TRAIN_MAP)
    print("Test mapping:", OUT_TEST_MAP)
    print("Centers CSV:", CENTERS_CSV)
    print("Histogram:", PLOT_PATH)

if __name__ == "__main__":
    main()
