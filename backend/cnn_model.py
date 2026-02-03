# backend/cnn_model.py
"""
Robust small multi-head CNN training script for x-ray fracture region+finger detection.
Features:
 - supports 'wrist' as an extra finger class
 - dynamic model outputs (uses NUM_REGION / NUM_FINGER)
 - class weighting computed from training CSV
 - gradient accumulation (ACCUM_STEPS)
 - robust concatenation of validation probs to avoid zero-dim errors
 - confusion matrix forced to full NxN using labels=range(NUM_FINGER)
 - classification_report uses zero_division=0
Run:
    python backend/cnn_model.py
"""

import os
import random
from pathlib import Path
import numpy as np
import pandas as pd
from collections import Counter

from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.preprocessing import label_binarize

# ----------------- USER PATHS (edit if needed) -----------------
TRAIN_CSV = "data/xray_aug/train_fracture_locations_kmeans.csv"
TEST_CSV  = "data/xray_aug/test_fracture_locations_kmeans.csv"
TRAIN_IMG_DIR = "data/xray_aug/train/images"
TEST_IMG_DIR  = "data/xray_aug/test/images"

UPLOADED_ZIP = "/mnt/data/archive (17).zip"
# --------------------------------------------------------------

# ---------- label maps ----------
REGION_TO_IDX = {"distal":0, "middle":1, "proximal":2}
IDX_TO_REGION = {v:k for k,v in REGION_TO_IDX.items()}

# include wrist class
FINGER_TO_IDX = {"thumb":0, "index":1, "middle":2, "ring":3, "little":4, "wrist":5}
IDX_TO_FINGER = {v:k for k,v in FINGER_TO_IDX.items()}

NUM_REGION = len(REGION_TO_IDX)
NUM_FINGER = len(FINGER_TO_IDX)

# ---------- hyperparams ----------
IMG_SIZE = 128
BATCH_SIZE = 4         # adjust for your RAM
ACCUM_STEPS = 4        # accumulate gradients to simulate larger batch
EPOCHS = 20
LR = 1e-3
NUM_WORKERS = 0
SEED = 42

# reproducibility seeds
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("DEVICE:", DEVICE)
print("Train CSV:", TRAIN_CSV)
print("Test  CSV:", TEST_CSV)
print("NUM_REGION:", NUM_REGION, "NUM_FINGER:", NUM_FINGER)

# --------- Dataset ----------
class MultiTaskDS(Dataset):
    def __init__(self, csv_path, img_dir, transform=None):
        self.df = pd.read_csv(csv_path)
        self.img_dir = Path(img_dir)
        self.transform = transform
        # filter missing images
        self.df = self.df[self.df['filename'].apply(lambda f: (self.img_dir / f).exists())].reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        r = self.df.iloc[idx]
        p = self.img_dir / r['filename']
        img = Image.open(p).convert("RGB")
        if self.transform:
            img = self.transform(img)
        # safe mapping (defaults)
        region = REGION_TO_IDX.get(r.get('region', None), 0)
        finger = FINGER_TO_IDX.get(r.get('finger', None), FINGER_TO_IDX['wrist'])
        return img, int(region), int(finger)

tf_train = transforms.Compose([transforms.Resize((IMG_SIZE,IMG_SIZE)), transforms.RandomHorizontalFlip(), transforms.ToTensor()])
tf_eval  = transforms.Compose([transforms.Resize((IMG_SIZE,IMG_SIZE)), transforms.ToTensor()])

train_ds = MultiTaskDS(TRAIN_CSV, TRAIN_IMG_DIR, transform=tf_train)
test_ds  = MultiTaskDS(TEST_CSV,  TEST_IMG_DIR,  transform=tf_eval)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=False)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=False)

print("Train samples:", len(train_ds))
print("Test  samples:", len(test_ds))
steps_per_epoch = max(1, int(np.ceil(len(train_ds) / BATCH_SIZE)))
print("Steps/epoch (approx):", steps_per_epoch)

# --------- Small multi-head model ----------
class SmallMultiHead(nn.Module):
    def __init__(self, img_size=IMG_SIZE):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3,16,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16,32,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        feat_size = 64 * (img_size//8) * (img_size//8)
        self.fc_shared = nn.Sequential(nn.Flatten(), nn.Linear(feat_size,128), nn.ReLU(), nn.Dropout(0.3))
        self.region_head = nn.Linear(128, NUM_REGION)
        self.finger_head = nn.Linear(128, NUM_FINGER)

    def forward(self, x):
        f = self.features(x)
        s = self.fc_shared(f)
        return self.region_head(s), self.finger_head(s)

model = SmallMultiHead().to(DEVICE)

# --------- compute class weights robustly ----------
train_df = pd.read_csv(TRAIN_CSV)
r_counter = Counter(train_df['region'].fillna("unknown").values)
f_counter = Counter(train_df['finger'].fillna("wrist").values)

r_counts_list = [ r_counter.get(k, 0) for k in REGION_TO_IDX.keys() ]
f_counts_list = [ f_counter.get(k, 0) for k in FINGER_TO_IDX.keys() ]

eps = 1e-6
r_weights = torch.tensor([1.0/(c + eps) for c in r_counts_list], dtype=torch.float32)
f_weights = torch.tensor([1.0/(c + eps) for c in f_counts_list], dtype=torch.float32)

optimizer = optim.Adam(model.parameters(), lr=LR)
criterion_region = None
criterion_finger = None

# --------- training loop ----------
history = {"train_loss":[], "val_loss":[], "train_region_acc":[], "val_region_acc":[], "train_finger_acc":[], "val_finger_acc":[]}
best_val_loss = float('inf')

# storage for final-epoch eval
all_r_probs = []
all_r_true = []
all_f_probs = []
all_f_true = []

_first_batch_checked = False

for epoch in range(1, EPOCHS+1):
    model.train()
    total_loss = 0.0
    r_correct = 0; f_correct = 0; total = 0

    if criterion_region is None:
        criterion_region = nn.CrossEntropyLoss(weight=r_weights.to(DEVICE))
        criterion_finger = nn.CrossEntropyLoss(weight=f_weights.to(DEVICE))

    optimizer.zero_grad()
    for step, (imgs, r_labels, f_labels) in enumerate(train_loader):
        imgs = imgs.to(DEVICE); r_labels = r_labels.to(DEVICE); f_labels = f_labels.to(DEVICE)

        r_out, f_out = model(imgs)

        if not _first_batch_checked:
            print("Sanity shapes (train first batch): r_out", tuple(r_out.shape), "f_out", tuple(f_out.shape))
            _first_batch_checked = True

        loss = criterion_region(r_out, r_labels) + criterion_finger(f_out, f_labels)
        loss = loss / ACCUM_STEPS
        loss.backward()

        if (step + 1) % ACCUM_STEPS == 0:
            optimizer.step()
            optimizer.zero_grad()

        total_loss += (loss.item() * imgs.size(0) * ACCUM_STEPS)
        preds_r = r_out.argmax(dim=1); preds_f = f_out.argmax(dim=1)
        r_correct += (preds_r == r_labels).sum().item()
        f_correct += (preds_f == f_labels).sum().item()
        total += imgs.size(0)

    train_loss = total_loss / total if total>0 else 0.0
    train_r_acc = r_correct / total if total>0 else 0.0
    train_f_acc = f_correct / total if total>0 else 0.0

    # validation
    model.eval()
    v_loss = 0.0
    vr_correct = 0; vf_correct = 0; v_total = 0

    epoch_r_probs = []; epoch_r_true = []
    epoch_f_probs = []; epoch_f_true = []

    with torch.no_grad():
        for imgs, r_labels, f_labels in test_loader:
            imgs = imgs.to(DEVICE); r_labels = r_labels.to(DEVICE); f_labels = f_labels.to(DEVICE)
            r_out, f_out = model(imgs)
            loss = criterion_region(r_out, r_labels) + criterion_finger(f_out, f_labels)
            v_loss += loss.item() * imgs.size(0)
            v_total += imgs.size(0)
            preds_r = r_out.argmax(dim=1); preds_f = f_out.argmax(dim=1)
            vr_correct += (preds_r == r_labels).sum().item()
            vf_correct += (preds_f == f_labels).sum().item()
            probs_r = torch.softmax(r_out, dim=1).cpu().numpy()
            probs_f = torch.softmax(f_out, dim=1).cpu().numpy()
            epoch_r_probs.append(probs_r); epoch_r_true.append(r_labels.cpu().numpy())
            epoch_f_probs.append(probs_f); epoch_f_true.append(f_labels.cpu().numpy())

    val_loss = v_loss / v_total if v_total>0 else 0.0
    val_r_acc = vr_correct / v_total if v_total>0 else 0.0
    val_f_acc = vf_correct / v_total if v_total>0 else 0.0

    history["train_loss"].append(train_loss); history["val_loss"].append(val_loss)
    history["train_region_acc"].append(train_r_acc); history["val_region_acc"].append(val_r_acc)
    history["train_finger_acc"].append(train_f_acc); history["val_finger_acc"].append(val_f_acc)

    # safe collate helpers
    def safe_vstack(list_of_arrays, expected_cols):
        good = []
        for a in list_of_arrays:
            if a is None: continue
            arr = np.asarray(a)
            if arr.size == 0: continue
            if arr.ndim == 1: arr = arr.reshape(1, -1)
            if arr.ndim >= 2 and arr.shape[1] == expected_cols:
                good.append(arr)
        if len(good) == 0:
            return np.zeros((0, expected_cols))
        return np.vstack(good)

    def safe_concat(list_of_arrays):
        good = []
        for a in list_of_arrays:
            if a is None: continue
            arr = np.asarray(a)
            if arr.size == 0: continue
            good.append(arr.reshape(-1))
        if len(good) == 0:
            return np.array([])
        return np.concatenate(good)

    all_r_probs = safe_vstack(epoch_r_probs, expected_cols=NUM_REGION)
    all_r_true  = safe_concat(epoch_r_true)
    all_f_probs = safe_vstack(epoch_f_probs, expected_cols=NUM_FINGER)
    all_f_true  = safe_concat(epoch_f_true)

    print(f"Epoch {epoch}/{EPOCHS}  loss {train_loss:.4f}/{val_loss:.4f} | region acc {train_r_acc:.3f}/{val_r_acc:.3f} | finger acc {train_f_acc:.3f}/{val_f_acc:.3f}")

    # save best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        os.makedirs("results_multi_small", exist_ok=True)
        torch.save(model.state_dict(), os.path.join("results_multi_small","best_multihead.pth"))
        print("Saved best model (best_multihead.pth) with val_loss:", best_val_loss)

# final saves
os.makedirs("results_multi_small", exist_ok=True)
pd.DataFrame(history).to_csv(os.path.join("results_multi_small","history.csv"), index=False)
torch.save(model.state_dict(), os.path.join("results_multi_small","multihead_small_last.pth"))

# --------- Evaluate & Save confusion/ROC (robust) ----------
import matplotlib.pyplot as plt
import seaborn as sns

# Finger confusion: force full label set so we always get NUM_FINGER x NUM_FINGER
if all_f_true.size > 0 and all_f_probs.shape[0] > 0:
    f_pred_labels = np.argmax(all_f_probs, axis=1)
    full_labels = list(range(NUM_FINGER))
    cm_f = confusion_matrix(all_f_true, f_pred_labels, labels=full_labels)

    # save full NxN confusion
    pd.DataFrame(cm_f,
                 index=[IDX_TO_FINGER[i] for i in full_labels],
                 columns=[IDX_TO_FINGER[i] for i in full_labels]).to_csv(
                 os.path.join("results_multi_small","confusion_finger.csv"))

    plt.figure(figsize=(8,6))
    sns.heatmap(cm_f, annot=True, fmt="d", xticklabels=[IDX_TO_FINGER[i] for i in full_labels], yticklabels=[IDX_TO_FINGER[i] for i in full_labels])
    plt.title("Finger Confusion"); plt.savefig(os.path.join("results_multi_small","confusion_finger.png")); plt.close()

    with open(os.path.join("results_multi_small","report_finger.txt"), "w") as f:
        f.write(classification_report(all_f_true, f_pred_labels, labels=full_labels, target_names=[IDX_TO_FINGER[i] for i in full_labels], zero_division=0))

# Region confusion
if all_r_true.size > 0 and all_r_probs.shape[0] > 0:
    r_pred_labels = np.argmax(all_r_probs, axis=1)
    full_r_labels = list(range(NUM_REGION))
    cm_r = confusion_matrix(all_r_true, r_pred_labels, labels=full_r_labels)
    pd.DataFrame(cm_r, index=[IDX_TO_REGION[i] for i in full_r_labels], columns=[IDX_TO_REGION[i] for i in full_r_labels]).to_csv(os.path.join("results_multi_small","confusion_region.csv"))
    plt.figure(figsize=(6,5)); sns.heatmap(cm_r, annot=True, fmt="d", xticklabels=[IDX_TO_REGION[i] for i in full_r_labels], yticklabels=[IDX_TO_REGION[i] for i in full_r_labels]); plt.title("Region Confusion"); plt.savefig(os.path.join("results_multi_small","confusion_region.png")); plt.close()
    with open(os.path.join("results_multi_small","report_region.txt"), "w") as f:
        f.write(classification_report(all_r_true, r_pred_labels, labels=full_r_labels, target_names=[IDX_TO_REGION[i] for i in full_r_labels], zero_division=0))

# ROC plotting (one-vs-rest)
def plot_multiclass_roc(y_true, y_score, n_classes, out_path, class_names):
    if y_true.size == 0 or y_score.shape[0] == 0:
        return
    y_bin = label_binarize(y_true, classes=list(range(n_classes)))
    plt.figure()
    for i in range(n_classes):
        try:
            fpr, tpr, _ = roc_curve(y_bin[:,i], y_score[:,i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f"{class_names[i]} (AUC={roc_auc:.2f})")
        except Exception:
            continue
    plt.plot([0,1],[0,1],'k--'); plt.xlabel("FPR"); plt.ylabel("TPR"); plt.legend(); plt.title("ROC Curves"); plt.savefig(out_path); plt.close()

plot_multiclass_roc(all_r_true, all_r_probs, NUM_REGION, os.path.join("results_multi_small","roc_region.png"), [IDX_TO_REGION[i] for i in range(NUM_REGION)])
plot_multiclass_roc(all_f_true, all_f_probs, NUM_FINGER, os.path.join("results_multi_small","roc_finger.png"), [IDX_TO_FINGER[i] for i in range(NUM_FINGER)])

# Loss/accuracy plots
plt.figure(); plt.plot(history["train_loss"], label="train_loss"); plt.plot(history["val_loss"], label="val_loss"); plt.legend(); plt.title("Loss"); plt.savefig(os.path.join("results_multi_small","loss_plot.png")); plt.close()
plt.figure(); plt.plot(history["train_region_acc"], label="train_region_acc"); plt.plot(history["val_region_acc"], label="val_region_acc"); plt.legend(); plt.title("Region Acc"); plt.savefig(os.path.join("results_multi_small","region_acc.png")); plt.close()
plt.figure(); plt.plot(history["train_finger_acc"], label="train_finger_acc"); plt.plot(history["val_finger_acc"], label="val_finger_acc"); plt.legend(); plt.title("Finger Acc"); plt.savefig(os.path.join("results_multi_small","finger_acc.png")); plt.close()

print("All outputs saved to ./results_multi_small/ (model, history, confusion, ROC, plots)")
print("Reference uploaded zip path:", UPLOADED_ZIP)
