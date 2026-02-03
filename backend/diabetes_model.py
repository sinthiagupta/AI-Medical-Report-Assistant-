# backend/train_ann_final.py
"""
Train a small ANN for the diabetes_binary target.
Safe defaults: EPOCHS=6, BATCH_SIZE=20 (as requested).
Saves model -> models/ann_blood.h5
Saves metrics -> outputs/metrics_ann.csv
Saves ROC and confusion-matrix images -> outputs/figures/
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path

# sklearn metrics
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix
)

# plotting
import matplotlib.pyplot as plt
import seaborn as sns

# tensorflow / keras
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

# ensure folders
os.makedirs("models", exist_ok=True)
os.makedirs("outputs/figures", exist_ok=True)

# --- CONFIG (safe for your laptop) ---
EPOCHS = 6
BATCH_SIZE = 20
VERBOSE = 2
RANDOM_SEED = 42

tf.random.set_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# --- LOAD DATA (these were created by preprocessing) ---
X_train = pd.read_csv("data/X_train_diabetes.csv")
X_test  = pd.read_csv("data/X_test_diabetes.csv")
y_train = pd.read_csv("data/y_train_diabetes.csv").iloc[:, 0]
y_test  = pd.read_csv("data/y_test_diabetes.csv").iloc[:, 0]

print("X_train:", X_train.shape, "X_test:", X_test.shape, "y_train:", y_train.shape)

# --- Build a small ANN (2 hidden layers) ---
input_dim = X_train.shape[1]

model = models.Sequential([
    layers.Input(shape=(input_dim,)),
    layers.Dense(32, activation="relu"),
    layers.Dropout(0.15),
    layers.Dense(16, activation="relu"),
    layers.Dense(1, activation="sigmoid")
])

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model.summary()

# --- Callbacks: EarlyStopping to avoid overtraining ---
es = callbacks.EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True, verbose=1)

# --- Train ---
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[es],
    verbose=VERBOSE
)

# --- Save model ---
model_path = "models/ann_blood.h5"
model.save(model_path)
print(f"Saved ANN model to {model_path}")

# --- Predictions on test set ---
y_prob = model.predict(X_test).ravel()
y_pred = (y_prob >= 0.5).astype(int)

# --- Metrics ---
def compute_metrics(y_true, probs, preds):
    out = {}
    out["auc"] = float(roc_auc_score(y_true, probs)) if len(np.unique(y_true))>1 else float("nan")
    out["accuracy"] = float(accuracy_score(y_true, preds))
    out["precision"] = float(precision_score(y_true, preds, zero_division=0))
    out["recall"] = float(recall_score(y_true, preds, zero_division=0))
    out["f1"] = float(f1_score(y_true, preds, zero_division=0))
    return out

metrics = compute_metrics(y_test, y_prob, y_pred)
metrics["model"] = "ann_blood"
print("Metrics:", metrics)

# Save metrics DataFrame
pd.DataFrame([metrics]).to_csv("outputs/metrics_ann.csv", index=False)
print("Saved metrics to outputs/metrics_ann.csv")

# --- ROC curve ---
fpr, tpr, _ = roc_curve(y_test, y_prob)
plt.figure(figsize=(6,6))
plt.plot(fpr, tpr, label=f"ANN (AUC={metrics['auc']:.3f})")
plt.plot([0,1],[0,1],"--", color="gray")
plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
plt.title("ROC - ANN (diabetes)")
plt.legend()
plt.savefig("outputs/figures/roc_ann.png")
plt.close()
print("Saved ROC to outputs/figures/roc_ann.png")

# --- Confusion Matrix ---
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(4,3))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix - ANN")
plt.xlabel("Predicted"); plt.ylabel("Actual")
plt.savefig("outputs/figures/cm_ann.png")
plt.close()
print("Saved CM to outputs/figures/cm_ann.png")

# --- Also save training history plot (loss + accuracy) ---
hist_df = pd.DataFrame(history.history)
hist_df.to_csv("outputs/figures/ann_history.csv", index=False)

plt.figure(figsize=(6,4))
plt.plot(history.history.get("loss", []), label="train_loss")
plt.plot(history.history.get("val_loss", []), label="val_loss")
plt.title("Loss")
plt.xlabel("epoch"); plt.ylabel("loss"); plt.legend()
plt.savefig("outputs/figures/loss_ann.png")
plt.close()

plt.figure(figsize=(6,4))
plt.plot(history.history.get("accuracy", []), label="train_acc")
plt.plot(history.history.get("val_accuracy", []), label="val_acc")
plt.title("Accuracy")
plt.xlabel("epoch"); plt.ylabel("accuracy"); plt.legend()
plt.savefig("outputs/figures/acc_ann.png")
plt.close()

print("Saved training history plots to outputs/figures/")

print("Training complete. Files created:")
print("- models/ann_blood.h5")
print("- outputs/metrics_ann.csv")
print("- outputs/figures/roc_ann.png")
print("- outputs/figures/cm_ann.png")
print("- outputs/figures/loss_ann.png")
print("- outputs/figures/acc_ann.png")
