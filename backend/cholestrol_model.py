# backend/train_chol_simple.py
import os
import pandas as pd
import numpy as np

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, roc_curve, confusion_matrix
)

import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras import layers, models, callbacks

# ensure folders
os.makedirs("models", exist_ok=True)
os.makedirs("outputs/figures", exist_ok=True)

# ---------------- LOAD PREPROCESSED DATA ----------------
X_train = pd.read_csv("data/X_train_chol.csv")
X_test = pd.read_csv("data/X_test_chol.csv")
y_train = pd.read_csv("data/y_train_chol.csv").iloc[:, 0]
y_test = pd.read_csv("data/y_test_chol.csv").iloc[:, 0]

print("Training data:", X_train.shape, "Test data:", X_test.shape)

# ---------------- ANN MODEL ----------------
input_dim = X_train.shape[1]

model = models.Sequential([
    layers.Input(shape=(input_dim,)),
    layers.Dense(32, activation="relu"),
    layers.Dropout(0.15),
    layers.Dense(16, activation="relu"),
    layers.Dense(1, activation="sigmoid")
])

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

EPOCHS = 6
BATCH_SIZE = 20

es = callbacks.EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True)

history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[es],
    verbose=2
)

model.save("models/ann_chol.h5")
print("Saved: models/ann_chol.h5")

# ---------------- METRICS ----------------
y_prob = model.predict(X_test).ravel()
y_pred = (y_prob >= 0.5).astype(int)

def metrics_report(y_true, y_prob, y_pred):
    return {
        "auc": float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true))>1 else float("nan"),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0))
    }

metrics = metrics_report(y_test, y_prob, y_pred)
pd.DataFrame([metrics]).to_csv("outputs/metrics_chol.csv", index=False)
print("Saved metrics -> outputs/metrics_chol.csv")
print(metrics)

# ---------------- ROC CURVE ----------------
fpr, tpr, _ = roc_curve(y_test, y_prob)
plt.figure(figsize=(6,6))
plt.plot(fpr, tpr, label=f"ANN Chol (AUC={metrics['auc']:.3f})")
plt.plot([0,1],[0,1],'--',color='gray')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC - High Cholesterol")
plt.legend()
plt.savefig("outputs/figures/roc_chol.png")
plt.close()
print("Saved ROC -> outputs/figures/roc_chol.png")

# ---------------- CONFUSION MATRIX ----------------
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(4,3))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix - Chol ANN")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig("outputs/figures/cm_chol_ann.png")
plt.close()
print("Saved confusion matrix -> outputs/figures/cm_chol_ann.png")

# ---------------- TRAINING HISTORY (CSV + PLOTS) ----------------
hist_df = pd.DataFrame(history.history)
hist_csv_path = "outputs/figures/history_chol.csv"
hist_df.to_csv(hist_csv_path, index=False)
print("Saved training history CSV ->", hist_csv_path)

# Loss plot
plt.figure(figsize=(6,4))
if "loss" in history.history:
    plt.plot(history.history["loss"], label="train_loss")
if "val_loss" in history.history:
    plt.plot(history.history["val_loss"], label="val_loss")
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.savefig("outputs/figures/loss_chol.png")
plt.close()
print("Saved loss plot -> outputs/figures/loss_chol.png")

# Accuracy plot
plt.figure(figsize=(6,4))
if "accuracy" in history.history:
    plt.plot(history.history["accuracy"], label="train_acc")
if "val_accuracy" in history.history:
    plt.plot(history.history["val_accuracy"], label="val_acc")
plt.title("Training Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.savefig("outputs/figures/acc_chol.png")
plt.close()
print("Saved accuracy plot -> outputs/figures/acc_chol.png")

print("All files saved in outputs/figures & outputs/metrics_chol.csv")
