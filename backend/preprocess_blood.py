# backend/preprocess_short.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os
from pathlib import Path

SRC = Path("data/blood_data.csv")   # uploaded CSV path (already on the machine)
OUT = Path("data")
OUT.mkdir(parents=True, exist_ok=True)

if not SRC.exists():
    raise SystemExit(f"Input CSV not found at {SRC}. Place your file there.")

# 1) Load CSV
df = pd.read_csv(SRC)
print("Loaded", SRC, "shape:", df.shape)

# 2) Clean CLASS -> diabetes labels (N->0, P->1, Y->2) and binary
if "CLASS" in df.columns:
    cls = df["CLASS"].astype(str).str.strip().str.upper()
    mapping = {}
    unique = cls.dropna().unique().tolist()
    # simple mapping rules
    for v in unique:
        s = str(v).strip().upper()
        if s in ("N","NORMAL","NO"):
            mapping[v] = 0
        elif s in ("P","PRE","PRED","PREDIABETES","PREDIABETIC"):
            mapping[v] = 1
        elif s in ("Y","YES","DIABETES","DIABETIC"):
            mapping[v] = 2
        else:
            # fallback heuristics
            if "P" in s and len(s) <= 3:
                mapping[v] = 1
            elif "Y" in s or "D" in s:
                mapping[v] = 2
            else:
                mapping[v] = 0
    df["diabetes_class"] = cls.map(mapping).fillna(0).astype(int)
    df["diabetes_binary"] = df["diabetes_class"].apply(lambda x: 1 if x >= 1 else 0)
    print("Created diabetes_class and diabetes_binary from CLASS.")
else:
    # fallback: derive from HbA1c >=5.7 or Glucose >=100
    print("CLASS not found — deriving diabetes_binary from HbA1c/Glucose thresholds.")
    a1c = pd.to_numeric(df.get("HbA1c", pd.Series([np.nan]*len(df))), errors="coerce")
    glu = pd.to_numeric(df.get("Glucose", pd.Series([np.nan]*len(df))), errors="coerce")
    df["diabetes_binary"] = ((a1c >= 5.7) | (glu >= 100)).astype(int).fillna(0)

# 3) Gender -> numeric
if "Gender" in df.columns:
    g = df["Gender"].astype(str).str.strip().str.upper()
    g = g.replace({"MALE":"M","FEMALE":"F"})
    df["Gender_num"] = g.map({"M":1, "F":0})
else:
    df["Gender_num"] = 0

# 4) Build feature set (select safe columns if exist)
feature_candidates = ["AGE", "Gender_num", "HbA1c", "Glucose", "Chol", "TG", "HDL", "LDL", "VLDL", "BMI", "CRP"]
# keep columns present (case-sensitive check)
features = [c for c in feature_candidates if c in df.columns]
print("Using features:", features)

X = df[features].copy()

# 5) Chol unit handling: if Chol median > 50 assume mg/dL and convert to mmol/L
if "Chol" in X.columns:
    chol_numeric = pd.to_numeric(X["Chol"], errors="coerce")
    med = chol_numeric.median(skipna=True)
    if pd.notna(med) and med > 50:
        print("Converting Chol mg/dL -> mmol/L (divide by 38.67). Median before:", med)
        X["Chol"] = chol_numeric / 38.67
    else:
        X["Chol"] = chol_numeric

# 6) Fill missing numeric with median
X = X.apply(pd.to_numeric, errors="coerce")
X = X.fillna(X.median())

# Create high_chol target using threshold 5.2 mmol/L
if "Chol" in X.columns:
    y_chol = (X["Chol"] > 5.2).astype(int)
else:
    y_chol = pd.Series([0]*len(X), index=X.index)

# diabetes binary target
y_diabetes = df["diabetes_binary"].astype(int)

# Save processed feature/full files
df.to_csv(OUT / "blood_processed_full.csv", index=False)
X.to_csv(OUT / "blood_processed_features.csv", index=False)
print("Saved processed files to", OUT)

# Train/test splits for both tasks (stratify when possible)
def save_splits(X, y, prefix):
    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    except Exception:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train.to_csv(OUT / f"X_train_{prefix}.csv", index=False)
    X_test.to_csv(OUT / f"X_test_{prefix}.csv", index=False)
    y_train.to_csv(OUT / f"y_train_{prefix}.csv", index=False)
    y_test.to_csv(OUT / f"y_test_{prefix}.csv", index=False)
    print(f"Saved splits for {prefix}: X_train_{prefix}.csv, X_test_{prefix}.csv, y_train_{prefix}.csv, y_test_{prefix}.csv")

save_splits(X, y_diabetes, "diabetes")
save_splits(X, y_chol, "chol")

print("Preprocessing (short) completed successfully.")
