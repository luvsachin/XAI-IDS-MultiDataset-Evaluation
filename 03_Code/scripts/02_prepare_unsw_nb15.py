from pathlib import Path
import gc
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "02_Data" / "raw" / "UNSW-NB15"
OUT = ROOT / "02_Data" / "processed" / "UNSW-NB15"
OUT.mkdir(parents=True, exist_ok=True)

train_path = RAW / "UNSW_NB15_training-set.csv"
test_path = RAW / "UNSW_NB15_testing-set.csv"

def save_csv_memory_safe(df, path):
    df.to_csv(path, index=False, chunksize=5000)

print("Loading UNSW-NB15...")
train_df = pd.read_csv(train_path, low_memory=False)
test_df = pd.read_csv(test_path, low_memory=False)

print("Train shape:", train_df.shape)
print("Test shape:", test_df.shape)

y_train_full = train_df["label"].astype(np.int8)
y_test = test_df["label"].astype(np.int8)

drop_cols = ["id", "attack_cat", "label"]
X_train_full = train_df.drop(columns=[c for c in drop_cols if c in train_df.columns])
X_test = test_df.drop(columns=[c for c in drop_cols if c in test_df.columns])

del train_df, test_df
gc.collect()

cat_cols = X_train_full.select_dtypes(include=["object", "category"]).columns.tolist()
print("Categorical columns:", cat_cols)

X_train_full = pd.get_dummies(X_train_full, columns=cat_cols, drop_first=False, dtype=np.uint8)
X_test = pd.get_dummies(X_test, columns=cat_cols, drop_first=False, dtype=np.uint8)

X_train_full, X_test = X_train_full.align(X_test, join="left", axis=1, fill_value=0)

print("Encoded train shape:", X_train_full.shape)
print("Encoded test shape:", X_test.shape)

X_train, X_val, y_train, y_val = train_test_split(
    X_train_full,
    y_train_full,
    test_size=0.20,
    random_state=42,
    stratify=y_train_full
)

del X_train_full, y_train_full
gc.collect()

scaler = StandardScaler()

print("Scaling train...")
X_train_scaled = scaler.fit_transform(X_train).astype(np.float32)
columns = X_train.columns.tolist()
del X_train
gc.collect()

print("Scaling validation...")
X_val_scaled = scaler.transform(X_val).astype(np.float32)
del X_val
gc.collect()

print("Scaling test...")
X_test_scaled = scaler.transform(X_test).astype(np.float32)
del X_test
gc.collect()

print("Saving processed files...")

save_csv_memory_safe(
    pd.DataFrame(X_train_scaled, columns=columns),
    OUT / "X_train_final.csv"
)
del X_train_scaled
gc.collect()

save_csv_memory_safe(
    pd.DataFrame(X_val_scaled, columns=columns),
    OUT / "X_val_final.csv"
)
del X_val_scaled
gc.collect()

save_csv_memory_safe(
    pd.DataFrame(X_test_scaled, columns=columns),
    OUT / "X_test_final.csv"
)
del X_test_scaled
gc.collect()

pd.DataFrame({"label": y_train.astype(np.int8)}).to_csv(OUT / "y_train_binary.csv", index=False)
pd.DataFrame({"label": y_val.astype(np.int8)}).to_csv(OUT / "y_val_binary.csv", index=False)
pd.DataFrame({"label": y_test.astype(np.int8)}).to_csv(OUT / "y_test_binary.csv", index=False)

joblib.dump(scaler, OUT / "scaler_unsw_nb15.joblib")

summary = pd.DataFrame([
    {"split": "train", "rows": len(y_train), "columns": len(columns), "attack_ratio": float(y_train.mean())},
    {"split": "val", "rows": len(y_val), "columns": len(columns), "attack_ratio": float(y_val.mean())},
    {"split": "test", "rows": len(y_test), "columns": len(columns), "attack_ratio": float(y_test.mean())},
])
summary.to_csv(OUT / "preprocessing_summary.csv", index=False)

print("\nUNSW-NB15 preprocessing complete.")
print(summary.to_string(index=False))
print("Saved to:", OUT)