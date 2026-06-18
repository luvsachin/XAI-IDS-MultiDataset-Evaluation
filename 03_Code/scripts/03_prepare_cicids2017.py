from pathlib import Path
import argparse
import gc
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

parser = argparse.ArgumentParser()
parser.add_argument("--max-rows", type=int, default=600000)
parser.add_argument("--chunk-size", type=int, default=25000)
parser.add_argument("--max-missing-col-ratio", type=float, default=0.50)
args = parser.parse_args()

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "02_Data" / "raw" / "CICIDS2017"
OUT = ROOT / "02_Data" / "processed" / "CICIDS2017"
METRICS = ROOT / "04_Results" / "metrics"
OUT.mkdir(parents=True, exist_ok=True)
METRICS.mkdir(parents=True, exist_ok=True)

def save_csv_memory_safe(df, path):
    df.to_csv(path, index=False, chunksize=5000)

csv_files = sorted(RAW.glob("*.csv"))
if not csv_files:
    raise FileNotFoundError(f"No CICIDS2017 CSV files found in {RAW}")

rows_per_file = max(1, args.max_rows // len(csv_files))

frames = []
file_summary = []
dropped_column_records = []
global_columns = None

print("Loading CICIDS2017 files with balanced file-wise sampling...")
print(f"Target total rows: {args.max_rows}")
print(f"Files detected: {len(csv_files)}")
print(f"Target rows per file: {rows_per_file}")

for path in csv_files:
    print(f"\nReading file: {path.name}")
    file_frames = []
    file_rows = 0

    chunk_iter = pd.read_csv(
        path,
        chunksize=args.chunk_size,
        low_memory=False,
        encoding="latin1"
    )

    for i, df in enumerate(chunk_iter, start=1):
        df.columns = [str(c).strip() for c in df.columns]

        if "Label" not in df.columns:
            raise ValueError(f"Label column not found in {path.name}. Found: {df.columns.tolist()}")

        label_text = df["Label"].astype(str).str.strip()
        y_chunk = (label_text.str.upper() != "BENIGN").astype(np.int8)

        X_chunk = df.drop(columns=["Label"])
        X_numeric = X_chunk.apply(pd.to_numeric, errors="coerce")
        X_numeric = X_numeric.replace([np.inf, -np.inf], np.nan)

        missing_ratio = X_numeric.isna().mean()
        cols_to_drop = missing_ratio[missing_ratio > args.max_missing_col_ratio].index.tolist()

        for col in cols_to_drop:
            dropped_column_records.append({
                "file": path.name,
                "chunk": i,
                "column": col,
                "missing_ratio": float(missing_ratio[col])
            })

        X_numeric = X_numeric.drop(columns=cols_to_drop, errors="ignore")

        medians = X_numeric.median(numeric_only=True).fillna(0)
        X_numeric = X_numeric.fillna(medians).fillna(0)

        clean_chunk = X_numeric.copy()
        clean_chunk["binary_label"] = y_chunk.values
        clean_chunk["source_file"] = path.name
        clean_chunk["original_label"] = label_text.values

        remaining = rows_per_file - file_rows
        if remaining <= 0:
            break

        if len(clean_chunk) > remaining:
            clean_chunk = clean_chunk.sample(n=remaining, random_state=42)

        file_frames.append(clean_chunk)
        file_rows += len(clean_chunk)

        print(f"  chunk {i}: file rows = {file_rows}")

        del df, X_chunk, X_numeric, clean_chunk, y_chunk, label_text
        gc.collect()

        if file_rows >= rows_per_file:
            break

    if file_frames:
        file_data = pd.concat(file_frames, axis=0, ignore_index=True)
        frames.append(file_data)

        file_summary.append({
            "file": path.name,
            "rows_collected": len(file_data),
            "attack_ratio": float(file_data["binary_label"].mean()),
            "labels": str(file_data["original_label"].value_counts().to_dict())
        })

        del file_data, file_frames
        gc.collect()

if not frames:
    raise RuntimeError("No valid CICIDS2017 rows were collected.")

data = pd.concat(frames, axis=0, ignore_index=True)
del frames
gc.collect()

if len(data) > args.max_rows:
    data = data.sample(n=args.max_rows, random_state=42).reset_index(drop=True)

pd.DataFrame(file_summary).to_csv(
    METRICS / "cicids2017_balanced_file_sampling_summary.csv",
    index=False
)

if dropped_column_records:
    pd.DataFrame(dropped_column_records).drop_duplicates().to_csv(
        METRICS / "cicids2017_dropped_columns_report.csv",
        index=False
    )

print("\nCombined CICIDS2017 shape after balanced sampling:", data.shape)

metadata_cols = ["source_file", "original_label"]
y = data["binary_label"].astype(np.int8)
X = data.drop(columns=["binary_label"] + metadata_cols)

# Align and clean final features
for col in X.columns:
    X[col] = pd.to_numeric(X[col], errors="coerce")

X = X.replace([np.inf, -np.inf], np.nan)
X = X.fillna(X.median(numeric_only=True).fillna(0)).fillna(0)

print("Final feature count:", X.shape[1])
print("Binary class distribution:")
print(y.value_counts().to_string())

print("\nOriginal CICIDS2017 label distribution in sample:")
print(data["original_label"].value_counts().to_string())

print("\nSource file distribution in sample:")
print(data["source_file"].value_counts().to_string())

# Save dataset composition reports
data[["source_file", "original_label", "binary_label"]].to_csv(
    METRICS / "cicids2017_sample_composition.csv",
    index=False
)

# Train/val/test split
X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

del X, y
gc.collect()

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=42,
    stratify=y_temp
)

del X_temp, y_temp
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

joblib.dump(scaler, OUT / "scaler_cicids2017.joblib")

summary = pd.DataFrame([
    {"split": "train", "rows": len(y_train), "columns": len(columns), "attack_ratio": float(y_train.mean())},
    {"split": "val", "rows": len(y_val), "columns": len(columns), "attack_ratio": float(y_val.mean())},
    {"split": "test", "rows": len(y_test), "columns": len(columns), "attack_ratio": float(y_test.mean())},
])
summary.to_csv(OUT / "preprocessing_summary.csv", index=False)

print("\nCICIDS2017 balanced preprocessing complete.")
print(summary.to_string(index=False))
print("Saved to:", OUT)
print(f"\nNote: This run used {len(y_train) + len(y_val) + len(y_test)} CICIDS2017 rows sampled across all CSV files.")