"""
CICIDS2017 credibility audit for Paper A.

This script addresses reviewer concerns about near-perfect CICIDS2017 scores.
It runs two complementary checks:

1. pooled_balanced_random_split
   Recreates a balanced file-wise sample across all CICIDS2017 CSV files and
   evaluates repeated sampling seeds. This measures sensitivity to the sample.

2. file_holdout_split
   Uses a stricter file-wise split: entire source CSV files are held out for test.
   This reduces the risk that highly correlated flows from the same file/day appear
   in both train and test.

Outputs:
04_Results/metrics/cicids2017_credibility_audit_results.csv
04_Results/metrics/cicids2017_credibility_audit_composition.csv
06_LaTeX/tables/table_cicids2017_credibility_audit.tex

Example:
python 03_Code/scripts/09_cicids2017_credibility_audit.py --max-rows 600000 --rows-per-file 75000 --chunk-size 25000
"""
from __future__ import annotations

from pathlib import Path
import argparse
import gc
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, confusion_matrix

warnings.filterwarnings("ignore")
try:
    from lightgbm import LGBMClassifier
except Exception as exc:
    raise ImportError("Install lightgbm before running this script: pip install lightgbm") from exc

parser = argparse.ArgumentParser()
parser.add_argument("--max-rows", type=int, default=600000)
parser.add_argument("--rows-per-file", type=int, default=75000)
parser.add_argument("--chunk-size", type=int, default=25000)
parser.add_argument("--sample-seeds", type=str, default="42,123,2024")
parser.add_argument("--holdout-files", type=str, default="Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv,Wednesday-workingHours.pcap_ISCX.csv",
                    help="Comma-separated CICIDS2017 CSV files held out completely for file_holdout_split.")
args = parser.parse_args()

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "02_Data" / "raw" / "CICIDS2017"
METRICS = ROOT / "04_Results" / "metrics"
TABLES = ROOT / "06_LaTeX" / "tables"
METRICS.mkdir(parents=True, exist_ok=True)
TABLES.mkdir(parents=True, exist_ok=True)

FEATURE_DROP_MISSING_RATIO = 0.50
SEEDS = [int(s.strip()) for s in args.sample_seeds.split(',') if s.strip()]
HOLDOUT_FILES = {s.strip() for s in args.holdout_files.split(',') if s.strip()}


def clean_chunk(df: pd.DataFrame, source_file: str, global_cols: list[str] | None = None):
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    if "Label" not in df.columns:
        raise ValueError(f"Label column not found in {source_file}")
    label_text = df["Label"].astype(str).str.strip()
    y = (label_text.str.upper() != "BENIGN").astype(np.int8)
    X = df.drop(columns=["Label"])
    X = X.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    miss = X.isna().mean()
    X = X.drop(columns=miss[miss > FEATURE_DROP_MISSING_RATIO].index.tolist(), errors="ignore")
    X = X.fillna(X.median(numeric_only=True).fillna(0)).fillna(0)
    if global_cols is not None:
        X = X.reindex(columns=global_cols, fill_value=0)
    out = X.copy()
    out["binary_label"] = y.values
    out["source_file"] = source_file
    out["original_label"] = label_text.values
    return out


def collect_balanced_sample(rows_per_file: int, seed: int):
    csv_files = sorted(RAW.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {RAW}")
    frames = []
    global_cols = None
    for path in csv_files:
        file_frames = []
        file_rows = 0
        for chunk in pd.read_csv(path, chunksize=args.chunk_size, low_memory=False, encoding="latin1"):
            clean = clean_chunk(chunk, path.name, global_cols)
            if global_cols is None:
                global_cols = [c for c in clean.columns if c not in ["binary_label", "source_file", "original_label"]]
            remaining = rows_per_file - file_rows
            if remaining <= 0:
                break
            if len(clean) > remaining:
                clean = clean.sample(n=remaining, random_state=seed)
            file_frames.append(clean)
            file_rows += len(clean)
            if file_rows >= rows_per_file:
                break
        if file_frames:
            frames.append(pd.concat(file_frames, ignore_index=True))
        gc.collect()
    data = pd.concat(frames, ignore_index=True)
    if len(data) > args.max_rows:
        data = data.sample(n=args.max_rows, random_state=seed).reset_index(drop=True)
    return data


def collect_filewise_data(seed: int):
    csv_files = sorted(RAW.glob("*.csv"))
    frames = []
    global_cols = None
    for path in csv_files:
        file_frames = []
        file_rows = 0
        for chunk in pd.read_csv(path, chunksize=args.chunk_size, low_memory=False, encoding="latin1"):
            clean = clean_chunk(chunk, path.name, global_cols)
            if global_cols is None:
                global_cols = [c for c in clean.columns if c not in ["binary_label", "source_file", "original_label"]]
            remaining = args.rows_per_file - file_rows
            if remaining <= 0:
                break
            if len(clean) > remaining:
                clean = clean.sample(n=remaining, random_state=seed)
            file_frames.append(clean)
            file_rows += len(clean)
            if file_rows >= args.rows_per_file:
                break
        if file_frames:
            frames.append(pd.concat(file_frames, ignore_index=True))
        gc.collect()
    return pd.concat(frames, ignore_index=True)


def evaluate_split(X_train, X_test, y_train, y_test, seed: int):
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train).astype(np.float32)
    X_test_s = scaler.transform(X_test).astype(np.float32)
    model = LGBMClassifier(n_estimators=500, learning_rate=0.05, num_leaves=64,
                           subsample=0.9, colsample_bytree=0.9, class_weight="balanced",
                           random_state=seed, n_jobs=-1, verbose=-1)
    model.fit(X_train_s, y_train)
    pred = model.predict(X_test_s)
    score = model.predict_proba(X_test_s)[:, 1]
    tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
    return {
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred, zero_division=0),
        "recall": recall_score(y_test, pred, zero_division=0),
        "f1": f1_score(y_test, pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, score),
        "pr_auc": average_precision_score(y_test, score),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "fpr": fp / (fp + tn) if (fp + tn) else 0,
        "fnr": fn / (fn + tp) if (fn + tp) else 0,
    }


def split_xy(data: pd.DataFrame):
    y = data["binary_label"].astype(int)
    X = data.drop(columns=["binary_label", "source_file", "original_label"], errors="ignore")
    return X, y

results = []
composition = []

for seed in SEEDS:
    print(f"\nPooled balanced random split audit, seed={seed}")
    data = collect_balanced_sample(args.rows_per_file, seed)
    for f, g in data.groupby("source_file"):
        composition.append({"protocol":"pooled_balanced_random_split", "seed":seed, "source_file":f,
                            "rows":len(g), "attack_ratio":float(g["binary_label"].mean()),
                            "labels":str(g["original_label"].value_counts().to_dict())})
    X, y = split_xy(data)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, stratify=y, random_state=seed)
    rec = {"protocol":"pooled_balanced_random_split", "seed":seed,
           "train_rows":len(X_train), "test_rows":len(X_test),
           "train_attack_ratio":float(y_train.mean()), "test_attack_ratio":float(y_test.mean())}
    rec.update(evaluate_split(X_train, X_test, y_train, y_test, seed))
    results.append(rec)
    del data, X, y, X_train, X_test, y_train, y_test
    gc.collect()

print("\nFile-wise holdout audit")
file_data = collect_filewise_data(seed=42)
train_data = file_data[~file_data["source_file"].isin(HOLDOUT_FILES)].reset_index(drop=True)
test_data = file_data[file_data["source_file"].isin(HOLDOUT_FILES)].reset_index(drop=True)
for split_name, data in [("train_files", train_data), ("heldout_test_files", test_data)]:
    for f, g in data.groupby("source_file"):
        composition.append({"protocol":"file_holdout_split", "seed":42, "split":split_name, "source_file":f,
                            "rows":len(g), "attack_ratio":float(g["binary_label"].mean()),
                            "labels":str(g["original_label"].value_counts().to_dict())})
X_train, y_train = split_xy(train_data)
X_test, y_test = split_xy(test_data)
rec = {"protocol":"file_holdout_split", "seed":42,
       "holdout_files":" | ".join(sorted(HOLDOUT_FILES)),
       "train_rows":len(X_train), "test_rows":len(X_test),
       "train_attack_ratio":float(y_train.mean()), "test_attack_ratio":float(y_test.mean())}
rec.update(evaluate_split(X_train, X_test, y_train, y_test, seed=42))
results.append(rec)

res_df = pd.DataFrame(results)
comp_df = pd.DataFrame(composition)
res_df.to_csv(METRICS / "cicids2017_credibility_audit_results.csv", index=False)
comp_df.to_csv(METRICS / "cicids2017_credibility_audit_composition.csv", index=False)

latex = res_df.copy()
for col in ["accuracy","precision","recall","f1","roc_auc","pr_auc","fpr","fnr","train_attack_ratio","test_attack_ratio"]:
    if col in latex.columns:
        latex[col] = latex[col].astype(float).round(4)
cols = [c for c in ["protocol","seed","train_rows","test_rows","train_attack_ratio","test_attack_ratio","f1","roc_auc","pr_auc","fpr","fnr"] if c in latex.columns]
latex_text = latex[cols].to_latex(index=False, caption="CICIDS2017 credibility audit under pooled-random and file-wise holdout protocols.", label="tab:cicids_audit", float_format="%.4f")
(TABLES / "table_cicids2017_credibility_audit.tex").write_text(latex_text, encoding="utf-8")
print("\nSaved CICIDS2017 credibility audit results:")
print(METRICS / "cicids2017_credibility_audit_results.csv")
print(TABLES / "table_cicids2017_credibility_audit.tex")
print(res_df.to_string(index=False))
