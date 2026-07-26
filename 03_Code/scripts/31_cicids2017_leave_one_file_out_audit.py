from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import argparse
import re
import time

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold

from lightgbm import LGBMClassifier


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------
RANDOM_STATE = 42

EXCLUDE_NAME_PATTERNS = [
    "processed",
    "summary",
    "metrics",
    "audit",
    "readme",
]


# ---------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------
def project_root_from_script() -> Path:
    return Path(__file__).resolve().parents[2]


def normalize_dataset_file_name(path: Path) -> str:
    name = path.stem
    name = re.sub(r"[_\-]+", " ", name)
    name = re.sub(r"\s+", " ", name)
    return name.strip()


def looks_like_cicids_file(path: Path) -> bool:
    if path.suffix.lower() != ".csv":
        return False

    s = str(path).lower().replace("\\", "/")
    name = path.name.lower()

    if any(p in s for p in EXCLUDE_NAME_PATTERNS):
        return False

    # Common CICIDS2017 raw file cues
    cues = [
        "cicids2017",
        "monday",
        "tuesday",
        "wednesday",
        "thursday",
        "friday",
        "ddos",
        "portscan",
        "botnet",
        "infiltration",
        "bruteforce",
        "web",
        "heartbleed",
    ]

    return any(c in s or c in name for c in cues)


def find_cicids_raw_csvs(root: Path) -> List[Path]:
    search_roots = [
        root / "01_Data",
        root / "02_Data",
        root / "data",
        root,
    ]

    files = []

    for sr in search_roots:
        if not sr.exists():
            continue
        for p in sr.rglob("*.csv"):
            if looks_like_cicids_file(p):
                files.append(p)

    # De-duplicate and prefer files that are not under processed/final metric folders.
    uniq = sorted(set(files), key=lambda p: str(p).lower())

    # Keep only files that appear to have Label column and some feature columns.
    valid = []
    for p in uniq:
        try:
            preview = pd.read_csv(p, nrows=5)
            cols = [str(c).strip().lower() for c in preview.columns]
            has_label = any(c == "label" or c.endswith(" label") for c in cols)
            if has_label and preview.shape[1] >= 10:
                valid.append(p)
        except Exception:
            continue

    return valid


# ---------------------------------------------------------------------
# Data preprocessing
# ---------------------------------------------------------------------
def find_label_col(df: pd.DataFrame) -> str:
    candidates = [c for c in df.columns if str(c).strip().lower() == "label"]
    if candidates:
        return candidates[0]

    candidates = [c for c in df.columns if "label" in str(c).strip().lower()]
    if candidates:
        return candidates[0]

    raise KeyError(f"No label column found. Columns: {list(df.columns)[:20]}")


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]
    return out


def binarize_label(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip().str.lower()
    return (~s.isin(["benign", "normal", "0"])).astype(int)


def load_sampled_file(path: Path, max_rows_per_file: int, random_state: int) -> pd.DataFrame:
    """
    Robust CICIDS2017 CSV loader.

    Some CICIDS2017 raw files contain non-UTF-8 characters. We try UTF-8 first,
    then fall back to Windows-1252/Latin-1 compatible decoding.
    """
    read_errors = []

    for enc in ["utf-8", "cp1252", "latin1"]:
        try:
            df = pd.read_csv(path, low_memory=False, encoding=enc)
            print(f"Loaded with encoding: {enc}")
            break
        except UnicodeDecodeError as e:
            read_errors.append(f"{enc}: {e}")
            df = None

    if df is None:
        raise UnicodeDecodeError(
            "utf-8",
            b"",
            0,
            1,
            "Could not decode file with utf-8, cp1252, or latin1. "
            + " | ".join(read_errors),
        )

    df = clean_columns(df)

    if max_rows_per_file > 0 and len(df) > max_rows_per_file:
        df = df.sample(max_rows_per_file, random_state=random_state)

    df["source_file"] = normalize_dataset_file_name(path)

    return df.reset_index(drop=True)


def prepare_xy(train_df: pd.DataFrame, test_df: pd.DataFrame):
    """
    CICIDS2017 raw files are expected to be numeric after label/source removal.
    To avoid accidental one-hot explosion from dirty numeric strings, every
    feature is coerced to numeric and non-convertible values become NaN.
    """
    label_col = find_label_col(train_df)

    y_train = binarize_label(train_df[label_col])
    y_test = binarize_label(test_df[label_col])

    drop_cols = [label_col, "source_file"]

    x_train = train_df.drop(columns=[c for c in drop_cols if c in train_df.columns])
    x_test = test_df.drop(columns=[c for c in drop_cols if c in test_df.columns])

    # Align columns safely
    common_cols = [c for c in x_train.columns if c in x_test.columns]
    x_train = x_train[common_cols].copy()
    x_test = x_test[common_cols].copy()

    # Clean common CICIDS malformed entries
    x_train = x_train.replace([np.inf, -np.inf, "Infinity", "inf", "-inf", "NaN", "nan", ""], np.nan)
    x_test = x_test.replace([np.inf, -np.inf, "Infinity", "inf", "-inf", "NaN", "nan", ""], np.nan)

    # Force every feature to numeric. Non-numeric cells become NaN.
    for col in common_cols:
        x_train[col] = pd.to_numeric(x_train[col], errors="coerce")
        x_test[col] = pd.to_numeric(x_test[col], errors="coerce")

    # Drop columns that are entirely missing in training
    non_empty_cols = [c for c in x_train.columns if not x_train[c].isna().all()]
    x_train = x_train[non_empty_cols]
    x_test = x_test[non_empty_cols]

    # Median imputation using training medians
    medians = x_train.median(numeric_only=True)
    x_train = x_train.fillna(medians)
    x_test = x_test.fillna(medians)

    # Any remaining NaN after alignment gets zero
    x_train = x_train.fillna(0.0)
    x_test = x_test.fillna(0.0)

    return x_train, y_train, x_test, y_test


def build_pipeline() -> Pipeline:
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("variance", VarianceThreshold(threshold=0.0)),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    # ColumnTransformer columns are resolved after fitting via make_preprocessor.
    model = LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    return model


def make_preprocessor(x_train: pd.DataFrame) -> ColumnTransformer:
    numeric_cols = x_train.select_dtypes(include=[np.number, "bool"]).columns.tolist()
    categorical_cols = [c for c in x_train.columns if c not in numeric_cols]

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("variance", VarianceThreshold(threshold=0.0)),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    transformers = []
    if numeric_cols:
        transformers.append(("num", numeric_transformer, numeric_cols))
    if categorical_cols:
        transformers.append(("cat", categorical_transformer, categorical_cols))

    return ColumnTransformer(transformers=transformers, remainder="drop")


# ---------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------
def compute_binary_metrics(y_true, y_pred, y_prob) -> Dict[str, float]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    fpr = fp / (fp + tn) if (fp + tn) > 0 else np.nan
    fnr = fn / (fn + tp) if (fn + tp) > 0 else np.nan

    try:
        roc_auc = roc_auc_score(y_true, y_prob)
    except Exception:
        roc_auc = np.nan

    try:
        pr_auc = average_precision_score(y_true, y_prob)
    except Exception:
        pr_auc = np.nan

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "fpr": fpr,
        "fnr": fnr,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


# ---------------------------------------------------------------------
# LaTeX
# ---------------------------------------------------------------------
def latex_escape(text: object) -> str:
    s = str(text)
    replacements = {
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    for old, new in replacements.items():
        s = s.replace(old, new)
    return s


def fmt(x: object, ndigits: int = 4) -> str:
    if pd.isna(x):
        return "--"
    return f"{float(x):.{ndigits}f}"


def write_latex_leave_one_file_table(df: pd.DataFrame, path: Path) -> None:
    lines = []

    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{CICIDS2017 leave-one-file-out audit using LightGBM. Each row holds out one source traffic file and trains on the remaining files. This audit is reported as an external dataset-shift stress test rather than as a RAISE-IDS scoring component.}")
    lines.append(r"\label{tab:cicids2017_leave_one_file_out_audit}")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{lrrrrrr}")
    lines.append(r"\hline")
    lines.append(r"\textbf{Held-out file} & \textbf{Test rows} & \textbf{Attack ratio} & \textbf{F1} & \textbf{PR-AUC} & \textbf{FPR} & \textbf{FNR} \\")
    lines.append(r"\hline")

    for _, row in df.sort_values("f1").iterrows():
        lines.append(
            f"{latex_escape(row['heldout_file'])} & "
            f"{int(row['test_rows'])} & "
            f"{fmt(row['test_attack_ratio'])} & "
            f"{fmt(row['f1'])} & "
            f"{fmt(row['pr_auc'])} & "
            f"{fmt(row['fpr'])} & "
            f"{fmt(row['fnr'])} \\\\"
        )

    lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table*}")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-rows-per-file", type=int, default=75000)
    parser.add_argument("--min-files", type=int, default=4)
    parser.add_argument("--random-state", type=int, default=RANDOM_STATE)
    args = parser.parse_args()

    root = project_root_from_script()
    metrics_dir = root / "04_Results" / "metrics"
    tables_dir = root / "06_LaTeX" / "tables"

    metrics_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    raw_files = find_cicids_raw_csvs(root)

    print("Detected CICIDS2017 candidate raw files:")
    for p in raw_files:
        print(f" - {p}")

    if len(raw_files) < args.min_files:
        raise RuntimeError(
            f"Only {len(raw_files)} CICIDS2017 raw-like files were found. "
            f"Need at least {args.min_files} for leave-one-file-out audit. "
            "Please confirm the raw CICIDS2017 CSV files are present under 01_Data, 02_Data, or data."
        )

    loaded = []

    for p in raw_files:
        print(f"\nLoading: {p}")
        df = load_sampled_file(p, args.max_rows_per_file, args.random_state)
        label_col = find_label_col(df)
        y = binarize_label(df[label_col])
        print(
            f"Rows used: {len(df)}, features including label/source: {df.shape[1]}, "
            f"attack_ratio={y.mean():.4f}"
        )
        loaded.append(df)

    all_files = [df["source_file"].iloc[0] for df in loaded]
    rows = []

    for heldout_name in all_files:
        print(f"\n=== Leave-one-file-out: held out {heldout_name} ===")

        test_parts = [df for df in loaded if df["source_file"].iloc[0] == heldout_name]
        train_parts = [df for df in loaded if df["source_file"].iloc[0] != heldout_name]

        train_df = pd.concat(train_parts, ignore_index=True)
        test_df = pd.concat(test_parts, ignore_index=True)

        x_train, y_train, x_test, y_test = prepare_xy(train_df, test_df)

        # Skip held-out files with only one class in test; metrics are unstable.
        if y_test.nunique() < 2:
            print(f"Skipping {heldout_name}: held-out file has only one class.")
            continue

        model = build_pipeline()

        start = time.time()
        model.fit(x_train, y_train)
        train_time = time.time() - start

        y_prob = model.predict_proba(x_test)[:, 1]        
        y_pred = (y_prob >= 0.5).astype(int)

        m = compute_binary_metrics(y_test, y_pred, y_prob)

        row = {
            "heldout_file": heldout_name,
            "train_rows": len(x_train),
            "test_rows": len(x_test),
            "train_attack_ratio": float(y_train.mean()),
            "test_attack_ratio": float(y_test.mean()),
            "train_time_sec": train_time,
            **m,
        }
        rows.append(row)

        print(
            f"F1={m['f1']:.4f}, PR-AUC={m['pr_auc']:.4f}, "
            f"FPR={m['fpr']:.4f}, FNR={m['fnr']:.4f}, "
            f"train_sec={train_time:.1f}"
        )

    result = pd.DataFrame(rows)

    if result.empty:
        raise RuntimeError("No valid leave-one-file-out folds were completed.")

    out_csv = metrics_dir / "cicids2017_leave_one_file_out_audit.csv"
    result.to_csv(out_csv, index=False)

    summary = pd.DataFrame(
        [
            {
                "num_completed_folds": len(result),
                "mean_f1": result["f1"].mean(),
                "median_f1": result["f1"].median(),
                "min_f1": result["f1"].min(),
                "max_f1": result["f1"].max(),
                "mean_pr_auc": result["pr_auc"].mean(),
                "mean_fnr": result["fnr"].mean(),
                "mean_fpr": result["fpr"].mean(),
                "worst_heldout_file_by_f1": result.sort_values("f1").iloc[0]["heldout_file"],
            }
        ]
    )

    out_summary = metrics_dir / "cicids2017_leave_one_file_out_summary.csv"
    summary.to_csv(out_summary, index=False)

    latex_table = tables_dir / "table_cicids2017_leave_one_file_out_audit.tex"
    write_latex_leave_one_file_table(result, latex_table)

    print("\nSaved:")
    print(out_csv)
    print(out_summary)
    print(latex_table)

    print("\nLeave-one-file-out results:")
    print(result.sort_values("f1").to_string(index=False))

    print("\nSummary:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()