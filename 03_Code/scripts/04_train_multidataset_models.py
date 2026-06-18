from pathlib import Path
import argparse
import time
import warnings
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix
)

warnings.filterwarnings("ignore")

try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except Exception:
    HAS_LGBM = False

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:
    HAS_XGB = False


parser = argparse.ArgumentParser()
parser.add_argument("--max-train-rows", type=int, default=0,
                    help="0 = use full training set. Use smaller value for testing.")
parser.add_argument("--random-state", type=int, default=42)
args = parser.parse_args()

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "02_Data" / "processed"
OUT = ROOT / "04_Results" / "metrics"
OUT.mkdir(parents=True, exist_ok=True)


DATASETS = {
    "NSL-KDD": PROCESSED,
    "UNSW-NB15": PROCESSED / "UNSW-NB15",
    "CICIDS2017": PROCESSED / "CICIDS2017",
}


def read_y(path):
    y_df = pd.read_csv(path)
    if y_df.shape[1] == 1:
        return y_df.iloc[:, 0].astype(int)
    if "label" in y_df.columns:
        return y_df["label"].astype(int)
    return y_df.iloc[:, -1].astype(int)


def load_dataset(dataset_name, dataset_path):
    print(f"\nLoading dataset: {dataset_name}")

    X_train = pd.read_csv(dataset_path / "X_train_final.csv").astype(np.float32)
    X_val = pd.read_csv(dataset_path / "X_val_final.csv").astype(np.float32)
    X_test = pd.read_csv(dataset_path / "X_test_final.csv").astype(np.float32)

    y_train = read_y(dataset_path / "y_train_binary.csv")
    y_val = read_y(dataset_path / "y_val_binary.csv")
    y_test = read_y(dataset_path / "y_test_binary.csv")

    print(f"  X_train: {X_train.shape}, X_val: {X_val.shape}, X_test: {X_test.shape}")
    print(f"  train attack ratio: {y_train.mean():.4f}")
    print(f"  val attack ratio:   {y_val.mean():.4f}")
    print(f"  test attack ratio:  {y_test.mean():.4f}")

    if args.max_train_rows > 0 and len(X_train) > args.max_train_rows:
        sample_idx = X_train.sample(
            n=args.max_train_rows,
            random_state=args.random_state
        ).index
        X_train = X_train.loc[sample_idx].reset_index(drop=True)
        y_train = y_train.loc[sample_idx].reset_index(drop=True)
        print(f"  using sampled training rows: {len(X_train)}")

    return X_train, X_val, X_test, y_train, y_val, y_test


def get_models():
    models = {
        "LogisticRegression": LogisticRegression(
            max_iter=1000,
            n_jobs=-1,
            class_weight="balanced",
            random_state=args.random_state
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            min_samples_leaf=1,
            n_jobs=-1,
            class_weight="balanced",
            random_state=args.random_state
        ),
        "ExtraTrees": ExtraTreesClassifier(
            n_estimators=200,
            max_depth=None,
            min_samples_leaf=1,
            n_jobs=-1,
            class_weight="balanced",
            random_state=args.random_state
        )
    }

    if HAS_LGBM:
        models["LightGBM"] = LGBMClassifier(
            n_estimators=500,
            learning_rate=0.05,
            num_leaves=64,
            subsample=0.9,
            colsample_bytree=0.9,
            class_weight="balanced",
            random_state=args.random_state,
            n_jobs=-1
        )

    if HAS_XGB:
        models["XGBoost"] = XGBClassifier(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=8,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            random_state=args.random_state,
            n_jobs=-1
        )

    return models


def evaluate_model(model, X, y):
    pred = model.predict(X)

    if hasattr(model, "predict_proba"):
        score = model.predict_proba(X)[:, 1]
    elif hasattr(model, "decision_function"):
        score = model.decision_function(X)
    else:
        score = pred

    tn, fp, fn, tp = confusion_matrix(y, pred).ravel()

    return {
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "roc_auc": roc_auc_score(y, score),
        "pr_auc": average_precision_score(y, score),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "fpr": fp / (fp + tn) if (fp + tn) > 0 else 0,
        "fnr": fn / (fn + tp) if (fn + tp) > 0 else 0
    }


validation_records = []
test_records = []
best_model_records = []

for dataset_name, dataset_path in DATASETS.items():
    required = [
        "X_train_final.csv", "X_val_final.csv", "X_test_final.csv",
        "y_train_binary.csv", "y_val_binary.csv", "y_test_binary.csv"
    ]

    missing = [f for f in required if not (dataset_path / f).exists()]
    if missing:
        print(f"\nSkipping {dataset_name}. Missing files: {missing}")
        continue

    X_train, X_val, X_test, y_train, y_val, y_test = load_dataset(dataset_name, dataset_path)
    models = get_models()

    best_name = None
    best_model = None
    best_f1 = -1

    for model_name, model in models.items():
        print(f"\nTraining {model_name} on {dataset_name}...")

        start_train = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - start_train

        start_infer = time.time()
        val_metrics = evaluate_model(model, X_val, y_val)
        val_infer_time = time.time() - start_infer

        val_record = {
            "dataset": dataset_name,
            "model": model_name,
            "split": "validation",
            "train_rows": len(X_train),
            "features": X_train.shape[1],
            "train_time_sec": train_time,
            "inference_time_sec": val_infer_time,
            "inference_ms_per_sample": (val_infer_time / len(X_val)) * 1000
        }
        val_record.update(val_metrics)
        validation_records.append(val_record)

        print(
            f"  Validation F1={val_metrics['f1']:.4f}, "
            f"ROC-AUC={val_metrics['roc_auc']:.4f}, "
            f"PR-AUC={val_metrics['pr_auc']:.4f}"
        )

        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            best_name = model_name
            best_model = model

    print(f"\nBest validation model for {dataset_name}: {best_name} with F1={best_f1:.4f}")

    start_test = time.time()
    test_metrics = evaluate_model(best_model, X_test, y_test)
    test_infer_time = time.time() - start_test

    test_record = {
        "dataset": dataset_name,
        "model": best_name,
        "split": "test",
        "train_rows": len(X_train),
        "features": X_train.shape[1],
        "inference_time_sec": test_infer_time,
        "inference_ms_per_sample": (test_infer_time / len(X_test)) * 1000
    }
    test_record.update(test_metrics)
    test_records.append(test_record)

    best_model_records.append({
        "dataset": dataset_name,
        "best_model": best_name,
        "validation_f1": best_f1,
        "test_f1": test_metrics["f1"],
        "test_roc_auc": test_metrics["roc_auc"],
        "test_pr_auc": test_metrics["pr_auc"]
    })

    print(
        f"  Test F1={test_metrics['f1']:.4f}, "
        f"ROC-AUC={test_metrics['roc_auc']:.4f}, "
        f"PR-AUC={test_metrics['pr_auc']:.4f}"
    )


validation_df = pd.DataFrame(validation_records)
test_df = pd.DataFrame(test_records)
best_df = pd.DataFrame(best_model_records)

validation_df.to_csv(OUT / "multidataset_validation_results.csv", index=False)
test_df.to_csv(OUT / "multidataset_test_results.csv", index=False)
best_df.to_csv(OUT / "multidataset_best_models.csv", index=False)

print("\nSaved:")
print(OUT / "multidataset_validation_results.csv")
print(OUT / "multidataset_test_results.csv")
print(OUT / "multidataset_best_models.csv")

print("\nBest model summary:")
print(best_df.to_string(index=False))