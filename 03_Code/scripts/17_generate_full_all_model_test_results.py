from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.neural_network import MLPClassifier

from lightgbm import LGBMClassifier
from xgboost import XGBClassifier


RANDOM_STATE = 42


def project_root_from_script() -> Path:
    return Path(__file__).resolve().parents[2]


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    return pd.read_csv(path)


def dataset_aliases(dataset_name: str) -> List[str]:
    mapping = {
        "NSL-KDD": ["nsl-kdd", "nsl_kdd", "nslkdd", "nsl kdd"],
        "UNSW-NB15": ["unsw-nb15", "unsw_nb15", "unswnb15", "unsw nb15"],
        "CICIDS2017": ["cicids2017", "cic ids 2017", "cic-ids2017", "cic_ids2017"],
    }
    return mapping.get(dataset_name, [dataset_name.lower()])


def find_first_existing(dir_path: Path, names: List[str]) -> Optional[Path]:
    for name in names:
        p = dir_path / name
        if p.exists():
            return p
    return None


def is_complete_processed_dir(dir_path: Path) -> bool:
    x_ok = all((dir_path / f).exists() for f in ["X_train_final.csv", "X_val_final.csv", "X_test_final.csv"])
    y_train = find_first_existing(dir_path, ["y_train_binary.csv", "y_train.csv"])
    y_val = find_first_existing(dir_path, ["y_val_binary.csv", "y_val.csv"])
    y_test = find_first_existing(dir_path, ["y_test_binary.csv", "y_test.csv"])
    return x_ok and y_train is not None and y_val is not None and y_test is not None


def find_processed_dataset_dir(root: Path, dataset_name: str) -> Path:
    search_root = root / "02_Data"
    if not search_root.exists():
        raise FileNotFoundError(f"Could not find 02_Data folder under: {root}")

    candidates: List[Path] = []

    # search recursively for X_train_final.csv and validate surrounding directory
    for x_train_file in search_root.rglob("X_train_final.csv"):
        candidate_dir = x_train_file.parent
        if is_complete_processed_dir(candidate_dir):
            candidates.append(candidate_dir)

    if not candidates:
        raise FileNotFoundError(
            f"No processed dataset directories with X_train_final/X_val_final/X_test_final and y_* files found under: {search_root}"
        )

    aliases = dataset_aliases(dataset_name)

    scored: List[Tuple[int, Path]] = []
    for c in candidates:
        path_str = str(c).lower().replace("\\", "/")
        score = 0
        for alias in aliases:
            alias_norm = alias.replace(" ", "").replace("-", "").replace("_", "")
            path_norm = path_str.replace(" ", "").replace("-", "").replace("_", "")
            if alias_norm in path_norm:
                score += 10
        # mild preference for "processed" directories
        if "processed" in path_str:
            score += 2
        scored.append((score, c))

    scored.sort(key=lambda x: (-x[0], str(x[1])))

    best_score, best_dir = scored[0]

    if best_score <= 0:
        candidate_text = "\n".join(str(p) for _, p in scored)
        raise FileNotFoundError(
            f"Could not confidently match processed directory for dataset '{dataset_name}'.\n"
            f"Available processed candidates were:\n{candidate_text}"
        )

    return best_dir


def load_dataset_splits(root: Path, dataset_name: str) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    proc_dir = find_processed_dataset_dir(root, dataset_name)
    print(f"Using processed directory for {dataset_name}: {proc_dir}")

    x_train = read_csv(proc_dir / "X_train_final.csv")
    x_val = read_csv(proc_dir / "X_val_final.csv")
    x_test = read_csv(proc_dir / "X_test_final.csv")

    y_train_path = find_first_existing(proc_dir, ["y_train_binary.csv", "y_train.csv"])
    y_val_path = find_first_existing(proc_dir, ["y_val_binary.csv", "y_val.csv"])
    y_test_path = find_first_existing(proc_dir, ["y_test_binary.csv", "y_test.csv"])

    if y_train_path is None or y_val_path is None or y_test_path is None:
        raise FileNotFoundError(f"Missing y files in processed directory: {proc_dir}")

    y_train = read_csv(y_train_path).iloc[:, 0]
    y_val = read_csv(y_val_path).iloc[:, 0]
    y_test = read_csv(y_test_path).iloc[:, 0]

    return x_train, y_train, x_val, y_val, x_test, y_test


def get_models() -> Dict[str, object]:
    return {
        "LogisticRegression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            solver="liblinear",
            random_state=RANDOM_STATE,
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=300,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            class_weight="balanced_subsample",
        ),
        "ExtraTrees": ExtraTreesClassifier(
            n_estimators=300,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            class_weight="balanced",
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "MLP": MLPClassifier(
            hidden_layer_sizes=(128, 64),
            activation="relu",
            solver="adam",
            learning_rate_init=1e-3,
            max_iter=100,
            early_stopping=True,
            validation_fraction=0.1,
            random_state=RANDOM_STATE,
        ),
    }


def probability_scores(model, x: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(x)[:, 1]
    if hasattr(model, "decision_function"):
        scores = model.decision_function(x)
        scores = np.asarray(scores, dtype=float)
        min_s, max_s = scores.min(), scores.max()
        if max_s - min_s < 1e-12:
            return np.full_like(scores, 0.5, dtype=float)
        return (scores - min_s) / (max_s - min_s)
    raise ValueError(f"Model {type(model).__name__} has neither predict_proba nor decision_function")


def compute_metrics(y_true: pd.Series, y_pred: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    try:
        roc = roc_auc_score(y_true, y_prob)
    except Exception:
        roc = np.nan

    try:
        pr = average_precision_score(y_true, y_prob)
    except Exception:
        pr = np.nan

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else np.nan
    fnr = fn / (fn + tp) if (fn + tp) > 0 else np.nan

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "roc_auc": roc,
        "pr_auc": pr,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "fpr": fpr,
        "fnr": fnr,
    }


def main() -> None:
    root = project_root_from_script()
    metrics_dir = root / "04_Results" / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    datasets = ["NSL-KDD", "UNSW-NB15", "CICIDS2017"]

    validation_rows: List[Dict[str, float]] = []
    test_rows: List[Dict[str, float]] = []

    for dataset in datasets:
        print(f"\n=== Processing {dataset} ===")
        x_train, y_train, x_val, y_val, x_test, y_test = load_dataset_splits(root, dataset)

        print(f"Train shape: {x_train.shape}, Val shape: {x_val.shape}, Test shape: {x_test.shape}")

        models = get_models()

        for model_name, model in models.items():
            print(f"Training {model_name} on {dataset}...")
            model.fit(x_train, y_train)

            val_pred = model.predict(x_val)
            val_prob = probability_scores(model, x_val)
            val_metrics = compute_metrics(y_val, val_pred, val_prob)
            validation_rows.append(
                {
                    "dataset": dataset,
                    "model": model_name,
                    "validation_accuracy": val_metrics["accuracy"],
                    "validation_precision": val_metrics["precision"],
                    "validation_recall": val_metrics["recall"],
                    "validation_f1": val_metrics["f1"],
                    "validation_roc_auc": val_metrics["roc_auc"],
                    "validation_pr_auc": val_metrics["pr_auc"],
                    "validation_fpr": val_metrics["fpr"],
                    "validation_fnr": val_metrics["fnr"],
                }
            )

            test_pred = model.predict(x_test)
            test_prob = probability_scores(model, x_test)
            tst_metrics = compute_metrics(y_test, test_pred, test_prob)
            test_rows.append(
                {
                    "dataset": dataset,
                    "model": model_name,
                    "test_accuracy": tst_metrics["accuracy"],
                    "test_precision": tst_metrics["precision"],
                    "test_recall": tst_metrics["recall"],
                    "test_f1": tst_metrics["f1"],
                    "test_roc_auc": tst_metrics["roc_auc"],
                    "test_pr_auc": tst_metrics["pr_auc"],
                    "tn": tst_metrics["tn"],
                    "fp": tst_metrics["fp"],
                    "fn": tst_metrics["fn"],
                    "tp": tst_metrics["tp"],
                    "test_fpr": tst_metrics["fpr"],
                    "test_fnr": tst_metrics["fnr"],
                }
            )

            print(
                f"  Val F1={val_metrics['f1']:.4f}, Test F1={tst_metrics['f1']:.4f}, "
                f"Test PR-AUC={tst_metrics['pr_auc']:.4f}, Test FNR={tst_metrics['fnr']:.4f}"
            )

    df_val = pd.DataFrame(validation_rows).sort_values(["dataset", "validation_f1"], ascending=[True, False])
    df_test = pd.DataFrame(test_rows).sort_values(["dataset", "test_f1"], ascending=[True, False])

    val_out = metrics_dir / "multidataset_validation_results_full.csv"
    test_out = metrics_dir / "multidataset_test_results_full.csv"

    df_val.to_csv(val_out, index=False)
    df_test.to_csv(test_out, index=False)

    print("\nSaved:")
    print(val_out)
    print(test_out)

    print("\nValidation summary:")
    print(df_val[["dataset", "model", "validation_f1", "validation_roc_auc", "validation_pr_auc"]].to_string(index=False))

    print("\nTest summary:")
    print(df_test[["dataset", "model", "test_f1", "test_roc_auc", "test_pr_auc", "test_fnr"]].to_string(index=False))


if __name__ == "__main__":
    main()