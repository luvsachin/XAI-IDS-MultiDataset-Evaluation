from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import argparse
import time

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------
DEFAULT_RANDOM_STATE = 42
DATASET_ORDER = ["NSL-KDD", "UNSW-NB15", "CICIDS2017"]


# ---------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------
def project_root_from_script() -> Path:
    return Path(__file__).resolve().parents[2]


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    return pd.read_csv(path)


def find_first_existing(dir_path: Path, names: List[str]) -> Optional[Path]:
    for name in names:
        p = dir_path / name
        if p.exists():
            return p
    return None


def dataset_aliases(dataset_name: str) -> List[str]:
    mapping = {
        "NSL-KDD": ["nsl-kdd", "nsl_kdd", "nslkdd", "nsl kdd"],
        "UNSW-NB15": ["unsw-nb15", "unsw_nb15", "unswnb15", "unsw nb15"],
        "CICIDS2017": ["cicids2017", "cic ids 2017", "cic-ids2017", "cic_ids2017"],
    }
    return mapping.get(dataset_name, [dataset_name.lower()])


def is_complete_processed_dir(dir_path: Path) -> bool:
    x_ok = all((dir_path / f).exists() for f in ["X_train_final.csv", "X_test_final.csv"])
    y_train = find_first_existing(dir_path, ["y_train_binary.csv", "y_train.csv"])
    y_test = find_first_existing(dir_path, ["y_test_binary.csv", "y_test.csv"])
    return x_ok and y_train is not None and y_test is not None


def find_processed_dataset_dir(root: Path, dataset_name: str) -> Path:
    search_root = root / "02_Data"
    if not search_root.exists():
        raise FileNotFoundError(f"Could not find 02_Data folder under: {root}")

    candidates: List[Path] = []
    for x_train_file in search_root.rglob("X_train_final.csv"):
        candidate_dir = x_train_file.parent
        if is_complete_processed_dir(candidate_dir):
            candidates.append(candidate_dir)

    if not candidates:
        raise FileNotFoundError(f"No processed dataset directories found under: {search_root}")

    aliases = dataset_aliases(dataset_name)
    scored: List[Tuple[int, Path]] = []

    for c in candidates:
        path_str = str(c).lower().replace("\\", "/")
        path_norm = path_str.replace(" ", "").replace("-", "").replace("_", "")

        score = 0
        for alias in aliases:
            alias_norm = alias.replace(" ", "").replace("-", "").replace("_", "")
            if alias_norm in path_norm:
                score += 10

        if "processed" in path_str:
            score += 2

        scored.append((score, c))

    scored.sort(key=lambda x: (-x[0], str(x[1])))
    best_score, best_dir = scored[0]

    if best_score <= 0:
        candidate_text = "\n".join(str(p) for _, p in scored)
        raise FileNotFoundError(
            f"Could not confidently match processed directory for dataset '{dataset_name}'.\n"
            f"Available candidates:\n{candidate_text}"
        )

    return best_dir


def load_train_test(root: Path, dataset_name: str):
    proc_dir = find_processed_dataset_dir(root, dataset_name)
    print(f"Using processed directory for {dataset_name}: {proc_dir}")

    x_train = read_csv(proc_dir / "X_train_final.csv")
    x_test = read_csv(proc_dir / "X_test_final.csv")

    y_train_path = find_first_existing(proc_dir, ["y_train_binary.csv", "y_train.csv"])
    y_test_path = find_first_existing(proc_dir, ["y_test_binary.csv", "y_test.csv"])

    if y_train_path is None or y_test_path is None:
        raise FileNotFoundError(f"Could not find y_train/y_test files in {proc_dir}")

    y_train = read_csv(y_train_path).iloc[:, 0].astype(int)
    y_test = read_csv(y_test_path).iloc[:, 0].astype(int)

    return x_train, y_train, x_test, y_test


# ---------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------
def compute_binary_metrics(y_true, y_pred, y_prob) -> dict:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    fpr = fp / (fp + tn) if (fp + tn) > 0 else np.nan
    fnr = fn / (fn + tp) if (fn + tp) > 0 else np.nan

    out = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_prob),
        "pr_auc": average_precision_score(y_true, y_prob),
        "fpr": fpr,
        "fnr": fnr,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }
    return out


def normalize_dataset_name(x: object) -> str:
    s = str(x).strip().lower().replace("_", " ").replace("-", " ")
    s = " ".join(s.split())
    if "nsl" in s and "kdd" in s:
        return "NSL-KDD"
    if "unsw" in s:
        return "UNSW-NB15"
    if "cic" in s:
        return "CICIDS2017"
    return str(x).strip()


def normalize_model_name(x: object) -> str:
    s = str(x).strip().lower().replace("_", " ").replace("-", " ")
    s = " ".join(s.split())
    mapping = {
        "lightgbm": "LightGBM",
        "xgboost": "XGBoost",
        "mlp": "MLP",
        "tuned mlp": "TunedMLP",
        "tunedmlp": "TunedMLP",
    }
    return mapping.get(s, str(x).strip())


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


# ---------------------------------------------------------------------
# LaTeX
# ---------------------------------------------------------------------
def write_latex_tuned_mlp_table(df_compare: pd.DataFrame, path: Path) -> None:
    lines = []

    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Tuned MLP neural baseline compared with the original lightweight MLP and the final RAISE-IDS winner. The tuned MLP uses a deeper $(256,128,64)$ architecture with early stopping.}")
    lines.append(r"\label{tab:raise_ids_tuned_mlp_baseline}")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{llrrrrrr}")
    lines.append(r"\hline")
    lines.append(r"\textbf{Dataset} & \textbf{Model} & \textbf{F1} & \textbf{PR-AUC} & \textbf{ROC-AUC} & \textbf{FPR} & \textbf{FNR} & \textbf{Training sec.} \\")
    lines.append(r"\hline")

    for _, row in df_compare.sort_values(["dataset_order", "display_order"]).iterrows():
        lines.append(
            f"{latex_escape(row['dataset'])} & "
            f"{latex_escape(row['model'])} & "
            f"{fmt(row['f1'])} & "
            f"{fmt(row['pr_auc'])} & "
            f"{fmt(row['roc_auc'])} & "
            f"{fmt(row['fpr'])} & "
            f"{fmt(row['fnr'])} & "
            f"{fmt(row['train_time_sec'], 2)} \\\\"
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
    parser.add_argument("--max-train-rows", type=int, default=150000)
    parser.add_argument("--hidden-layers", type=str, default="256,128,64")
    parser.add_argument("--max-iter", type=int, default=300)
    parser.add_argument("--random-state", type=int, default=DEFAULT_RANDOM_STATE)
    args = parser.parse_args()

    root = project_root_from_script()
    metrics_dir = root / "04_Results" / "metrics"
    tables_dir = root / "06_LaTeX" / "tables"

    metrics_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    hidden_layers = tuple(int(x.strip()) for x in args.hidden_layers.split(",") if x.strip())

    rows = []

    for dataset in DATASET_ORDER:
        print(f"\n=== Tuned MLP baseline: {dataset} ===")

        x_train, y_train, x_test, y_test = load_train_test(root, dataset)

        if args.max_train_rows > 0 and len(x_train) > args.max_train_rows:
            x_train_use = x_train.sample(args.max_train_rows, random_state=args.random_state)
            y_train_use = y_train.loc[x_train_use.index]
            x_train_use = x_train_use.reset_index(drop=True)
            y_train_use = y_train_use.reset_index(drop=True)
        else:
            x_train_use = x_train.reset_index(drop=True)
            y_train_use = y_train.reset_index(drop=True)

        x_test_use = x_test.reset_index(drop=True)
        y_test_use = y_test.reset_index(drop=True)

        print(f"Training rows used: {len(x_train_use)}, test rows used: {len(x_test_use)}")
        print(f"Hidden layers: {hidden_layers}, max_iter={args.max_iter}")

        model = Pipeline(
            steps=[
                ("scaler", StandardScaler(with_mean=True, with_std=True)),
                (
                    "mlp",
                    MLPClassifier(
                        hidden_layer_sizes=hidden_layers,
                        activation="relu",
                        solver="adam",
                        alpha=1e-4,
                        batch_size=256,
                        learning_rate="adaptive",
                        learning_rate_init=1e-3,
                        max_iter=args.max_iter,
                        early_stopping=True,
                        validation_fraction=0.10,
                        n_iter_no_change=15,
                        random_state=args.random_state,
                        verbose=False,
                    ),
                ),
            ]
        )

        start = time.time()
        model.fit(x_train_use, y_train_use)
        train_time = time.time() - start

        start_pred = time.time()
        y_prob = model.predict_proba(x_test_use)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)
        inference_time = time.time() - start_pred
        inference_ms = 1000.0 * inference_time / len(x_test_use)

        metrics = compute_binary_metrics(y_test_use, y_pred, y_prob)

        row = {
            "dataset": dataset,
            "model": "TunedMLP",
            "hidden_layers": str(hidden_layers),
            "train_rows_used": len(x_train_use),
            "test_rows_used": len(x_test_use),
            "train_time_sec": train_time,
            "inference_ms_per_sample": inference_ms,
            **metrics,
        }
        rows.append(row)

        print(
            f"TunedMLP: F1={metrics['f1']:.6f}, PR-AUC={metrics['pr_auc']:.6f}, "
            f"ROC-AUC={metrics['roc_auc']:.6f}, FPR={metrics['fpr']:.6f}, "
            f"FNR={metrics['fnr']:.6f}, train_sec={train_time:.2f}"
        )

    tuned_df = pd.DataFrame(rows)

    out_tuned = metrics_dir / "raise_ids_tuned_mlp_test_results.csv"
    tuned_df.to_csv(out_tuned, index=False)

    # Build comparison table with old MLP and final RAISE-IDS winners.
    full_test = read_csv(metrics_dir / "raise_ids_full_all_model_independent_test_table.csv")
    full_test["dataset"] = full_test["dataset"].map(normalize_dataset_name)
    full_test["model"] = full_test["model"].map(normalize_model_name)

    final_winners = read_csv(metrics_dir / "raise_ids_stage2_top_model_summary_formal_T.csv")
    final_winners["dataset"] = final_winners["dataset"].map(normalize_dataset_name)
    final_winners["formal_T_stage2_winner"] = final_winners["formal_T_stage2_winner"].map(normalize_model_name)

    compare_rows = []

    dataset_order = {d: i for i, d in enumerate(DATASET_ORDER)}

    for dataset in DATASET_ORDER:
        winner = final_winners.loc[
            final_winners["dataset"] == dataset, "formal_T_stage2_winner"
        ].iloc[0]

        # Final winner from all-model test table
        winner_row = full_test[
            (full_test["dataset"] == dataset) & (full_test["model"] == winner)
        ].iloc[0]

        compare_rows.append(
            {
                "dataset": dataset,
                "dataset_order": dataset_order[dataset],
                "display_order": 1,
                "model": f"RAISE-IDS winner ({winner})",
                "f1": winner_row["test_f1"],
                "pr_auc": winner_row["test_pr_auc"],
                "roc_auc": winner_row["test_roc_auc"],
                "fpr": winner_row["test_fpr"],
                "fnr": winner_row["test_fnr"],
                "train_time_sec": np.nan,
            }
        )

        old_mlp = full_test[(full_test["dataset"] == dataset) & (full_test["model"] == "MLP")]
        if not old_mlp.empty:
            old_row = old_mlp.iloc[0]
            compare_rows.append(
                {
                    "dataset": dataset,
                    "dataset_order": dataset_order[dataset],
                    "display_order": 2,
                    "model": "Original lightweight MLP",
                    "f1": old_row["test_f1"],
                    "pr_auc": old_row["test_pr_auc"],
                    "roc_auc": old_row["test_roc_auc"],
                    "fpr": old_row["test_fpr"],
                    "fnr": old_row["test_fnr"],
                    "train_time_sec": np.nan,
                }
            )

        tuned_row = tuned_df[tuned_df["dataset"] == dataset].iloc[0]
        compare_rows.append(
            {
                "dataset": dataset,
                "dataset_order": dataset_order[dataset],
                "display_order": 3,
                "model": "Tuned MLP",
                "f1": tuned_row["f1"],
                "pr_auc": tuned_row["pr_auc"],
                "roc_auc": tuned_row["roc_auc"],
                "fpr": tuned_row["fpr"],
                "fnr": tuned_row["fnr"],
                "train_time_sec": tuned_row["train_time_sec"],
            }
        )

    compare_df = pd.DataFrame(compare_rows)

    out_compare = metrics_dir / "raise_ids_tuned_mlp_comparison.csv"
    compare_df.to_csv(out_compare, index=False)

    latex_table = tables_dir / "table_raise_ids_tuned_mlp_baseline.tex"
    write_latex_tuned_mlp_table(compare_df, latex_table)

    print("\nSaved:")
    print(out_tuned)
    print(out_compare)
    print(latex_table)

    print("\nTuned MLP results:")
    print(tuned_df.to_string(index=False))

    print("\nTuned MLP comparison:")
    print(compare_df.sort_values(["dataset_order", "display_order"]).to_string(index=False))


if __name__ == "__main__":
    main()