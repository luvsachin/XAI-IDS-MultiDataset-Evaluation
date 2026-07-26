from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


# -------------------------
# Configuration
# -------------------------
LAMBDA_FNR = 0.70  # false negatives penalized more than false positives
CORE_WEIGHTS = {
    "Q": 0.45,  # predictive quality
    "O": 0.35,  # operational safety
    "G": 0.20,  # generalization consistency
}
TOP_K_SHORTLIST = 3


# -------------------------
# Helpers
# -------------------------
def project_root_from_script() -> Path:
    return Path(__file__).resolve().parents[2]


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    return pd.read_csv(path)


def geometric_mean_weighted(values: Dict[str, float], weights: Dict[str, float]) -> float:
    prod = 1.0
    total_w = sum(weights.values())
    for k, v in values.items():
        w = weights[k] / total_w
        v = float(np.clip(v, 1e-8, 1.0))
        prod *= v ** w
    return prod


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
        "logistic regression": "LogisticRegression",
        "logisticregression": "LogisticRegression",
        "random forest": "RandomForest",
        "randomforest": "RandomForest",
        "extra trees": "ExtraTrees",
        "extratrees": "ExtraTrees",
        "lightgbm": "LightGBM",
        "xgboost": "XGBoost",
        "mlp": "MLP",
    }
    return mapping.get(s, str(x).strip())


# -------------------------
# Main
# -------------------------
def main() -> None:
    root = project_root_from_script()
    metrics_dir = root / "04_Results" / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    val_path = metrics_dir / "multidataset_validation_results_full.csv"
    test_path = metrics_dir / "multidataset_test_results_full.csv"

    df_val = read_csv(val_path)
    df_test = read_csv(test_path)

    df_val["dataset"] = df_val["dataset"].map(normalize_dataset_name)
    df_test["dataset"] = df_test["dataset"].map(normalize_dataset_name)
    df_val["model"] = df_val["model"].map(normalize_model_name)
    df_test["model"] = df_test["model"].map(normalize_model_name)

    keep_val = ["dataset", "model", "validation_f1"]
    keep_test = ["dataset", "model", "test_f1", "test_pr_auc", "test_fpr", "test_fnr"]

    df = df_val[keep_val].merge(df_test[keep_test], on=["dataset", "model"], how="inner")

    # Q = sqrt(F1_test * PR_AUC_test)
    df["Q"] = np.sqrt(df["test_f1"] * df["test_pr_auc"])

    # O = 1 - (lambda*FNR + (1-lambda)*FPR)
    df["O"] = 1.0 - (LAMBDA_FNR * df["test_fnr"] + (1.0 - LAMBDA_FNR) * df["test_fpr"])

    # G = 1 - max(0, F1_val - F1_test)
    df["G"] = 1.0 - np.maximum(0.0, df["validation_f1"] - df["test_f1"])

    # Core score
    core_scores: List[float] = []
    for _, row in df.iterrows():
        vals = {"Q": row["Q"], "O": row["O"], "G": row["G"]}
        core_scores.append(geometric_mean_weighted(vals, CORE_WEIGHTS))
    df["raise_ids_core_score"] = core_scores

    # Ranks
    df["rank_validation_f1"] = df.groupby("dataset")["validation_f1"].rank(ascending=False, method="dense")
    df["rank_test_f1"] = df.groupby("dataset")["test_f1"].rank(ascending=False, method="dense")
    df["rank_raise_ids_core"] = df.groupby("dataset")["raise_ids_core_score"].rank(ascending=False, method="dense")

    # Rank-shift summary
    df["shift_val_to_test"] = df["rank_test_f1"] - df["rank_validation_f1"]
    df["shift_test_to_core"] = df["rank_raise_ids_core"] - df["rank_test_f1"]
    df["shift_val_to_core"] = df["rank_raise_ids_core"] - df["rank_validation_f1"]

    # Top model by each rule
    summary_rows = []
    shortlist_rows = []

    for dataset, grp in df.groupby("dataset"):
        grp_sorted_core = grp.sort_values(["raise_ids_core_score", "test_f1"], ascending=[False, False]).copy()
        grp_sorted_val = grp.sort_values(["validation_f1", "test_f1"], ascending=[False, False]).copy()
        grp_sorted_test = grp.sort_values(["test_f1", "test_pr_auc"], ascending=[False, False]).copy()

        top_val = grp_sorted_val.iloc[0]["model"]
        top_test = grp_sorted_test.iloc[0]["model"]
        top_core = grp_sorted_core.iloc[0]["model"]

        summary_rows.append(
            {
                "dataset": dataset,
                "top_model_by_validation_f1": top_val,
                "top_model_by_test_f1": top_test,
                "top_model_by_raise_ids_core": top_core,
                "validation_vs_test_changed": top_val != top_test,
                "test_vs_core_changed": top_test != top_core,
                "validation_vs_core_changed": top_val != top_core,
            }
        )

        shortlist = grp_sorted_core.head(TOP_K_SHORTLIST).copy()
        shortlist["shortlist_rank"] = range(1, len(shortlist) + 1)
        shortlist_rows.append(shortlist)

    df_summary = pd.DataFrame(summary_rows)
    df_shortlist = pd.concat(shortlist_rows, ignore_index=True)

    # Save outputs
    out_core = metrics_dir / "raise_ids_core_scores.csv"
    out_shift = metrics_dir / "raise_ids_core_rank_shift.csv"
    out_summary = metrics_dir / "raise_ids_core_top_model_summary.csv"
    out_shortlist = metrics_dir / "raise_ids_stage2_shortlist.csv"

    df.sort_values(["dataset", "rank_raise_ids_core"]).to_csv(out_core, index=False)
    df.sort_values(["dataset", "rank_raise_ids_core"]).to_csv(out_shift, index=False)
    df_summary.to_csv(out_summary, index=False)
    df_shortlist.to_csv(out_shortlist, index=False)

    print("Saved:")
    print(out_core)
    print(out_shift)
    print(out_summary)
    print(out_shortlist)

    print("\nTop-model summary:")
    print(df_summary.to_string(index=False))

    print("\nCore ranking preview:")
    print(
        df.sort_values(["dataset", "rank_raise_ids_core"])[
            [
                "dataset",
                "model",
                "validation_f1",
                "test_f1",
                "test_pr_auc",
                "test_fnr",
                "Q",
                "O",
                "G",
                "raise_ids_core_score",
                "rank_validation_f1",
                "rank_test_f1",
                "rank_raise_ids_core",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()