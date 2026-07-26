from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


# -----------------------------------
# Configuration
# -----------------------------------
STAGE2_WEIGHTS = {
    "core": 0.60,   # Stage-1 reliability-aware core score
    "E": 0.25,      # seed-wise SHAP stability
    "T": 0.15,      # statistical support coefficient
}


# -----------------------------------
# Helpers
# -----------------------------------
def project_root_from_script() -> Path:
    return Path(__file__).resolve().parents[2]


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    return pd.read_csv(path)


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


def weighted_geom(values: Dict[str, float], weights: Dict[str, float]) -> float:
    available = {k: v for k, v in values.items() if pd.notna(v)}
    if not available:
        return np.nan

    used_weights = {k: weights[k] for k in available}
    total_w = sum(used_weights.values())
    used_weights = {k: v / total_w for k, v in used_weights.items()}

    prod = 1.0
    for k, v in available.items():
        v = float(np.clip(v, 1e-6, 1.0))
        prod *= v ** used_weights[k]
    return prod


def significance_support_lookup(df_sig: pd.DataFrame) -> Dict[Tuple[str, str], float]:
    """
    Maps (dataset, model) -> statistical support coefficient T.
    T is intentionally ordinal rather than binary:
        1.00  = supported superiority by both tests
        0.75  = supported superiority by one test
        0.50  = neutral / no meaningful support either way
        0.30  = supported inferiority by one test
        0.15  = supported inferiority by both tests
    """
    lookup: Dict[Tuple[str, str], float] = {}

    if df_sig.empty:
        return lookup

    df = df_sig.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]

    for _, row in df.iterrows():
        dataset = normalize_dataset_name(row.get("dataset", ""))
        model_a = normalize_model_name(row.get("model_a", ""))
        model_b = normalize_model_name(row.get("model_b", ""))

        diff = float(row.get("mean_f1_difference_a_minus_b", 0.0))
        wilcoxon_p = row.get("wilcoxon_p_value", np.nan)
        mcnemar_p = row.get("mcnemar_p_value", np.nan)

        sig_count = 0
        if pd.notna(wilcoxon_p) and float(wilcoxon_p) < 0.05:
            sig_count += 1
        if pd.notna(mcnemar_p) and float(mcnemar_p) < 0.05:
            sig_count += 1

        if sig_count == 0:
            score_a, score_b = 0.50, 0.50
        else:
            if diff > 0:
                if sig_count == 2:
                    score_a, score_b = 1.00, 0.15
                else:
                    score_a, score_b = 0.75, 0.30
            elif diff < 0:
                if sig_count == 2:
                    score_a, score_b = 0.15, 1.00
                else:
                    score_a, score_b = 0.30, 0.75
            else:
                score_a, score_b = 0.50, 0.50

        lookup[(dataset, model_a)] = score_a
        lookup[(dataset, model_b)] = score_b

    return lookup


def optional_shift_lookup(df_holdout: pd.DataFrame, df_test_full: pd.DataFrame) -> Dict[Tuple[str, str], float]:
    """
    Optional auxiliary shift-robustness indicator.
    If the holdout audit file lacks a model column, we assume it corresponds to the
    audited CICIDS2017 LightGBM run from the current paper.
    This value is NOT used in Stage-2 scoring; it is only carried as diagnostic evidence.
    """
    lookup: Dict[Tuple[str, str], float] = {}

    if df_holdout.empty:
        return lookup

    holdout = df_holdout.copy()
    holdout.columns = [str(c).strip().lower().replace(" ", "_") for c in holdout.columns]

    if "f1" not in holdout.columns:
        return lookup

    dataset_name = "CICIDS2017"

    if "model" in holdout.columns:
        holdout["model"] = holdout["model"].map(normalize_model_name)
        grouped = holdout.groupby("model")["f1"].mean()
    else:
        grouped = pd.Series({"LightGBM": holdout["f1"].mean()})

    df_test = df_test_full.copy()
    df_test["dataset"] = df_test["dataset"].map(normalize_dataset_name)
    df_test["model"] = df_test["model"].map(normalize_model_name)

    for model_name, mean_holdout_f1 in grouped.items():
        pooled_row = df_test.loc[
            (df_test["dataset"] == dataset_name) & (df_test["model"] == model_name),
            "test_f1",
        ]
        if pooled_row.empty:
            continue
        pooled_f1 = float(pooled_row.iloc[0])
        if pooled_f1 <= 0:
            continue
        lookup[(dataset_name, model_name)] = float(np.clip(mean_holdout_f1 / pooled_f1, 0.0, 1.0))

    return lookup


# -----------------------------------
# Main
# -----------------------------------
def main() -> None:
    root = project_root_from_script()
    metrics_dir = root / "04_Results" / "metrics"

    shortlist_path = metrics_dir / "raise_ids_stage2_shortlist.csv"
    stability_path = metrics_dir / "raise_ids_shortlist_seed_stability_summary.csv"
    sig_path = metrics_dir / "statistical_significance_summary.csv"
    core_summary_path = metrics_dir / "raise_ids_core_top_model_summary.csv"
    full_test_path = metrics_dir / "multidataset_test_results_full.csv"
    holdout_path = metrics_dir / "cicids2017_multi_holdout_audit_results.csv"

    df_short = read_csv(shortlist_path)
    df_stab = read_csv(stability_path)
    df_sig = read_csv(sig_path) if sig_path.exists() else pd.DataFrame()
    df_core_summary = read_csv(core_summary_path)
    df_test_full = read_csv(full_test_path)
    df_holdout = read_csv(holdout_path) if holdout_path.exists() else pd.DataFrame()

    df_short["dataset"] = df_short["dataset"].map(normalize_dataset_name)
    df_short["model"] = df_short["model"].map(normalize_model_name)

    df_stab["dataset"] = df_stab["dataset"].map(normalize_dataset_name)
    df_stab["model"] = df_stab["model"].map(normalize_model_name)

    stability_lookup = {
        (row["dataset"], row["model"]): float(row["mean_pairwise_jaccard"])
        for _, row in df_stab.iterrows()
    }

    T_lookup = significance_support_lookup(df_sig)
    H_lookup = optional_shift_lookup(df_holdout, df_test_full)

    df = df_short.copy()
    df["E_seed_stability"] = [
        stability_lookup.get((d, m), np.nan) for d, m in zip(df["dataset"], df["model"])
    ]
    df["T_statistical_support"] = [
        T_lookup.get((d, m), 0.50) for d, m in zip(df["dataset"], df["model"])
    ]
    df["H_shift_auxiliary"] = [
        H_lookup.get((d, m), np.nan) for d, m in zip(df["dataset"], df["model"])
    ]

    # Stage-2 full score (Core + E + T)
    stage2_scores = []
    score_core_E = []
    score_core_T = []

    for _, row in df.iterrows():
        vals_full = {
            "core": row["raise_ids_core_score"],
            "E": row["E_seed_stability"],
            "T": row["T_statistical_support"],
        }
        vals_core_E = {
            "core": row["raise_ids_core_score"],
            "E": row["E_seed_stability"],
        }
        vals_core_T = {
            "core": row["raise_ids_core_score"],
            "T": row["T_statistical_support"],
        }

        stage2_scores.append(weighted_geom(vals_full, STAGE2_WEIGHTS))
        score_core_E.append(weighted_geom(vals_core_E, STAGE2_WEIGHTS))
        score_core_T.append(weighted_geom(vals_core_T, STAGE2_WEIGHTS))

    df["raise_ids_stage2_score"] = stage2_scores
    df["score_core_plus_E"] = score_core_E
    df["score_core_plus_T"] = score_core_T
    df["score_core_only"] = df["raise_ids_core_score"]

    # ranks
    df["rank_stage2"] = df.groupby("dataset")["raise_ids_stage2_score"].rank(ascending=False, method="dense")
    df["rank_core_plus_E"] = df.groupby("dataset")["score_core_plus_E"].rank(ascending=False, method="dense")
    df["rank_core_plus_T"] = df.groupby("dataset")["score_core_plus_T"].rank(ascending=False, method="dense")

    # top-model summaries
    summary_rows = []
    ablation_rows = []

    df_core_summary["dataset"] = df_core_summary["dataset"].map(normalize_dataset_name)

    for dataset, grp in df.groupby("dataset"):
        grp_stage2 = grp.sort_values(["raise_ids_stage2_score", "raise_ids_core_score"], ascending=[False, False])
        grp_coreE = grp.sort_values(["score_core_plus_E", "raise_ids_core_score"], ascending=[False, False])
        grp_coreT = grp.sort_values(["score_core_plus_T", "raise_ids_core_score"], ascending=[False, False])

        top_stage2 = grp_stage2.iloc[0]["model"]
        top_coreE = grp_coreE.iloc[0]["model"]
        top_coreT = grp_coreT.iloc[0]["model"]

        core_summary_row = df_core_summary.loc[df_core_summary["dataset"] == dataset].iloc[0]

        top_val = core_summary_row["top_model_by_validation_f1"]
        top_test = core_summary_row["top_model_by_test_f1"]
        top_core = core_summary_row["top_model_by_raise_ids_core"]

        summary_rows.append(
            {
                "dataset": dataset,
                "top_model_by_validation_f1": top_val,
                "top_model_by_test_f1": top_test,
                "top_model_by_raise_ids_core": top_core,
                "top_model_by_raise_ids_stage2": top_stage2,
                "core_vs_stage2_changed": top_core != top_stage2,
                "validation_vs_stage2_changed": top_val != top_stage2,
                "test_vs_stage2_changed": top_test != top_stage2,
            }
        )

        ablation_rows.append(
            {
                "dataset": dataset,
                "top_model_core_only": top_core,
                "top_model_core_plus_E": top_coreE,
                "top_model_core_plus_T": top_coreT,
                "top_model_stage2_full": top_stage2,
                "core_to_coreE_changed": top_core != top_coreE,
                "core_to_coreT_changed": top_core != top_coreT,
                "core_to_stage2_changed": top_core != top_stage2,
            }
        )

    df_summary = pd.DataFrame(summary_rows)
    df_ablation = pd.DataFrame(ablation_rows)

    # Save
    out_stage2 = metrics_dir / "raise_ids_stage2_refined_scores.csv"
    out_summary = metrics_dir / "raise_ids_stage2_top_model_summary.csv"
    out_ablation = metrics_dir / "raise_ids_stage2_ablation_summary.csv"

    df.sort_values(["dataset", "rank_stage2"]).to_csv(out_stage2, index=False)
    df_summary.to_csv(out_summary, index=False)
    df_ablation.to_csv(out_ablation, index=False)

    print("Saved:")
    print(out_stage2)
    print(out_summary)
    print(out_ablation)

    print("\nStage-2 top-model summary:")
    print(df_summary.to_string(index=False))

    print("\nAblation summary:")
    print(df_ablation.to_string(index=False))

    print("\nStage-2 ranking preview:")
    print(
        df.sort_values(["dataset", "rank_stage2"])[
            [
                "dataset",
                "model",
                "raise_ids_core_score",
                "E_seed_stability",
                "T_statistical_support",
                "H_shift_auxiliary",
                "raise_ids_stage2_score",
                "rank_stage2",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()