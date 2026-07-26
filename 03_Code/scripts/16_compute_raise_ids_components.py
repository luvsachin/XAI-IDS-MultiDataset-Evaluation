from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# =========================
# Configuration
# =========================
LAMBDA_FNR = 0.70  # operational safety penalty weight for false negatives
BASE_WEIGHTS = {
    "Q": 0.25,  # predictive quality
    "O": 0.20,  # operational safety
    "G": 0.15,  # generalization consistency
    "E": 0.15,  # seed stability
    "U": 0.10,  # subsample robustness
    "H": 0.10,  # shift robustness
    "T": 0.05,  # statistical support
}


# =========================
# Helpers
# =========================
def project_root_from_script() -> Path:
    return Path(__file__).resolve().parents[2]


def ensure_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")


def read_csv(path: Path) -> pd.DataFrame:
    ensure_file(path)
    return pd.read_csv(path)


def normalize_label(x: object) -> str:
    s = str(x).strip()
    s = s.replace("_", " ").replace("-", " ")
    s = " ".join(s.split())
    return s.lower()


def normalize_dataset_name(x: object) -> str:
    s = normalize_label(x)
    if "nsl" in s and "kdd" in s:
        return "NSL-KDD"
    if "unsw" in s:
        return "UNSW-NB15"
    if "cic" in s:
        return "CICIDS2017"
    return str(x).strip()


def normalize_model_name(x: object) -> str:
    s = normalize_label(x)
    mapping = {
        "lightgbm": "LightGBM",
        "xgboost": "XGBoost",
        "randomforest": "RandomForest",
        "random forest": "RandomForest",
        "extremely randomized trees": "ExtraTrees",
        "extra trees": "ExtraTrees",
        "extratrees": "ExtraTrees",
        "logisticregression": "LogisticRegression",
        "logistic regression": "LogisticRegression",
        "mlp": "MLP",
        "multi layer perceptron": "MLP",
    }
    s_compact = s.replace(" ", "")
    if s in mapping:
        return mapping[s]
    if s_compact in mapping:
        return mapping[s_compact]
    return str(x).strip()


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [normalize_label(c) for c in out.columns]
    return out


def find_first_existing(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    cols = set(df.columns)
    for c in candidates:
        if normalize_label(c) in cols:
            return normalize_label(c)
    return None


def safe_series(df: pd.DataFrame, candidates: List[str], default: Optional[float] = None) -> pd.Series:
    c = find_first_existing(df, candidates)
    if c is None:
        if default is None:
            return pd.Series([np.nan] * len(df), index=df.index)
        return pd.Series([default] * len(df), index=df.index)
    return pd.to_numeric(df[c], errors="coerce")


def geometric_mean_weighted(component_dict: Dict[str, float], weights: Dict[str, float]) -> float:
    available = {k: v for k, v in component_dict.items() if pd.notna(v)}
    if not available:
        return np.nan

    used_weights = {k: weights[k] for k in available}
    total_w = sum(used_weights.values())
    used_weights = {k: v / total_w for k, v in used_weights.items()}

    prod = 1.0
    for k, v in available.items():
        v_clipped = float(np.clip(v, 1e-8, 1.0))
        prod *= v_clipped ** used_weights[k]
    return prod


def make_significance_lookup(df_sig: pd.DataFrame) -> Dict[Tuple[str, str], float]:
    """
    Returns mapping (dataset, model) -> T in [0,1].
    Default for models not in significance comparison is 0.5.
    """
    lookup: Dict[Tuple[str, str], float] = {}

    if df_sig.empty:
        return lookup

    for _, row in df_sig.iterrows():
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
            score_a, score_b = 0.5, 0.5
        else:
            if diff > 0:
                if sig_count == 2:
                    score_a, score_b = 1.0, 0.0
                else:
                    score_a, score_b = 0.75, 0.25
            elif diff < 0:
                if sig_count == 2:
                    score_a, score_b = 0.0, 1.0
                else:
                    score_a, score_b = 0.25, 0.75
            else:
                score_a, score_b = 0.5, 0.5

        lookup[(dataset, model_a)] = score_a
        lookup[(dataset, model_b)] = score_b

    return lookup


def compute_shift_lookup(
    df_holdout: pd.DataFrame,
    df_best_models: pd.DataFrame,
    df_test: pd.DataFrame,
) -> Dict[Tuple[str, str], float]:
    """
    Currently supports CICIDS2017 multi-holdout file for the best audited model.
    If model column is absent in holdout audit, assigns the shift robustness to the best model on CICIDS2017.
    """
    lookup: Dict[Tuple[str, str], float] = {}

    if df_holdout.empty:
        return lookup

    holdout = df_holdout.copy()
    holdout.columns = [normalize_label(c) for c in holdout.columns]

    dataset_col = find_first_existing(holdout, ["dataset"])
    model_col = find_first_existing(holdout, ["model"])
    f1_col = find_first_existing(holdout, ["f1", "test_f1", "holdout_f1"])

    if f1_col is None:
        return lookup

    # infer dataset if absent
    if dataset_col is None:
        holdout["dataset"] = "CICIDS2017"
        dataset_col = "dataset"

    holdout["dataset"] = holdout[dataset_col].map(normalize_dataset_name)

    if model_col is None:
        # infer best CICIDS2017 model from best-models file
        df_best_tmp = df_best_models.copy()
        df_best_tmp.columns = [normalize_label(c) for c in df_best_tmp.columns]
        dcol = find_first_existing(df_best_tmp, ["dataset"])
        mcol = find_first_existing(df_best_tmp, ["best_model", "model"])
        if dcol is None or mcol is None:
            return lookup

        cic_best = df_best_tmp.loc[df_best_tmp[dcol].map(normalize_dataset_name) == "CICIDS2017", mcol]
        if cic_best.empty:
            return lookup
        inferred_model = normalize_model_name(cic_best.iloc[0])
        holdout["model"] = inferred_model
        model_col = "model"

    holdout["model"] = holdout[model_col].map(normalize_model_name)

    df_test_tmp = df_test.copy()
    df_test_tmp["dataset"] = df_test_tmp["dataset"].map(normalize_dataset_name)
    df_test_tmp["model"] = df_test_tmp["model"].map(normalize_model_name)

    for (dataset, model), grp in holdout.groupby(["dataset", "model"]):
        pooled = df_test_tmp.loc[
            (df_test_tmp["dataset"] == dataset) & (df_test_tmp["model"] == model),
            "test_f1",
        ]
        if pooled.empty or pooled.iloc[0] <= 0:
            continue

        pooled_f1 = float(pooled.iloc[0])
        holdout_f1 = pd.to_numeric(grp[f1_col], errors="coerce").dropna()
        if holdout_f1.empty:
            continue

        ratio = np.clip((holdout_f1 / pooled_f1).mean(), 0.0, 1.0)
        lookup[(dataset, model)] = float(ratio)

    return lookup


# =========================
# Main
# =========================
def main() -> None:
    root = project_root_from_script()
    metrics_dir = root / "04_Results" / "metrics"
    out_dir = metrics_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    test_path = metrics_dir / "multidataset_test_results.csv"
    val_path = metrics_dir / "multidataset_validation_results.csv"
    best_path = metrics_dir / "multidataset_best_models.csv"
    sig_path = metrics_dir / "statistical_significance_summary.csv"

    seed_ext_path = metrics_dir / "shap_seed_stability_summary_extended.csv"
    seed_base_path = metrics_dir / "shap_seed_stability_summary.csv"
    subsample_path = metrics_dir / "shap_subsample_robustness_summary.csv"
    holdout_path = metrics_dir / "cicids2017_multi_holdout_audit_results.csv"

    df_test_raw = read_csv(test_path)
    df_val_raw = read_csv(val_path)
    df_best_raw = read_csv(best_path)
    df_sig_raw = read_csv(sig_path) if sig_path.exists() else pd.DataFrame()
    df_seed_raw = read_csv(seed_ext_path) if seed_ext_path.exists() else read_csv(seed_base_path)
    df_sub_raw = read_csv(subsample_path) if subsample_path.exists() else pd.DataFrame()
    df_holdout_raw = read_csv(holdout_path) if holdout_path.exists() else pd.DataFrame()

    # normalize test
    df_test = normalize_columns(df_test_raw)
    dataset_col = find_first_existing(df_test, ["dataset"])
    model_col = find_first_existing(df_test, ["model"])
    if dataset_col is None or model_col is None:
        raise ValueError("multidataset_test_results.csv must contain dataset and model columns.")

    df_test["dataset"] = df_test[dataset_col].map(normalize_dataset_name)
    df_test["model"] = df_test[model_col].map(normalize_model_name)

    df_test["test_f1"] = safe_series(df_test, ["test_f1", "f1"])
    df_test["test_pr_auc"] = safe_series(df_test, ["test_pr_auc", "pr_auc"])
    df_test["test_recall"] = safe_series(df_test, ["test_recall", "recall"])
    df_test["test_fpr"] = safe_series(df_test, ["test_fpr", "fpr"])
    df_test["test_fnr"] = safe_series(df_test, ["test_fnr", "fnr"])

    # derive FNR if absent and recall is available
    mask_missing_fnr = df_test["test_fnr"].isna() & df_test["test_recall"].notna()
    df_test.loc[mask_missing_fnr, "test_fnr"] = 1.0 - df_test.loc[mask_missing_fnr, "test_recall"]

    # normalize val
    df_val = normalize_columns(df_val_raw)
    dcol_v = find_first_existing(df_val, ["dataset"])
    mcol_v = find_first_existing(df_val, ["model"])
    if dcol_v is None or mcol_v is None:
        raise ValueError("multidataset_validation_results.csv must contain dataset and model columns.")

    df_val["dataset"] = df_val[dcol_v].map(normalize_dataset_name)
    df_val["model"] = df_val[mcol_v].map(normalize_model_name)
    df_val["val_f1"] = safe_series(df_val, ["validation_f1", "val_f1", "f1"])

    # reduce validation table to best F1 per dataset-model if multiple rows exist
    df_val = (
        df_val[["dataset", "model", "val_f1"]]
        .dropna(subset=["val_f1"])
        .groupby(["dataset", "model"], as_index=False)["val_f1"]
        .max()
    )

    # seed stability
    df_seed = normalize_columns(df_seed_raw)
    dcol_s = find_first_existing(df_seed, ["dataset"])
    mcol_s = find_first_existing(df_seed, ["model"])
    jcol_s = find_first_existing(
        df_seed,
        [
            "mean_pairwise_jaccard",
            "mean_pairwise_jaccard_excluding_diagonal",
            "jaccard_mean",
        ],
    )
    if dcol_s is None or mcol_s is None or jcol_s is None:
        raise ValueError("Seed stability summary file is missing required dataset/model/jaccard columns.")
    df_seed["dataset"] = df_seed[dcol_s].map(normalize_dataset_name)
    df_seed["model"] = df_seed[mcol_s].map(normalize_model_name)
    df_seed["seed_stability"] = pd.to_numeric(df_seed[jcol_s], errors="coerce")
    df_seed = (
        df_seed[["dataset", "model", "seed_stability"]]
        .dropna()
        .groupby(["dataset", "model"], as_index=False)["seed_stability"]
        .mean()
    )

    # subsample robustness (optional)
    if not df_sub_raw.empty:
        df_sub = normalize_columns(df_sub_raw)
        dcol_u = find_first_existing(df_sub, ["dataset"])
        mcol_u = find_first_existing(df_sub, ["model"])
        jcol_u = find_first_existing(
            df_sub,
            [
                "mean_pairwise_jaccard",
                "mean_pairwise_jaccard_excluding_diagonal",
                "jaccard_mean",
                "subsample_jaccard_mean",
            ],
        )
        if dcol_u is not None and mcol_u is not None and jcol_u is not None:
            df_sub["dataset"] = df_sub[dcol_u].map(normalize_dataset_name)
            df_sub["model"] = df_sub[mcol_u].map(normalize_model_name)
            df_sub["subsample_robustness"] = pd.to_numeric(df_sub[jcol_u], errors="coerce")
            df_sub = (
                df_sub[["dataset", "model", "subsample_robustness"]]
                .dropna()
                .groupby(["dataset", "model"], as_index=False)["subsample_robustness"]
                .mean()
            )
        else:
            df_sub = pd.DataFrame(columns=["dataset", "model", "subsample_robustness"])
    else:
        df_sub = pd.DataFrame(columns=["dataset", "model", "subsample_robustness"])

    # significance lookup
    if not df_sig_raw.empty:
        df_sig = normalize_columns(df_sig_raw)
        sig_lookup = make_significance_lookup(df_sig)
    else:
        sig_lookup = {}

    # shift robustness lookup
    shift_lookup = compute_shift_lookup(df_holdout_raw, df_best_raw, df_test)

    # merge everything
    df = df_test.merge(df_val, on=["dataset", "model"], how="left")
    df = df.merge(df_seed, on=["dataset", "model"], how="left")
    df = df.merge(df_sub, on=["dataset", "model"], how="left")

    # components
    df["Q"] = np.sqrt(df["test_f1"] * df["test_pr_auc"])

    # Operational safety
    df["operational_mode"] = np.where(df["test_fpr"].notna(), "fnr_fpr", "fnr_only")
    df["O"] = np.where(
        df["test_fpr"].notna() & df["test_fnr"].notna(),
        1.0 - (LAMBDA_FNR * df["test_fnr"] + (1.0 - LAMBDA_FNR) * df["test_fpr"]),
        np.where(
            df["test_fnr"].notna(),
            1.0 - df["test_fnr"],
            np.nan,
        ),
    )

    # Generalization consistency
    df["G"] = 1.0 - np.maximum(0.0, df["val_f1"] - df["test_f1"])

    # Explanation stability
    df["E"] = df["seed_stability"]
    df["U"] = df["subsample_robustness"]

    # Shift robustness
    df["H"] = [
        shift_lookup.get((normalize_dataset_name(d), normalize_model_name(m)), np.nan)
        for d, m in zip(df["dataset"], df["model"])
    ]

    # Statistical support
    df["T"] = [
        sig_lookup.get((normalize_dataset_name(d), normalize_model_name(m)), 0.5)
        for d, m in zip(df["dataset"], df["model"])
    ]

    # final score
    component_cols = ["Q", "O", "G", "E", "U", "H", "T"]
    component_notes = []
    raise_scores = []

    for _, row in df.iterrows():
        comp = {c: row[c] for c in component_cols}
        available = [c for c, v in comp.items() if pd.notna(v)]
        component_notes.append(",".join(available))
        raise_scores.append(geometric_mean_weighted(comp, BASE_WEIGHTS))

    df["raise_ids_components_used"] = component_notes
    df["raise_ids_score"] = raise_scores

    # keep useful columns
    keep_cols = [
        "dataset",
        "model",
        "test_f1",
        "test_pr_auc",
        "test_recall",
        "test_fpr",
        "test_fnr",
        "val_f1",
        "seed_stability",
        "subsample_robustness",
        "Q",
        "O",
        "G",
        "E",
        "U",
        "H",
        "T",
        "operational_mode",
        "raise_ids_components_used",
        "raise_ids_score",
    ]
    df_out = df[keep_cols].copy()

    # sort
    df_out = df_out.sort_values(["dataset", "raise_ids_score", "test_f1"], ascending=[True, False, False])

    # save component scores
    component_path = out_dir / "raise_ids_component_scores.csv"
    df_out.to_csv(component_path, index=False)

    # save simple ranking per dataset
    ranking = df_out.copy()
    ranking["raise_ids_rank_within_dataset"] = ranking.groupby("dataset")["raise_ids_score"].rank(
        ascending=False, method="dense"
    )
    ranking_path = out_dir / "raise_ids_dataset_ranking.csv"
    ranking.to_csv(ranking_path, index=False)

    print("Saved:")
    print(component_path)
    print(ranking_path)
    print("\nTop rows:")
    print(
        ranking[
            ["dataset", "model", "raise_ids_score", "raise_ids_rank_within_dataset", "Q", "O", "G", "E", "U", "H", "T"]
        ]
        .head(12)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()