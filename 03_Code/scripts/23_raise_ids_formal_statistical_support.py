from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------
ALPHA = 0.05

# Practical-effect threshold for F1 difference.
# 0.0005 = 0.05 percentage-point F1 difference.
# This prevents statistically detectable but operationally negligible
# differences from being over-weighted in T.
MIN_PRACTICAL_F1_DELTA = 0.0005

STAGE2_WEIGHTS = {
    "core": 0.60,
    "E": 0.25,
    "T": 0.15,
}


# ---------------------------------------------------------------------
# Path and general helpers
# ---------------------------------------------------------------------
def project_root_from_script() -> Path:
    return Path(__file__).resolve().parents[2]


def read_csv(path: Path, required: bool = True) -> pd.DataFrame:
    if not path.exists():
        if required:
            raise FileNotFoundError(f"Missing required file: {path}")
        return pd.DataFrame()
    return pd.read_csv(path)


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [
        str(c).strip().lower().replace(" ", "_").replace("-", "_")
        for c in out.columns
    ]
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


def find_col(df: pd.DataFrame, candidates: List[str], required: bool = True) -> Optional[str]:
    cols = list(df.columns)

    for c in candidates:
        if c in cols:
            return c

    # fallback: all tokens contained
    for c in candidates:
        tokens = c.split("_")
        for col in cols:
            if all(t in col for t in tokens):
                return col

    if required:
        raise KeyError(
            f"Could not find any of the candidate columns {candidates}. "
            f"Available columns: {cols}"
        )
    return None


def safe_float(x: object, default: float = np.nan) -> float:
    try:
        if pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default


def holm_adjust(p_values: List[float]) -> List[float]:
    """
    Holm-Bonferroni adjusted p-values.
    Missing p-values remain NaN.
    """
    p = np.array([np.nan if pd.isna(v) else float(v) for v in p_values], dtype=float)
    adjusted = np.full_like(p, np.nan, dtype=float)

    valid_idx = np.where(~np.isnan(p))[0]
    if len(valid_idx) == 0:
        return adjusted.tolist()

    valid_p = p[valid_idx]
    order = np.argsort(valid_p)
    sorted_idx = valid_idx[order]
    sorted_p = valid_p[order]

    m = len(sorted_p)
    raw_adj = np.array([(m - i) * sorted_p[i] for i in range(m)], dtype=float)

    # Holm adjusted p-values must be monotone non-decreasing in sorted order
    monotone_adj = np.maximum.accumulate(raw_adj)
    monotone_adj = np.minimum(monotone_adj, 1.0)

    for idx, adj in zip(sorted_idx, monotone_adj):
        adjusted[idx] = adj

    return adjusted.tolist()


def weighted_geom(values: Dict[str, float], weights: Dict[str, float]) -> float:
    available = {k: v for k, v in values.items() if pd.notna(v)}
    if not available:
        return np.nan

    used_weights = {k: weights[k] for k in available}
    total_w = sum(used_weights.values())
    if total_w <= 0:
        return np.nan

    prod = 1.0
    for k, v in available.items():
        w = used_weights[k] / total_w
        v = float(np.clip(v, 1e-6, 1.0))
        prod *= v ** w

    return float(prod)


# ---------------------------------------------------------------------
# T-coefficient formal mapping
# ---------------------------------------------------------------------
def assign_pairwise_t(
    diff_a_minus_b: float,
    wilcoxon_p_holm: float,
    mcnemar_p_holm: float,
    alpha: float = ALPHA,
    min_practical_delta: float = MIN_PRACTICAL_F1_DELTA,
) -> Tuple[float, float, str, int, bool]:
    """
    Formal pairwise T mapping.

    Let delta = mean_F1(A) - mean_F1(B).

    Step 1: Practical effect filter
        If |delta| < min_practical_delta, both models receive T = 0.50.
        This avoids rewarding statistically detectable but practically negligible
        differences.

    Step 2: Corrected statistical evidence
        Count how many corrected tests support a non-negligible difference:
        Wilcoxon-Holm and McNemar-Holm.

    Step 3: Directional assignment
        If A is favored:
            both tests significant   -> A=1.00, B=0.15
            one test significant     -> A=0.75, B=0.30
            no test significant      -> A=0.50, B=0.50
        If B is favored, reverse the scores.

    Returns:
        T_A, T_B, evidence_label, significant_test_count, practical_effect
    """
    diff = safe_float(diff_a_minus_b, default=0.0)
    practical = abs(diff) >= min_practical_delta

    wil_sig = pd.notna(wilcoxon_p_holm) and float(wilcoxon_p_holm) < alpha
    mcn_sig = pd.notna(mcnemar_p_holm) and float(mcnemar_p_holm) < alpha
    sig_count = int(wil_sig) + int(mcn_sig)

    if not practical:
        if sig_count > 0:
            return (
                0.50,
                0.50,
                "statistically_detectable_but_practically_negligible",
                sig_count,
                False,
            )
        return (0.50, 0.50, "neutral_no_practical_effect", sig_count, False)

    if sig_count == 0:
        return (0.50, 0.50, "neutral_no_corrected_statistical_support", sig_count, True)

    a_favored = diff > 0

    if sig_count == 2:
        if a_favored:
            return (1.00, 0.15, "A_favored_by_both_corrected_tests", sig_count, True)
        return (0.15, 1.00, "B_favored_by_both_corrected_tests", sig_count, True)

    # sig_count == 1
    if a_favored:
        return (0.75, 0.30, "A_favored_by_one_corrected_test", sig_count, True)
    return (0.30, 0.75, "B_favored_by_one_corrected_test", sig_count, True)


def write_latex_t_mapping_table(path: Path) -> None:
    text = r"""\begin{table}[t]
\centering
\caption{Formal mapping from corrected statistical evidence to the RAISE-IDS statistical-support coefficient $T$. The mapping is applied only when the absolute F1 difference exceeds the practical-effect threshold $\delta_{\min}$. Otherwise, both models receive neutral support.}
\label{tab:raise_ids_t_mapping}
\small
\begin{tabular}{p{6.1cm}cc}
\hline
\textbf{Evidence condition} & \textbf{Favored model} & \textbf{Comparator} \\
\hline
Favored by both Holm-corrected Wilcoxon and McNemar tests, with $|\Delta F1| \geq \delta_{\min}$ & 1.00 & 0.15 \\
Favored by one corrected test, with $|\Delta F1| \geq \delta_{\min}$ & 0.75 & 0.30 \\
No corrected statistical support or statistically indistinguishable & 0.50 & 0.50 \\
Statistically detectable but $|\Delta F1| < \delta_{\min}$ & 0.50 & 0.50 \\
\hline
\end{tabular}
\end{table}
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


# ---------------------------------------------------------------------
# Main computation
# ---------------------------------------------------------------------
def main() -> None:
    root = project_root_from_script()
    metrics_dir = root / "04_Results" / "metrics"
    tables_dir = root / "06_LaTeX" / "tables"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    sig_path = metrics_dir / "statistical_significance_summary.csv"
    stage2_path = metrics_dir / "raise_ids_stage2_refined_scores.csv"
    core_summary_path = metrics_dir / "raise_ids_stage2_top_model_summary.csv"

    df_sig_raw = read_csv(sig_path, required=True)
    df_stage2_raw = read_csv(stage2_path, required=True)
    df_core_summary_raw = read_csv(core_summary_path, required=False)

    df_sig = normalize_columns(df_sig_raw)
    df_stage2 = df_stage2_raw.copy()
    df_core_summary = df_core_summary_raw.copy()

    # Resolve columns robustly
    dataset_col = find_col(df_sig, ["dataset"])
    model_a_col = find_col(df_sig, ["model_a", "modela", "model_1", "model1"])
    model_b_col = find_col(df_sig, ["model_b", "modelb", "model_2", "model2"])

    a_f1_col = find_col(
        df_sig,
        ["a_mean_f1", "model_a_mean_f1", "mean_f1_a", "mean_f1_model_a"],
        required=False,
    )
    b_f1_col = find_col(
        df_sig,
        ["b_mean_f1", "model_b_mean_f1", "mean_f1_b", "mean_f1_model_b"],
        required=False,
    )
    diff_col = find_col(
        df_sig,
        [
            "mean_f1_difference_a_minus_b",
            "mean_f1_diff_a_minus_b",
            "delta_f1",
            "diff_f1",
            "f1_difference",
        ],
        required=False,
    )

    wilcoxon_col = find_col(
        df_sig,
        ["wilcoxon_p_value", "wilcoxon_p", "wilcoxon_pvalue", "wilcoxon"],
        required=False,
    )
    mcnemar_col = find_col(
        df_sig,
        ["mcnemar_p_value", "mcnemar_p", "mcnemar_pvalue", "mcnemar"],
        required=False,
    )

    if diff_col is None and (a_f1_col is None or b_f1_col is None):
        raise KeyError(
            "Could not resolve F1 difference. Need either a difference column or A/B mean F1 columns."
        )

    # Normalize key values
    df_sig["_dataset"] = df_sig[dataset_col].map(normalize_dataset_name)
    df_sig["_model_a"] = df_sig[model_a_col].map(normalize_model_name)
    df_sig["_model_b"] = df_sig[model_b_col].map(normalize_model_name)

    if a_f1_col is not None:
        df_sig["_a_mean_f1"] = df_sig[a_f1_col].map(safe_float)
    else:
        df_sig["_a_mean_f1"] = np.nan

    if b_f1_col is not None:
        df_sig["_b_mean_f1"] = df_sig[b_f1_col].map(safe_float)
    else:
        df_sig["_b_mean_f1"] = np.nan

    if diff_col is not None:
        df_sig["_diff_a_minus_b"] = df_sig[diff_col].map(safe_float)
    else:
        df_sig["_diff_a_minus_b"] = df_sig["_a_mean_f1"] - df_sig["_b_mean_f1"]

    if wilcoxon_col is not None:
        df_sig["_wilcoxon_p"] = df_sig[wilcoxon_col].map(safe_float)
    else:
        df_sig["_wilcoxon_p"] = np.nan

    if mcnemar_col is not None:
        df_sig["_mcnemar_p"] = df_sig[mcnemar_col].map(safe_float)
    else:
        df_sig["_mcnemar_p"] = np.nan

    # Holm corrections across reported comparisons
    df_sig["_wilcoxon_p_holm"] = holm_adjust(df_sig["_wilcoxon_p"].tolist())
    df_sig["_mcnemar_p_holm"] = holm_adjust(df_sig["_mcnemar_p"].tolist())

    # Assign formal pairwise T
    pair_rows = []
    model_support_rows = []

    for _, row in df_sig.iterrows():
        t_a, t_b, label, sig_count, practical = assign_pairwise_t(
            diff_a_minus_b=row["_diff_a_minus_b"],
            wilcoxon_p_holm=row["_wilcoxon_p_holm"],
            mcnemar_p_holm=row["_mcnemar_p_holm"],
        )

        dataset = row["_dataset"]
        model_a = row["_model_a"]
        model_b = row["_model_b"]

        pair_rows.append(
            {
                "dataset": dataset,
                "model_a": model_a,
                "model_b": model_b,
                "a_mean_f1": row["_a_mean_f1"],
                "b_mean_f1": row["_b_mean_f1"],
                "delta_f1_a_minus_b": row["_diff_a_minus_b"],
                "abs_delta_f1": abs(row["_diff_a_minus_b"]),
                "min_practical_f1_delta": MIN_PRACTICAL_F1_DELTA,
                "wilcoxon_p_raw": row["_wilcoxon_p"],
                "mcnemar_p_raw": row["_mcnemar_p"],
                "wilcoxon_p_holm": row["_wilcoxon_p_holm"],
                "mcnemar_p_holm": row["_mcnemar_p_holm"],
                "significant_corrected_test_count": sig_count,
                "practical_effect": practical,
                "evidence_label": label,
                "T_model_a": t_a,
                "T_model_b": t_b,
            }
        )

        model_support_rows.append(
            {
                "dataset": dataset,
                "model": model_a,
                "comparator": model_b,
                "pairwise_T": t_a,
                "evidence_label": label,
            }
        )
        model_support_rows.append(
            {
                "dataset": dataset,
                "model": model_b,
                "comparator": model_a,
                "pairwise_T": t_b,
                "evidence_label": label,
            }
        )

    df_pairs = pd.DataFrame(pair_rows)
    df_model_support = pd.DataFrame(model_support_rows)

    # Aggregate T per model if multiple pairwise comparisons exist.
    # Mean is transparent and conservative for multiple direct evidence streams.
    if not df_model_support.empty:
        df_t = (
            df_model_support.groupby(["dataset", "model"], as_index=False)
            .agg(
                T_statistical_support_formal=("pairwise_T", "mean"),
                direct_statistical_evidence_count=("pairwise_T", "count"),
                evidence_labels=("evidence_label", lambda x: " | ".join(sorted(set(map(str, x))))),
            )
        )
    else:
        df_t = pd.DataFrame(
            columns=[
                "dataset",
                "model",
                "T_statistical_support_formal",
                "direct_statistical_evidence_count",
                "evidence_labels",
            ]
        )

    # Recompute Stage-2 refined scores with formal T
    df_stage2["dataset"] = df_stage2["dataset"].map(normalize_dataset_name)
    df_stage2["model"] = df_stage2["model"].map(normalize_model_name)

    df_stage2_v2 = df_stage2.merge(df_t, on=["dataset", "model"], how="left")

    df_stage2_v2["T_statistical_support_old"] = df_stage2_v2.get(
        "T_statistical_support", np.nan
    )

    # Models without direct statistical comparison receive neutral T=0.50.
    df_stage2_v2["T_statistical_support_formal"] = df_stage2_v2[
        "T_statistical_support_formal"
    ].fillna(0.50)

    df_stage2_v2["direct_statistical_evidence_count"] = df_stage2_v2[
        "direct_statistical_evidence_count"
    ].fillna(0).astype(int)

    df_stage2_v2["evidence_labels"] = df_stage2_v2["evidence_labels"].fillna(
        "no_direct_pairwise_statistical_evidence_neutral_T"
    )

    new_scores = []
    for _, row in df_stage2_v2.iterrows():
        vals = {
            "core": row["raise_ids_core_score"],
            "E": row["E_seed_stability"],
            "T": row["T_statistical_support_formal"],
        }
        new_scores.append(weighted_geom(vals, STAGE2_WEIGHTS))

    df_stage2_v2["raise_ids_stage2_score_formal_T"] = new_scores
    df_stage2_v2["rank_stage2_formal_T"] = df_stage2_v2.groupby("dataset")[
        "raise_ids_stage2_score_formal_T"
    ].rank(ascending=False, method="dense")

    # Summary table
    summary_rows = []
    if not df_core_summary.empty:
        df_core_summary["dataset"] = df_core_summary["dataset"].map(normalize_dataset_name)

    for dataset, grp in df_stage2_v2.groupby("dataset"):
        grp_sorted = grp.sort_values(
            ["raise_ids_stage2_score_formal_T", "raise_ids_core_score"],
            ascending=[False, False],
        )
        top_formal = grp_sorted.iloc[0]["model"]

        old_top = np.nan
        top_val = np.nan
        top_test = np.nan
        top_core = np.nan

        if not df_core_summary.empty:
            row = df_core_summary.loc[df_core_summary["dataset"] == dataset]
            if not row.empty:
                row0 = row.iloc[0]
                top_val = row0.get("top_model_by_validation_f1", np.nan)
                top_test = row0.get("top_model_by_test_f1", np.nan)
                top_core = row0.get("top_model_by_raise_ids_core", np.nan)
                old_top = row0.get("top_model_by_raise_ids_stage2", np.nan)

        summary_rows.append(
            {
                "dataset": dataset,
                "top_model_by_validation_f1": top_val,
                "top_model_by_test_f1": top_test,
                "top_model_by_raise_ids_core": top_core,
                "old_stage2_winner": old_top,
                "formal_T_stage2_winner": top_formal,
                "old_vs_formal_T_changed": old_top != top_formal,
            }
        )

    df_summary_v2 = pd.DataFrame(summary_rows)

    # Save outputs
    out_pairs = metrics_dir / "raise_ids_formal_statistical_support_pairs.csv"
    out_t = metrics_dir / "raise_ids_formal_T_by_model.csv"
    out_stage2 = metrics_dir / "raise_ids_stage2_refined_scores_formal_T.csv"
    out_summary = metrics_dir / "raise_ids_stage2_top_model_summary_formal_T.csv"
    out_mapping = metrics_dir / "raise_ids_T_mapping_definition.csv"
    out_latex_mapping = tables_dir / "table_raise_ids_t_mapping.tex"

    mapping_df = pd.DataFrame(
        [
            {
                "condition": "Favored by both Holm-corrected Wilcoxon and McNemar tests with practical effect",
                "favored_model_T": 1.00,
                "comparator_T": 0.15,
            },
            {
                "condition": "Favored by one Holm-corrected test with practical effect",
                "favored_model_T": 0.75,
                "comparator_T": 0.30,
            },
            {
                "condition": "No corrected statistical support or statistically indistinguishable",
                "favored_model_T": 0.50,
                "comparator_T": 0.50,
            },
            {
                "condition": "Statistically detectable but practically negligible difference",
                "favored_model_T": 0.50,
                "comparator_T": 0.50,
            },
        ]
    )

    df_pairs.to_csv(out_pairs, index=False)
    df_t.to_csv(out_t, index=False)
    df_stage2_v2.sort_values(["dataset", "rank_stage2_formal_T"]).to_csv(out_stage2, index=False)
    df_summary_v2.to_csv(out_summary, index=False)
    mapping_df.to_csv(out_mapping, index=False)
    write_latex_t_mapping_table(out_latex_mapping)

    print("Saved:")
    print(out_pairs)
    print(out_t)
    print(out_stage2)
    print(out_summary)
    print(out_mapping)
    print(out_latex_mapping)

    print("\nFormal T pairwise evidence:")
    print(
        df_pairs[
            [
                "dataset",
                "model_a",
                "model_b",
                "delta_f1_a_minus_b",
                "wilcoxon_p_holm",
                "mcnemar_p_holm",
                "practical_effect",
                "evidence_label",
                "T_model_a",
                "T_model_b",
            ]
        ].to_string(index=False)
    )

    print("\nFormal T by model:")
    print(df_t.to_string(index=False))

    print("\nStage-2 ranking with formal T:")
    print(
        df_stage2_v2.sort_values(["dataset", "rank_stage2_formal_T"])[
            [
                "dataset",
                "model",
                "raise_ids_core_score",
                "E_seed_stability",
                "T_statistical_support_old",
                "T_statistical_support_formal",
                "direct_statistical_evidence_count",
                "raise_ids_stage2_score_formal_T",
                "rank_stage2_formal_T",
            ]
        ].to_string(index=False)
    )

    print("\nFormal T winner summary:")
    print(df_summary_v2.to_string(index=False))


if __name__ == "__main__":
    main()