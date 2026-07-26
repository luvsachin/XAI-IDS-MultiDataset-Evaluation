from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import math
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------
DEFAULT_SEED_COUNT_FOR_WILCOXON = 20
PRACTICAL_F1_DELTA = 0.0005


# ---------------------------------------------------------------------
# Helpers
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

    for c in candidates:
        tokens = c.split("_")
        for col in cols:
            if all(t in col for t in tokens):
                return col

    if required:
        raise KeyError(f"Could not find any of {candidates}. Available columns: {cols}")
    return None


def safe_float(x: object, default: float = np.nan) -> float:
    try:
        if pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default


def holm_adjust(p_values: List[float]) -> List[float]:
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
    monotone_adj = np.maximum.accumulate(raw_adj)
    monotone_adj = np.minimum(monotone_adj, 1.0)

    for idx, adj in zip(sorted_idx, monotone_adj):
        adjusted[idx] = adj

    return adjusted.tolist()


def approx_norm_isf_two_sided_p(p: float) -> float:
    """
    Approximate z for a two-sided p-value without requiring scipy.
    Uses Python statistics if available; falls back to a conservative approximation.
    """
    p = max(min(float(p), 1.0), 1e-300)
    q = 1.0 - p / 2.0

    try:
        from statistics import NormalDist
        return abs(NormalDist().inv_cdf(q))
    except Exception:
        # Approximation for very small p; enough for effect-size labeling.
        return math.sqrt(max(0.0, -2.0 * math.log(p / 2.0)))


def wilcoxon_r_from_p(p: float, n: int) -> float:
    if pd.isna(p) or n <= 0:
        return np.nan
    z = approx_norm_isf_two_sided_p(p)
    return float(z / math.sqrt(n))


def label_effect_size_r(r: float) -> str:
    if pd.isna(r):
        return "not_available"
    ar = abs(float(r))
    if ar < 0.10:
        return "negligible"
    if ar < 0.30:
        return "small"
    if ar < 0.50:
        return "medium"
    return "large"


def label_delta(delta: float) -> str:
    if pd.isna(delta):
        return "not_available"
    if abs(float(delta)) < PRACTICAL_F1_DELTA:
        return "practically_negligible"
    if abs(float(delta)) < 0.005:
        return "small"
    if abs(float(delta)) < 0.02:
        return "moderate"
    return "large"


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


def fmt_p(x: object) -> str:
    if pd.isna(x):
        return "--"
    x = float(x)
    if x == 0:
        return r"$<10^{-300}$"
    if x < 0.001:
        return f"{x:.1e}"
    return f"{x:.4f}"


# ---------------------------------------------------------------------
# Statistical evidence table
# ---------------------------------------------------------------------
def build_statistical_effect_table(metrics_dir: Path) -> pd.DataFrame:
    sig_path = metrics_dir / "raise_ids_formal_statistical_support_pairs.csv"
    if sig_path.exists():
        df = read_csv(sig_path)
    else:
        df = normalize_columns(read_csv(metrics_dir / "statistical_significance_summary.csv"))

    df = normalize_columns(df)

    dataset_col = find_col(df, ["dataset"])
    model_a_col = find_col(df, ["model_a", "modela", "model_1", "model1"])
    model_b_col = find_col(df, ["model_b", "modelb", "model_2", "model2"])

    delta_col = find_col(
        df,
        ["delta_f1_a_minus_b", "mean_f1_difference_a_minus_b", "mean_f1_diff_a_minus_b", "delta_f1", "diff_f1"],
        required=False,
    )
    a_f1_col = find_col(df, ["a_mean_f1", "model_a_mean_f1", "mean_f1_a"], required=False)
    b_f1_col = find_col(df, ["b_mean_f1", "model_b_mean_f1", "mean_f1_b"], required=False)

    wil_raw_col = find_col(df, ["wilcoxon_p_raw", "wilcoxon_p_value", "wilcoxon_p"], required=False)
    mcn_raw_col = find_col(df, ["mcnemar_p_raw", "mcnemar_p_value", "mcnemar_p"], required=False)
    wil_holm_col = find_col(df, ["wilcoxon_p_holm"], required=False)
    mcn_holm_col = find_col(df, ["mcnemar_p_holm"], required=False)

    # Optional discordant-count columns
    b_col = find_col(df, ["b", "discordant_b", "a_correct_b_wrong", "model_a_correct_model_b_wrong"], required=False)
    c_col = find_col(df, ["c", "discordant_c", "a_wrong_b_correct", "model_a_wrong_model_b_correct"], required=False)
    seed_count_col = find_col(df, ["seed_count", "n_seeds", "num_seeds"], required=False)

    out = pd.DataFrame()
    out["dataset"] = df[dataset_col].map(normalize_dataset_name)
    out["model_a"] = df[model_a_col].map(normalize_model_name)
    out["model_b"] = df[model_b_col].map(normalize_model_name)

    if a_f1_col:
        out["a_mean_f1"] = df[a_f1_col].map(safe_float)
    else:
        out["a_mean_f1"] = np.nan

    if b_f1_col:
        out["b_mean_f1"] = df[b_f1_col].map(safe_float)
    else:
        out["b_mean_f1"] = np.nan

    if delta_col:
        out["delta_f1_a_minus_b"] = df[delta_col].map(safe_float)
    else:
        out["delta_f1_a_minus_b"] = out["a_mean_f1"] - out["b_mean_f1"]

    if wil_raw_col:
        out["wilcoxon_p_raw"] = df[wil_raw_col].map(safe_float)
    else:
        out["wilcoxon_p_raw"] = np.nan

    if mcn_raw_col:
        out["mcnemar_p_raw"] = df[mcn_raw_col].map(safe_float)
    else:
        out["mcnemar_p_raw"] = np.nan

    if wil_holm_col:
        out["wilcoxon_p_holm"] = df[wil_holm_col].map(safe_float)
    else:
        out["wilcoxon_p_holm"] = holm_adjust(out["wilcoxon_p_raw"].tolist())

    if mcn_holm_col:
        out["mcnemar_p_holm"] = df[mcn_holm_col].map(safe_float)
    else:
        out["mcnemar_p_holm"] = holm_adjust(out["mcnemar_p_raw"].tolist())

    if seed_count_col:
        out["seed_count"] = df[seed_count_col].map(lambda x: int(safe_float(x, DEFAULT_SEED_COUNT_FOR_WILCOXON)))
    else:
        out["seed_count"] = DEFAULT_SEED_COUNT_FOR_WILCOXON

    out["wilcoxon_r_approx"] = [
        wilcoxon_r_from_p(p, n)
        for p, n in zip(out["wilcoxon_p_holm"], out["seed_count"])
    ]
    out["wilcoxon_effect_label"] = out["wilcoxon_r_approx"].map(label_effect_size_r)
    out["delta_f1_effect_label"] = out["delta_f1_a_minus_b"].map(label_delta)

    if b_col and c_col:
        out["mcnemar_b"] = df[b_col].map(safe_float)
        out["mcnemar_c"] = df[c_col].map(safe_float)
        out["mcnemar_discordant_total"] = out["mcnemar_b"] + out["mcnemar_c"]
        out["mcnemar_discordance_imbalance"] = (
            (out["mcnemar_b"] - out["mcnemar_c"]).abs()
            / out["mcnemar_discordant_total"].replace(0, np.nan)
        )
    else:
        out["mcnemar_b"] = np.nan
        out["mcnemar_c"] = np.nan
        out["mcnemar_discordant_total"] = np.nan
        out["mcnemar_discordance_imbalance"] = np.nan

    out["practical_effect"] = out["delta_f1_a_minus_b"].abs() >= PRACTICAL_F1_DELTA
    out["interpretation"] = np.where(
        out["practical_effect"],
        "statistically_interpretable_with_practical_delta",
        "statistically_detectable_but_practically_negligible",
    )

    return out


# ---------------------------------------------------------------------
# All-model transparency tables
# ---------------------------------------------------------------------
def build_all_model_test_and_gap_tables(metrics_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    val = read_csv(metrics_dir / "multidataset_validation_results_full.csv")
    test = read_csv(metrics_dir / "multidataset_test_results_full.csv")

    val["dataset"] = val["dataset"].map(normalize_dataset_name)
    test["dataset"] = test["dataset"].map(normalize_dataset_name)
    val["model"] = val["model"].map(normalize_model_name)
    test["model"] = test["model"].map(normalize_model_name)

    keep_val = ["dataset", "model", "validation_f1", "validation_roc_auc", "validation_pr_auc"]
    keep_test = [
        "dataset",
        "model",
        "test_accuracy",
        "test_precision",
        "test_recall",
        "test_f1",
        "test_roc_auc",
        "test_pr_auc",
        "test_fpr",
        "test_fnr",
    ]

    merged = val[keep_val].merge(test[keep_test], on=["dataset", "model"], how="inner")

    merged["f1_gap_val_minus_test"] = merged["validation_f1"] - merged["test_f1"]
    merged["absolute_f1_gap"] = merged["f1_gap_val_minus_test"].abs()
    merged["test_rank_f1"] = merged.groupby("dataset")["test_f1"].rank(ascending=False, method="dense")
    merged["validation_rank_f1"] = merged.groupby("dataset")["validation_f1"].rank(ascending=False, method="dense")
    merged["rank_shift_validation_to_test"] = merged["test_rank_f1"] - merged["validation_rank_f1"]

    # Compact all-model independent-test table
    test_table = merged[
        [
            "dataset",
            "model",
            "test_rank_f1",
            "test_f1",
            "test_pr_auc",
            "test_roc_auc",
            "test_fpr",
            "test_fnr",
            "test_accuracy",
        ]
    ].sort_values(["dataset", "test_rank_f1", "model"])

    gap_table = merged[
        [
            "dataset",
            "model",
            "validation_rank_f1",
            "test_rank_f1",
            "validation_f1",
            "test_f1",
            "f1_gap_val_minus_test",
            "absolute_f1_gap",
            "rank_shift_validation_to_test",
        ]
    ].sort_values(["dataset", "validation_rank_f1", "model"])

    return test_table, gap_table


# ---------------------------------------------------------------------
# LaTeX writers
# ---------------------------------------------------------------------
def write_latex_stat_table(df: pd.DataFrame, path: Path) -> None:
    lines = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Corrected statistical evidence and practical-effect interpretation for top model comparisons. Holm correction is applied separately to Wilcoxon and McNemar p-values. The Wilcoxon effect is reported as an approximate $r$ effect size derived from the corrected two-sided p-value and seed count.}")
    lines.append(r"\label{tab:raise_ids_corrected_statistical_evidence}")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{lllrrrrll}")
    lines.append(r"\hline")
    lines.append(r"\textbf{Dataset} & \textbf{Model A} & \textbf{Model B} & \textbf{$\Delta F1$} & \textbf{Wilc. $p_H$} & \textbf{McN. $p_H$} & \textbf{Wilc. $r$} & \textbf{Effect} & \textbf{Interpretation} \\")
    lines.append(r"\hline")

    for _, row in df.iterrows():
        lines.append(
            f"{latex_escape(row['dataset'])} & "
            f"{latex_escape(row['model_a'])} & "
            f"{latex_escape(row['model_b'])} & "
            f"{fmt(row['delta_f1_a_minus_b'])} & "
            f"{fmt_p(row['wilcoxon_p_holm'])} & "
            f"{fmt_p(row['mcnemar_p_holm'])} & "
            f"{fmt(row['wilcoxon_r_approx'])} & "
            f"{latex_escape(row['wilcoxon_effect_label'])} & "
            f"{latex_escape(row['delta_f1_effect_label'])} \\\\"
        )

    lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table*}")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def write_latex_all_model_test_table(df: pd.DataFrame, path: Path) -> None:
    lines = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Full all-model independent-test performance. This table complements the validation-selected summary by showing the complete independent-test ranking used for reliability-aware evidence synthesis.}")
    lines.append(r"\label{tab:raise_ids_full_all_model_test_performance}")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{llrrrrrr}")
    lines.append(r"\hline")
    lines.append(r"\textbf{Dataset} & \textbf{Model} & \textbf{Rank} & \textbf{F1} & \textbf{PR-AUC} & \textbf{ROC-AUC} & \textbf{FPR} & \textbf{FNR} \\")
    lines.append(r"\hline")

    for _, row in df.iterrows():
        lines.append(
            f"{latex_escape(row['dataset'])} & "
            f"{latex_escape(row['model'])} & "
            f"{int(row['test_rank_f1'])} & "
            f"{fmt(row['test_f1'])} & "
            f"{fmt(row['test_pr_auc'])} & "
            f"{fmt(row['test_roc_auc'])} & "
            f"{fmt(row['test_fpr'])} & "
            f"{fmt(row['test_fnr'])} \\\\"
        )

    lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table*}")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def write_latex_gap_table(df: pd.DataFrame, path: Path) -> None:
    lines = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Validation-to-test F1 gap for all evaluated models. Positive gap values indicate validation optimism. Rank shift is defined as independent-test F1 rank minus validation F1 rank.}")
    lines.append(r"\label{tab:raise_ids_all_model_validation_test_gap}")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{llrrrrr}")
    lines.append(r"\hline")
    lines.append(r"\textbf{Dataset} & \textbf{Model} & \textbf{Val. rank} & \textbf{Test rank} & \textbf{Val. F1} & \textbf{Test F1} & \textbf{Gap} \\")
    lines.append(r"\hline")

    for _, row in df.iterrows():
        lines.append(
            f"{latex_escape(row['dataset'])} & "
            f"{latex_escape(row['model'])} & "
            f"{int(row['validation_rank_f1'])} & "
            f"{int(row['test_rank_f1'])} & "
            f"{fmt(row['validation_f1'])} & "
            f"{fmt(row['test_f1'])} & "
            f"{fmt(row['f1_gap_val_minus_test'])} \\\\"
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
    root = project_root_from_script()
    metrics_dir = root / "04_Results" / "metrics"
    tables_dir = root / "06_LaTeX" / "tables"

    metrics_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    stat_table = build_statistical_effect_table(metrics_dir)
    all_test_table, gap_table = build_all_model_test_and_gap_tables(metrics_dir)

    out_stat = metrics_dir / "raise_ids_corrected_statistical_evidence_with_effects.csv"
    out_test = metrics_dir / "raise_ids_full_all_model_independent_test_table.csv"
    out_gap = metrics_dir / "raise_ids_all_model_validation_test_gap_table.csv"

    stat_table.to_csv(out_stat, index=False)
    all_test_table.to_csv(out_test, index=False)
    gap_table.to_csv(out_gap, index=False)

    latex_stat = tables_dir / "table_raise_ids_corrected_statistical_evidence.tex"
    latex_test = tables_dir / "table_raise_ids_full_all_model_test_performance.tex"
    latex_gap = tables_dir / "table_raise_ids_all_model_validation_test_gap.tex"

    write_latex_stat_table(stat_table, latex_stat)
    write_latex_all_model_test_table(all_test_table, latex_test)
    write_latex_gap_table(gap_table, latex_gap)

    print("Saved:")
    print(out_stat)
    print(out_test)
    print(out_gap)
    print(latex_stat)
    print(latex_test)
    print(latex_gap)

    print("\nCorrected statistical evidence with effect sizes:")
    print(
        stat_table[
            [
                "dataset",
                "model_a",
                "model_b",
                "delta_f1_a_minus_b",
                "wilcoxon_p_holm",
                "mcnemar_p_holm",
                "wilcoxon_r_approx",
                "wilcoxon_effect_label",
                "delta_f1_effect_label",
                "interpretation",
            ]
        ].to_string(index=False)
    )

    print("\nFull all-model independent-test table:")
    print(all_test_table.to_string(index=False))

    print("\nAll-model validation-to-test gap table:")
    print(gap_table.to_string(index=False))


if __name__ == "__main__":
    main()