from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------
LAMBDA_FNR = 0.70

CORE_WEIGHTS = {
    "Q": 0.45,
    "O": 0.35,
    "G": 0.20,
}

STAGE2_WEIGHTS = {
    "core": 0.60,
    "E": 0.25,
    "T": 0.15,
}


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
    prod = 1.0
    total_w = sum(weights[k] for k in values)

    for k, v in values.items():
        w = weights[k] / total_w
        v = float(np.clip(v, 1e-6, 1.0))
        prod *= v ** w

    return float(prod)


def harmonic_mean(values: List[float]) -> float:
    vals = np.asarray(values, dtype=float)
    vals = np.clip(vals, 1e-6, 1.0)
    return float(len(vals) / np.sum(1.0 / vals))


def topsis_score(df: pd.DataFrame, component_cols: List[str], weights: Dict[str, float] | None = None) -> pd.Series:
    """
    TOPSIS score for benefit criteria.
    All components are assumed beneficial and in [0,1].
    """
    X = df[component_cols].astype(float).to_numpy()

    # Vector normalization
    denom = np.sqrt((X ** 2).sum(axis=0))
    denom[denom == 0] = 1.0
    R = X / denom

    if weights is None:
        w = np.ones(len(component_cols), dtype=float) / len(component_cols)
    else:
        w = np.array([weights[c] for c in component_cols], dtype=float)
        w = w / w.sum()

    V = R * w

    ideal_pos = V.max(axis=0)
    ideal_neg = V.min(axis=0)

    d_pos = np.sqrt(((V - ideal_pos) ** 2).sum(axis=1))
    d_neg = np.sqrt(((V - ideal_neg) ** 2).sum(axis=1))

    closeness = d_neg / (d_pos + d_neg + 1e-12)
    return pd.Series(closeness, index=df.index)


def borda_score(df: pd.DataFrame, component_cols: List[str]) -> pd.Series:
    """
    Borda aggregation over descending ranks.
    Higher component value gets higher Borda score.
    """
    n = len(df)
    total = pd.Series(0.0, index=df.index)

    for col in component_cols:
        ranks = df[col].rank(ascending=False, method="average")
        total += n - ranks

    return total


def pareto_front(df: pd.DataFrame, component_cols: List[str]) -> List[str]:
    """
    Returns non-dominated models for benefit criteria.
    A model is dominated if another model is >= on all components and > on at least one.
    """
    models = df["model"].tolist()
    X = df[component_cols].astype(float).to_numpy()

    non_dominated = []

    for i, model in enumerate(models):
        xi = X[i]
        dominated = False

        for j in range(len(models)):
            if i == j:
                continue
            xj = X[j]
            if np.all(xj >= xi) and np.any(xj > xi):
                dominated = True
                break

        if not dominated:
            non_dominated.append(model)

    return non_dominated


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


def write_latex_selection_table(df_summary: pd.DataFrame, path: Path) -> None:
    """
    Compact manuscript-ready table.
    """
    preferred_rules = [
        "Validation F1",
        "Test F1",
        "Test PR-AUC",
        "Arithmetic mean",
        "Borda",
        "TOPSIS",
        "RAISE-IDS Core",
        "RAISE-IDS Stage 2",
        "Pareto front",
    ]

    df = df_summary[df_summary["selection_rule"].isin(preferred_rules)].copy()

    lines = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Comparison of RAISE-IDS with simple and MCDA-style model-selection baselines. Pareto front entries list all non-dominated candidates under the corresponding evidence space.}")
    lines.append(r"\label{tab:raise_ids_mcda_baseline_comparison}")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{llccc}")
    lines.append(r"\hline")
    lines.append(r"\textbf{Selection rule} & \textbf{Evidence space} & \textbf{CICIDS2017} & \textbf{NSL-KDD} & \textbf{UNSW-NB15} \\")
    lines.append(r"\hline")

    for rule in preferred_rules:
        sub = df[df["selection_rule"] == rule]
        if sub.empty:
            continue

        evidence_space = sub["evidence_space"].iloc[0]

        values = {}
        for ds in ["CICIDS2017", "NSL-KDD", "UNSW-NB15"]:
            row = sub[sub["dataset"] == ds]
            if row.empty:
                values[ds] = "--"
            else:
                values[ds] = row.iloc[0]["selected_model_or_front"]

        lines.append(
            f"{latex_escape(rule)} & "
            f"{latex_escape(evidence_space)} & "
            f"{latex_escape(values['CICIDS2017'])} & "
            f"{latex_escape(values['NSL-KDD'])} & "
            f"{latex_escape(values['UNSW-NB15'])} \\\\"
        )

    lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table*}")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------
# Data construction
# ---------------------------------------------------------------------
def build_core_frame(metrics_dir: Path) -> pd.DataFrame:
    val_path = metrics_dir / "multidataset_validation_results_full.csv"
    test_path = metrics_dir / "multidataset_test_results_full.csv"

    df_val = read_csv(val_path)
    df_test = read_csv(test_path)

    df_val["dataset"] = df_val["dataset"].map(normalize_dataset_name)
    df_test["dataset"] = df_test["dataset"].map(normalize_dataset_name)
    df_val["model"] = df_val["model"].map(normalize_model_name)
    df_test["model"] = df_test["model"].map(normalize_model_name)

    df = df_val[
        [
            "dataset",
            "model",
            "validation_f1",
            "validation_roc_auc",
            "validation_pr_auc",
        ]
    ].merge(
        df_test[
            [
                "dataset",
                "model",
                "test_f1",
                "test_roc_auc",
                "test_pr_auc",
                "test_fpr",
                "test_fnr",
            ]
        ],
        on=["dataset", "model"],
        how="inner",
    )

    df["Q"] = np.sqrt(df["test_f1"] * df["test_pr_auc"])
    df["O"] = 1.0 - (LAMBDA_FNR * df["test_fnr"] + (1.0 - LAMBDA_FNR) * df["test_fpr"])
    df["G"] = 1.0 - np.maximum(0.0, df["validation_f1"] - df["test_f1"])

    df["Arithmetic_QOG"] = df[["Q", "O", "G"]].mean(axis=1)
    df["Harmonic_QOG"] = df[["Q", "O", "G"]].apply(lambda r: harmonic_mean(r.tolist()), axis=1)
    df["Minimum_QOG"] = df[["Q", "O", "G"]].min(axis=1)

    df["RAISE_IDS_Core"] = df.apply(
        lambda r: weighted_geom({"Q": r["Q"], "O": r["O"], "G": r["G"]}, CORE_WEIGHTS),
        axis=1,
    )

    # MCDA scores per dataset
    tmp_rows = []
    for dataset, grp in df.groupby("dataset"):
        grp = grp.copy()
        grp["Borda_QOG"] = borda_score(grp, ["Q", "O", "G"])
        grp["TOPSIS_QOG"] = topsis_score(grp, ["Q", "O", "G"])
        tmp_rows.append(grp)

    df = pd.concat(tmp_rows, ignore_index=True)

    return df


def build_stage2_frame(metrics_dir: Path) -> pd.DataFrame:
    stage2_path = metrics_dir / "raise_ids_stage2_refined_scores_formal_T.csv"
    df = read_csv(stage2_path)

    df["dataset"] = df["dataset"].map(normalize_dataset_name)
    df["model"] = df["model"].map(normalize_model_name)

    # Make a consistent Stage-2 formal score column.
    if "raise_ids_stage2_score_formal_T" not in df.columns:
        raise KeyError("Missing raise_ids_stage2_score_formal_T. Run Script 23 first.")

    df["Stage2_Arithmetic"] = df[
        [
            "raise_ids_core_score",
            "E_seed_stability",
            "T_statistical_support_formal",
        ]
    ].mean(axis=1)

    df["Stage2_Harmonic"] = df[
        [
            "raise_ids_core_score",
            "E_seed_stability",
            "T_statistical_support_formal",
        ]
    ].apply(lambda r: harmonic_mean(r.tolist()), axis=1)

    df["Stage2_Minimum"] = df[
        [
            "raise_ids_core_score",
            "E_seed_stability",
            "T_statistical_support_formal",
        ]
    ].min(axis=1)

    tmp_rows = []
    for dataset, grp in df.groupby("dataset"):
        grp = grp.copy()
        grp["Stage2_Borda"] = borda_score(
            grp,
            [
                "raise_ids_core_score",
                "E_seed_stability",
                "T_statistical_support_formal",
            ],
        )
        grp["Stage2_TOPSIS"] = topsis_score(
            grp,
            [
                "raise_ids_core_score",
                "E_seed_stability",
                "T_statistical_support_formal",
            ],
        )
        tmp_rows.append(grp)

    df = pd.concat(tmp_rows, ignore_index=True)

    return df


# ---------------------------------------------------------------------
# Selection summaries
# ---------------------------------------------------------------------
def top_model(grp: pd.DataFrame, score_col: str) -> Tuple[str, float]:
    sorted_grp = grp.sort_values([score_col, "test_f1" if "test_f1" in grp.columns else score_col], ascending=[False, False])
    return sorted_grp.iloc[0]["model"], float(sorted_grp.iloc[0][score_col])


def build_core_selection_summary(df_core: pd.DataFrame) -> pd.DataFrame:
    rules = [
        ("Validation F1", "Single metric", "validation_f1"),
        ("Test F1", "Single metric", "test_f1"),
        ("Test PR-AUC", "Single metric", "test_pr_auc"),
        ("Operational safety", "Single metric", "O"),
        ("Arithmetic mean", "Q/O/G", "Arithmetic_QOG"),
        ("Harmonic mean", "Q/O/G", "Harmonic_QOG"),
        ("Minimum operator", "Q/O/G", "Minimum_QOG"),
        ("Borda", "Q/O/G", "Borda_QOG"),
        ("TOPSIS", "Q/O/G", "TOPSIS_QOG"),
        ("RAISE-IDS Core", "Q/O/G", "RAISE_IDS_Core"),
    ]

    rows = []

    for dataset, grp in df_core.groupby("dataset"):
        pareto = pareto_front(grp, ["Q", "O", "G"])

        for rule_name, evidence_space, score_col in rules:
            model, score = top_model(grp, score_col)
            rows.append(
                {
                    "stage": "core_all_models",
                    "dataset": dataset,
                    "selection_rule": rule_name,
                    "evidence_space": evidence_space,
                    "selected_model_or_front": model,
                    "score_or_note": score,
                }
            )

        rows.append(
            {
                "stage": "core_all_models",
                "dataset": dataset,
                "selection_rule": "Pareto front",
                "evidence_space": "Q/O/G",
                "selected_model_or_front": " | ".join(pareto),
                "score_or_note": len(pareto),
            }
        )

    return pd.DataFrame(rows)


def build_stage2_selection_summary(df_stage2: pd.DataFrame) -> pd.DataFrame:
    rules = [
        ("Stage2 arithmetic", "Core/E/T", "Stage2_Arithmetic"),
        ("Stage2 harmonic", "Core/E/T", "Stage2_Harmonic"),
        ("Stage2 minimum", "Core/E/T", "Stage2_Minimum"),
        ("Stage2 Borda", "Core/E/T", "Stage2_Borda"),
        ("Stage2 TOPSIS", "Core/E/T", "Stage2_TOPSIS"),
        ("RAISE-IDS Stage 2", "Core/E/T", "raise_ids_stage2_score_formal_T"),
    ]

    rows = []

    for dataset, grp in df_stage2.groupby("dataset"):
        pareto = pareto_front(
            grp,
            [
                "raise_ids_core_score",
                "E_seed_stability",
                "T_statistical_support_formal",
            ],
        )

        for rule_name, evidence_space, score_col in rules:
            model, score = top_model(grp, score_col)
            rows.append(
                {
                    "stage": "stage2_shortlist",
                    "dataset": dataset,
                    "selection_rule": rule_name,
                    "evidence_space": evidence_space,
                    "selected_model_or_front": model,
                    "score_or_note": score,
                }
            )

        rows.append(
            {
                "stage": "stage2_shortlist",
                "dataset": dataset,
                "selection_rule": "Stage2 Pareto front",
                "evidence_space": "Core/E/T",
                "selected_model_or_front": " | ".join(pareto),
                "score_or_note": len(pareto),
            }
        )

    return pd.DataFrame(rows)


def build_decision_alignment_summary(df_combined: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for dataset, grp in df_combined.groupby("dataset"):
        core_winner = grp[
            (grp["stage"] == "core_all_models") &
            (grp["selection_rule"] == "RAISE-IDS Core")
        ]["selected_model_or_front"].iloc[0]

        stage2_winner = grp[
            (grp["stage"] == "stage2_shortlist") &
            (grp["selection_rule"] == "RAISE-IDS Stage 2")
        ]["selected_model_or_front"].iloc[0]

        for _, row in grp.iterrows():
            comparator = stage2_winner if row["stage"] == "stage2_shortlist" else core_winner

            rows.append(
                {
                    "dataset": dataset,
                    "stage": row["stage"],
                    "selection_rule": row["selection_rule"],
                    "selected_model_or_front": row["selected_model_or_front"],
                    "raise_ids_reference_winner": comparator,
                    "matches_raise_ids_reference": row["selected_model_or_front"] == comparator,
                }
            )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main() -> None:
    root = project_root_from_script()
    metrics_dir = root / "04_Results" / "metrics"
    tables_dir = root / "06_LaTeX" / "tables"

    metrics_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    df_core = build_core_frame(metrics_dir)
    df_stage2 = build_stage2_frame(metrics_dir)

    core_summary = build_core_selection_summary(df_core)
    stage2_summary = build_stage2_selection_summary(df_stage2)

    combined_summary = pd.concat([core_summary, stage2_summary], ignore_index=True)
    alignment_summary = build_decision_alignment_summary(combined_summary)

    # Save detailed score files
    out_core_scores = metrics_dir / "raise_ids_mcda_core_all_model_scores.csv"
    out_stage2_scores = metrics_dir / "raise_ids_mcda_stage2_shortlist_scores.csv"
    out_core_summary = metrics_dir / "raise_ids_mcda_core_selection_summary.csv"
    out_stage2_summary = metrics_dir / "raise_ids_mcda_stage2_selection_summary.csv"
    out_combined_summary = metrics_dir / "raise_ids_mcda_baseline_comparison_summary.csv"
    out_alignment = metrics_dir / "raise_ids_mcda_alignment_summary.csv"

    df_core.sort_values(["dataset", "RAISE_IDS_Core"], ascending=[True, False]).to_csv(out_core_scores, index=False)
    df_stage2.sort_values(["dataset", "raise_ids_stage2_score_formal_T"], ascending=[True, False]).to_csv(out_stage2_scores, index=False)
    core_summary.to_csv(out_core_summary, index=False)
    stage2_summary.to_csv(out_stage2_summary, index=False)
    combined_summary.to_csv(out_combined_summary, index=False)
    alignment_summary.to_csv(out_alignment, index=False)

    latex_table = tables_dir / "table_raise_ids_mcda_baseline_comparison.tex"
    write_latex_selection_table(combined_summary, latex_table)

    print("Saved:")
    print(out_core_scores)
    print(out_stage2_scores)
    print(out_core_summary)
    print(out_stage2_summary)
    print(out_combined_summary)
    print(out_alignment)
    print(latex_table)

    print("\nCore all-model selection summary:")
    print(
        core_summary.sort_values(["dataset", "selection_rule"])[
            [
                "dataset",
                "selection_rule",
                "evidence_space",
                "selected_model_or_front",
                "score_or_note",
            ]
        ].to_string(index=False)
    )

    print("\nStage-2 shortlist selection summary:")
    print(
        stage2_summary.sort_values(["dataset", "selection_rule"])[
            [
                "dataset",
                "selection_rule",
                "evidence_space",
                "selected_model_or_front",
                "score_or_note",
            ]
        ].to_string(index=False)
    )

    print("\nAlignment summary:")
    print(alignment_summary.to_string(index=False))


if __name__ == "__main__":
    main()