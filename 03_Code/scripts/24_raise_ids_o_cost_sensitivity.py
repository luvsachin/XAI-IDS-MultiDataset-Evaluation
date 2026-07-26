from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------
LAMBDA_VALUES = [0.50, 0.60, 0.70, 0.80, 0.90]

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
    available = {k: v for k, v in values.items() if pd.notna(v)}
    if not available:
        return np.nan

    total_w = sum(weights[k] for k in available)
    if total_w <= 0:
        return np.nan

    prod = 1.0
    for k, v in available.items():
        w = weights[k] / total_w
        v = float(np.clip(v, 1e-6, 1.0))
        prod *= v ** w

    return float(prod)


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


def format_float(x: object, ndigits: int = 4) -> str:
    if pd.isna(x):
        return "--"
    return f"{float(x):.{ndigits}f}"


# ---------------------------------------------------------------------
# Main calculations
# ---------------------------------------------------------------------
def build_full_metric_frame(metrics_dir: Path) -> pd.DataFrame:
    val_path = metrics_dir / "multidataset_validation_results_full.csv"
    test_path = metrics_dir / "multidataset_test_results_full.csv"

    df_val = read_csv(val_path)
    df_test = read_csv(test_path)

    df_val["dataset"] = df_val["dataset"].map(normalize_dataset_name)
    df_test["dataset"] = df_test["dataset"].map(normalize_dataset_name)
    df_val["model"] = df_val["model"].map(normalize_model_name)
    df_test["model"] = df_test["model"].map(normalize_model_name)

    keep_val = ["dataset", "model", "validation_f1"]
    keep_test = [
        "dataset",
        "model",
        "test_f1",
        "test_pr_auc",
        "test_fpr",
        "test_fnr",
    ]

    df = df_val[keep_val].merge(df_test[keep_test], on=["dataset", "model"], how="inner")

    df["Q"] = np.sqrt(df["test_f1"] * df["test_pr_auc"])
    df["G"] = 1.0 - np.maximum(0.0, df["validation_f1"] - df["test_f1"])

    return df


def compute_core_sensitivity(df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, float]] = []

    for lam in LAMBDA_VALUES:
        tmp = df.copy()

        tmp["lambda_fnr"] = lam
        tmp["O_lambda"] = 1.0 - (lam * tmp["test_fnr"] + (1.0 - lam) * tmp["test_fpr"])

        core_scores = []
        for _, row in tmp.iterrows():
            vals = {
                "Q": row["Q"],
                "O": row["O_lambda"],
                "G": row["G"],
            }
            core_scores.append(weighted_geom(vals, CORE_WEIGHTS))

        tmp["raise_ids_core_score_lambda"] = core_scores
        tmp["rank_core_lambda"] = tmp.groupby("dataset")["raise_ids_core_score_lambda"].rank(
            ascending=False,
            method="dense",
        )

        rows.extend(tmp.to_dict("records"))

    return pd.DataFrame(rows)


def summarize_core_sensitivity(df_core_lambda: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for (dataset, lam), grp in df_core_lambda.groupby(["dataset", "lambda_fnr"]):
        grp_sorted = grp.sort_values(
            ["raise_ids_core_score_lambda", "test_f1"],
            ascending=[False, False],
        )
        top3 = grp_sorted.head(3)

        rows.append(
            {
                "dataset": dataset,
                "lambda_fnr": lam,
                "top_core_model": grp_sorted.iloc[0]["model"],
                "top_core_score": grp_sorted.iloc[0]["raise_ids_core_score_lambda"],
                "top3_core_models": " | ".join(top3["model"].tolist()),
                "top3_core_scores": " | ".join(
                    f"{x:.4f}" for x in top3["raise_ids_core_score_lambda"].tolist()
                ),
            }
        )

    return pd.DataFrame(rows)


def compute_stage2_fixed_shortlist_sensitivity(
    df_core_lambda: pd.DataFrame,
    df_stage2_formal: pd.DataFrame,
) -> pd.DataFrame:
    """
    Recomputes Stage-2 score under changing O_lambda, while holding the Stage-2
    shortlist evidence fixed to models for which E and formal T have already been
    computed.

    This answers the specific reviewer question:
    Does changing the FNR/FPR operational-cost tradeoff alter the final Stage-2
    decision among fully audited shortlisted models?
    """
    df_stage2 = df_stage2_formal.copy()
    df_stage2["dataset"] = df_stage2["dataset"].map(normalize_dataset_name)
    df_stage2["model"] = df_stage2["model"].map(normalize_model_name)

    if "T_statistical_support_formal" not in df_stage2.columns:
        raise KeyError(
            "Expected column T_statistical_support_formal in raise_ids_stage2_refined_scores_formal_T.csv"
        )

    rows = []

    for _, row in df_core_lambda.iterrows():
        dataset = row["dataset"]
        model = row["model"]
        lam = row["lambda_fnr"]

        evidence = df_stage2[
            (df_stage2["dataset"] == dataset) & (df_stage2["model"] == model)
        ]

        if evidence.empty:
            continue

        ev = evidence.iloc[0]

        vals = {
            "core": row["raise_ids_core_score_lambda"],
            "E": ev["E_seed_stability"],
            "T": ev["T_statistical_support_formal"],
        }

        stage2_score = weighted_geom(vals, STAGE2_WEIGHTS)

        out = {
            "dataset": dataset,
            "model": model,
            "lambda_fnr": lam,
            "Q": row["Q"],
            "O_lambda": row["O_lambda"],
            "G": row["G"],
            "raise_ids_core_score_lambda": row["raise_ids_core_score_lambda"],
            "E_seed_stability": ev["E_seed_stability"],
            "T_statistical_support_formal": ev["T_statistical_support_formal"],
            "raise_ids_stage2_score_lambda": stage2_score,
        }

        rows.append(out)

    df = pd.DataFrame(rows)

    if df.empty:
        return df

    df["rank_stage2_lambda"] = df.groupby(["dataset", "lambda_fnr"])[
        "raise_ids_stage2_score_lambda"
    ].rank(ascending=False, method="dense")

    return df


def summarize_stage2_sensitivity(df_stage2_lambda: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for (dataset, lam), grp in df_stage2_lambda.groupby(["dataset", "lambda_fnr"]):
        grp_sorted = grp.sort_values(
            ["raise_ids_stage2_score_lambda", "raise_ids_core_score_lambda"],
            ascending=[False, False],
        )
        rows.append(
            {
                "dataset": dataset,
                "lambda_fnr": lam,
                "top_stage2_model": grp_sorted.iloc[0]["model"],
                "top_stage2_score": grp_sorted.iloc[0]["raise_ids_stage2_score_lambda"],
                "second_stage2_model": grp_sorted.iloc[1]["model"] if len(grp_sorted) > 1 else np.nan,
                "second_stage2_score": grp_sorted.iloc[1]["raise_ids_stage2_score_lambda"] if len(grp_sorted) > 1 else np.nan,
                "margin_top_minus_second": (
                    grp_sorted.iloc[0]["raise_ids_stage2_score_lambda"]
                    - grp_sorted.iloc[1]["raise_ids_stage2_score_lambda"]
                    if len(grp_sorted) > 1
                    else np.nan
                ),
            }
        )

    return pd.DataFrame(rows)


def summarize_winner_stability(df_summary: pd.DataFrame, winner_col: str) -> pd.DataFrame:
    rows = []

    for dataset, grp in df_summary.groupby("dataset"):
        counts = grp[winner_col].value_counts()
        top_model = counts.index[0]
        top_count = int(counts.iloc[0])
        rows.append(
            {
                "dataset": dataset,
                "dominant_model": top_model,
                "dominant_model_frequency": top_count / len(grp),
                "dominant_model_count": top_count,
                "num_lambda_values": len(grp),
                "winner_sequence": " | ".join(
                    f"{row.lambda_fnr:.2f}:{getattr(row, winner_col)}"
                    for row in grp.sort_values("lambda_fnr").itertuples()
                ),
            }
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------
def write_latex_o_sensitivity_table(df_stage2_summary: pd.DataFrame, path: Path) -> None:
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Operational-cost sensitivity of RAISE-IDS Stage-2 winners under alternative false-negative weights $\lambda$ in $O_\lambda=1-(\lambda FNR+(1-\lambda)FPR)$.}")
    lines.append(r"\label{tab:raise_ids_o_cost_sensitivity}")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{lccr}")
    lines.append(r"\hline")
    lines.append(r"\textbf{Dataset} & \textbf{$\lambda$} & \textbf{Stage-2 winner} & \textbf{Margin} \\")
    lines.append(r"\hline")

    for _, row in df_stage2_summary.sort_values(["dataset", "lambda_fnr"]).iterrows():
        lines.append(
            f"{latex_escape(row['dataset'])} & "
            f"{format_float(row['lambda_fnr'], 2)} & "
            f"{latex_escape(row['top_stage2_model'])} & "
            f"{format_float(row['margin_top_minus_second'])} \\\\"
        )

    lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def make_o_sensitivity_figure(df_stage2_summary: pd.DataFrame, path: Path) -> None:
    datasets = ["NSL-KDD", "UNSW-NB15", "CICIDS2017"]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), sharey=False)

    for ax, dataset in zip(axes, datasets):
        grp = df_stage2_summary[df_stage2_summary["dataset"] == dataset].sort_values("lambda_fnr")

        if grp.empty:
            ax.set_title(dataset)
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            continue

        ax.plot(grp["lambda_fnr"], grp["top_stage2_score"], marker="o", label="Winner score")
        ax.plot(grp["lambda_fnr"], grp["second_stage2_score"], marker="s", label="Second score")
        ax.set_title(dataset)
        ax.set_xlabel(r"$\lambda$ in $O_\lambda$")
        ax.set_ylabel("Stage-2 score")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)

    fig.suptitle("RAISE-IDS Stage-2 sensitivity to false-negative operational-cost weight")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main() -> None:
    root = project_root_from_script()
    metrics_dir = root / "04_Results" / "metrics"
    tables_dir = root / "06_LaTeX" / "tables"
    figures_dir = root / "05_Figures" / "final"

    metrics_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    df_base = build_full_metric_frame(metrics_dir)
    df_core_lambda = compute_core_sensitivity(df_base)
    df_core_summary = summarize_core_sensitivity(df_core_lambda)

    stage2_formal_path = metrics_dir / "raise_ids_stage2_refined_scores_formal_T.csv"
    df_stage2_formal = read_csv(stage2_formal_path)

    df_stage2_lambda = compute_stage2_fixed_shortlist_sensitivity(
        df_core_lambda=df_core_lambda,
        df_stage2_formal=df_stage2_formal,
    )
    df_stage2_summary = summarize_stage2_sensitivity(df_stage2_lambda)

    df_core_stability = summarize_winner_stability(df_core_summary, "top_core_model")
    df_stage2_stability = summarize_winner_stability(df_stage2_summary, "top_stage2_model")

    # Check whether changing lambda changes the top-3 Stage-1 shortlist.
    baseline_top3 = (
        df_core_summary[df_core_summary["lambda_fnr"] == 0.70]
        [["dataset", "top3_core_models"]]
        .rename(columns={"top3_core_models": "baseline_lambda_0_70_top3"})
    )
    df_core_summary = df_core_summary.merge(baseline_top3, on="dataset", how="left")
    df_core_summary["top3_changed_vs_lambda_0_70"] = (
        df_core_summary["top3_core_models"] != df_core_summary["baseline_lambda_0_70_top3"]
    )

    # Save outputs
    out_core_all = metrics_dir / "raise_ids_o_cost_sensitivity_core_all_models.csv"
    out_core_summary = metrics_dir / "raise_ids_o_cost_sensitivity_core_summary.csv"
    out_stage2_all = metrics_dir / "raise_ids_o_cost_sensitivity_stage2_fixed_shortlist.csv"
    out_stage2_summary = metrics_dir / "raise_ids_o_cost_sensitivity_stage2_summary.csv"
    out_core_stability = metrics_dir / "raise_ids_o_cost_sensitivity_core_winner_stability.csv"
    out_stage2_stability = metrics_dir / "raise_ids_o_cost_sensitivity_stage2_winner_stability.csv"

    df_core_lambda.sort_values(["dataset", "lambda_fnr", "rank_core_lambda"]).to_csv(out_core_all, index=False)
    df_core_summary.sort_values(["dataset", "lambda_fnr"]).to_csv(out_core_summary, index=False)
    df_stage2_lambda.sort_values(["dataset", "lambda_fnr", "rank_stage2_lambda"]).to_csv(out_stage2_all, index=False)
    df_stage2_summary.sort_values(["dataset", "lambda_fnr"]).to_csv(out_stage2_summary, index=False)
    df_core_stability.to_csv(out_core_stability, index=False)
    df_stage2_stability.to_csv(out_stage2_stability, index=False)

    latex_table = tables_dir / "table_raise_ids_o_cost_sensitivity.tex"
    fig_path = figures_dir / "fig_raise_ids_o_cost_sensitivity.pdf"

    write_latex_o_sensitivity_table(df_stage2_summary, latex_table)
    make_o_sensitivity_figure(df_stage2_summary, fig_path)

    print("Saved:")
    print(out_core_all)
    print(out_core_summary)
    print(out_stage2_all)
    print(out_stage2_summary)
    print(out_core_stability)
    print(out_stage2_stability)
    print(latex_table)
    print(fig_path)

    print("\nStage-1 Core winner stability under O_lambda:")
    print(df_core_stability.to_string(index=False))

    print("\nStage-2 winner stability under O_lambda:")
    print(df_stage2_stability.to_string(index=False))

    print("\nStage-1 Core summary:")
    print(
        df_core_summary[
            [
                "dataset",
                "lambda_fnr",
                "top_core_model",
                "top_core_score",
                "top3_core_models",
                "top3_changed_vs_lambda_0_70",
            ]
        ].sort_values(["dataset", "lambda_fnr"]).to_string(index=False)
    )

    print("\nStage-2 fixed-shortlist sensitivity summary:")
    print(
        df_stage2_summary[
            [
                "dataset",
                "lambda_fnr",
                "top_stage2_model",
                "top_stage2_score",
                "second_stage2_model",
                "second_stage2_score",
                "margin_top_minus_second",
            ]
        ].sort_values(["dataset", "lambda_fnr"]).to_string(index=False)
    )


if __name__ == "__main__":
    main()