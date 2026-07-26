from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -----------------------------------
# Helpers
# -----------------------------------
def project_root_from_script() -> Path:
    return Path(__file__).resolve().parents[2]


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    return pd.read_csv(path)


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
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    for old, new in replacements.items():
        s = s.replace(old, new)
    return s


def format_float(x: object, ndigits: int = 4) -> str:
    if pd.isna(x):
        return "--"
    return f"{float(x):.{ndigits}f}"


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


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


# -----------------------------------
# LaTeX table writers
# -----------------------------------
def make_component_definition_table() -> str:
    rows = [
        ("$Q$", "Predictive quality", r"$\sqrt{F1^{test}\cdot PR\text{-}AUC^{test}}$"),
        ("$O$", "Operational safety", r"$1-(0.7\cdot FNR + 0.3\cdot FPR)$"),
        ("$G$", "Generalization consistency", r"$1-\max(0, F1^{val}-F1^{test})$"),
        ("$E$", "Explanation seed stability", r"Mean pairwise top-$k$ SHAP Jaccard overlap"),
        ("$T$", "Statistical support", r"Ordinal support coefficient from Wilcoxon + McNemar evidence"),
        ("Core", "Stage-1 score", r"Weighted geometric aggregation of $Q$, $O$, and $G$"),
        ("Stage 2", "Refined score", r"Weighted geometric aggregation of Core, $E$, and $T$"),
    ]

    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{RAISE-IDS component definitions used in the reliability-aware model-selection framework.}")
    lines.append(r"\label{tab:raise_ids_components}")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{p{1.4cm}p{4.1cm}p{7.0cm}}")
    lines.append(r"\hline")
    lines.append(r"\textbf{Symbol} & \textbf{Meaning} & \textbf{Definition} \\")
    lines.append(r"\hline")
    for a, b, c in rows:
        lines.append(f"{a} & {latex_escape(b)} & {c} \\\\")
    lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def make_core_summary_table(df: pd.DataFrame) -> str:
    cols = [
        "dataset",
        "top_model_by_validation_f1",
        "top_model_by_test_f1",
        "top_model_by_raise_ids_core",
        "top_model_by_raise_ids_stage2",
    ]
    lines = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Comparison of preferred models under validation-only selection, independent-test ranking, Stage-1 RAISE-IDS Core screening, and Stage-2 refinement.}")
    lines.append(r"\label{tab:raise_ids_selection_summary}")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{lcccc}")
    lines.append(r"\hline")
    lines.append(r"\textbf{Dataset} & \textbf{Validation F1 winner} & \textbf{Test F1 winner} & \textbf{RAISE-IDS Core winner} & \textbf{RAISE-IDS Stage 2 winner} \\")
    lines.append(r"\hline")
    for _, row in df[cols].iterrows():
        lines.append(
            f"{latex_escape(row['dataset'])} & "
            f"{latex_escape(row['top_model_by_validation_f1'])} & "
            f"{latex_escape(row['top_model_by_test_f1'])} & "
            f"{latex_escape(row['top_model_by_raise_ids_core'])} & "
            f"{latex_escape(row['top_model_by_raise_ids_stage2'])} \\\\"
        )
    lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table*}")
    return "\n".join(lines)


def make_stage2_score_table(df: pd.DataFrame) -> str:
    lines = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Shortlisted Stage-2 RAISE-IDS refined scores. Core is the Stage-1 score based on predictive quality, operational safety, and generalization consistency; $E$ is seed-wise SHAP stability; $T$ is statistical-support coefficient.}")
    lines.append(r"\label{tab:raise_ids_stage2_scores}")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{llccccc}")
    lines.append(r"\hline")
    lines.append(r"\textbf{Dataset} & \textbf{Model} & \textbf{Core} & \textbf{$E$} & \textbf{$T$} & \textbf{Stage 2} & \textbf{Rank} \\")
    lines.append(r"\hline")
    for _, row in df.sort_values(["dataset", "rank_stage2"]).iterrows():
        lines.append(
            f"{latex_escape(row['dataset'])} & "
            f"{latex_escape(row['model'])} & "
            f"{format_float(row['raise_ids_core_score'])} & "
            f"{format_float(row['E_seed_stability'])} & "
            f"{format_float(row['T_statistical_support'], 2)} & "
            f"{format_float(row['raise_ids_stage2_score'])} & "
            f"{int(row['rank_stage2'])} \\\\"
        )
    lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table*}")
    return "\n".join(lines)


def make_ablation_table(df: pd.DataFrame) -> str:
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Ablation-based preferred models under the RAISE-IDS shortlist refinement stage.}")
    lines.append(r"\label{tab:raise_ids_ablation}")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{lcccc}")
    lines.append(r"\hline")
    lines.append(r"\textbf{Dataset} & \textbf{Core only} & \textbf{Core+$E$} & \textbf{Core+$T$} & \textbf{Stage 2 full} \\")
    lines.append(r"\hline")
    for _, row in df.iterrows():
        lines.append(
            f"{latex_escape(row['dataset'])} & "
            f"{latex_escape(row['top_model_core_only'])} & "
            f"{latex_escape(row['top_model_core_plus_E'])} & "
            f"{latex_escape(row['top_model_core_plus_T'])} & "
            f"{latex_escape(row['top_model_stage2_full'])} \\\\"
        )
    lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def make_weight_sensitivity_table(df: pd.DataFrame) -> str:
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Weight-sensitivity analysis of RAISE-IDS Stage-2 selection under 5,000 sampled weight configurations per dataset.}")
    lines.append(r"\label{tab:raise_ids_weight_sensitivity}")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{lccc}")
    lines.append(r"\hline")
    lines.append(r"\textbf{Dataset} & \textbf{Dominant model} & \textbf{Top-rank frequency} & \textbf{Mean margin} \\")
    lines.append(r"\hline")
    for _, row in df.iterrows():
        lines.append(
            f"{latex_escape(row['dataset'])} & "
            f"{latex_escape(row['dominant_model'])} & "
            f"{format_float(row['top_rank_frequency'])} & "
            f"{format_float(row['mean_margin_top_minus_second'])} \\\\"
        )
    lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


# -----------------------------------
# Figures
# -----------------------------------
def make_rank_shift_figure(df_scores: pd.DataFrame, fig_path: Path) -> None:
    datasets = ["NSL-KDD", "UNSW-NB15", "CICIDS2017"]
    stages = ["Validation", "Test", "Core", "Stage2"]
    x = np.arange(len(stages))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)

    for ax, dataset in zip(axes, datasets):
        grp = df_scores[df_scores["dataset"] == dataset].sort_values("rank_stage2")
        for _, row in grp.iterrows():
            y = [
                row["rank_validation_f1"],
                row["rank_test_f1"],
                row["rank_raise_ids_core"],
                row["rank_stage2"],
            ]
            ax.plot(x, y, marker="o", linewidth=1.8, label=row["model"])
        ax.set_title(dataset)
        ax.set_xticks(x)
        ax.set_xticklabels(stages, rotation=20)
        ax.set_ylim(max(df_scores[["rank_validation_f1", "rank_test_f1", "rank_raise_ids_core", "rank_stage2"]].max()) + 0.5, 0.5)
        ax.set_ylabel("Rank (1 = best)")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)

    fig.suptitle("Model rank changes from validation selection to RAISE-IDS Stage-2 refinement")
    fig.tight_layout()
    fig.savefig(fig_path, bbox_inches="tight")
    plt.close(fig)


def make_stage2_score_figure(df_scores: pd.DataFrame, fig_path: Path) -> None:
    datasets = ["NSL-KDD", "UNSW-NB15", "CICIDS2017"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=False)

    for ax, dataset in zip(axes, datasets):
        grp = df_scores[df_scores["dataset"] == dataset].sort_values("rank_stage2")
        ax.bar(grp["model"], grp["raise_ids_stage2_score"])
        ax.set_title(dataset)
        ax.set_ylabel("RAISE-IDS Stage-2 score")
        ax.tick_params(axis="x", rotation=30)
        ax.grid(True, axis="y", alpha=0.25)

    fig.suptitle("RAISE-IDS Stage-2 refined scores for shortlisted models")
    fig.tight_layout()
    fig.savefig(fig_path, bbox_inches="tight")
    plt.close(fig)


def make_weight_sensitivity_figure(df_ws: pd.DataFrame, fig_path: Path) -> None:
    df_plot = df_ws[df_ws["model"] != "__DATASET_SUMMARY__"].copy()
    datasets = ["NSL-KDD", "UNSW-NB15", "CICIDS2017"]
    models = sorted(df_plot["model"].unique().tolist())

    width = 0.22
    x = np.arange(len(datasets))

    fig, ax = plt.subplots(figsize=(10, 5))

    for i, model in enumerate(models):
        vals = []
        for ds in datasets:
            row = df_plot[(df_plot["dataset"] == ds) & (df_plot["model"] == model)]
            vals.append(float(row["top_rank_frequency"].iloc[0]) if not row.empty else 0.0)
        ax.bar(x + i * width, vals, width=width, label=model)

    ax.set_xticks(x + width * (len(models) - 1) / 2)
    ax.set_xticklabels(datasets)
    ax.set_ylabel("Top-rank frequency")
    ax.set_title("Weight-sensitivity of RAISE-IDS Stage-2 top-model selection")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_path, bbox_inches="tight")
    plt.close(fig)


# -----------------------------------
# Main
# -----------------------------------
def main() -> None:
    root = project_root_from_script()
    metrics_dir = root / "04_Results" / "metrics"
    figures_dir = root / "05_Figures" / "final"
    tables_dir = root / "06_LaTeX" / "tables"

    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    core_summary = read_csv(metrics_dir / "raise_ids_stage2_top_model_summary.csv")
    stage2_scores = read_csv(metrics_dir / "raise_ids_stage2_refined_scores.csv")
    ablation = read_csv(metrics_dir / "raise_ids_stage2_ablation_summary.csv")
    weight_summary = read_csv(metrics_dir / "raise_ids_weight_sensitivity_summary.csv")

    core_summary["dataset"] = core_summary["dataset"].map(normalize_dataset_name)
    stage2_scores["dataset"] = stage2_scores["dataset"].map(normalize_dataset_name)
    ablation["dataset"] = ablation["dataset"].map(normalize_dataset_name)
    weight_summary["dataset"] = weight_summary["dataset"].map(normalize_dataset_name)

    stage2_scores["model"] = stage2_scores["model"].map(normalize_model_name)
    ablation["top_model_core_only"] = ablation["top_model_core_only"].map(normalize_model_name)
    ablation["top_model_core_plus_E"] = ablation["top_model_core_plus_E"].map(normalize_model_name)
    ablation["top_model_core_plus_T"] = ablation["top_model_core_plus_T"].map(normalize_model_name)
    ablation["top_model_stage2_full"] = ablation["top_model_stage2_full"].map(normalize_model_name)
    weight_summary["model"] = weight_summary["model"].astype(str).map(lambda x: x if x == "__DATASET_SUMMARY__" else normalize_model_name(x))

    # manuscript summary csv
    ws_dataset_summary = weight_summary[weight_summary["model"] == "__DATASET_SUMMARY__"].copy()
    manuscript_rows = []
    for _, row in core_summary.iterrows():
        ds = row["dataset"]
        ws_row = ws_dataset_summary[ws_dataset_summary["dataset"] == ds].iloc[0]
        manuscript_rows.append(
            {
                "dataset": ds,
                "validation_winner": row["top_model_by_validation_f1"],
                "test_winner": row["top_model_by_test_f1"],
                "raise_ids_core_winner": row["top_model_by_raise_ids_core"],
                "raise_ids_stage2_winner": row["top_model_by_raise_ids_stage2"],
                "weight_sensitivity_dominant_model": ws_row["dominant_model"],
                "weight_sensitivity_top_frequency": ws_row["top_rank_frequency"],
                "weight_sensitivity_mean_margin": ws_row["mean_margin_top_minus_second"],
            }
        )
    manuscript_summary = pd.DataFrame(manuscript_rows)
    manuscript_summary_out = metrics_dir / "raise_ids_manuscript_summary.csv"
    manuscript_summary.to_csv(manuscript_summary_out, index=False)

    # save tables
    write_text(tables_dir / "table_raise_ids_component_definitions.tex", make_component_definition_table())
    write_text(tables_dir / "table_raise_ids_selection_summary.tex", make_core_summary_table(core_summary))
    write_text(tables_dir / "table_raise_ids_stage2_scores.tex", make_stage2_score_table(stage2_scores))
    write_text(tables_dir / "table_raise_ids_ablation_summary.tex", make_ablation_table(ablation))

    ws_display = ws_dataset_summary[[
        "dataset",
        "dominant_model",
        "top_rank_frequency",
        "mean_margin_top_minus_second",
    ]].copy()
    write_text(tables_dir / "table_raise_ids_weight_sensitivity.tex", make_weight_sensitivity_table(ws_display))

    # figures
    make_rank_shift_figure(stage2_scores, figures_dir / "fig_raise_ids_rank_shift.pdf")
    make_stage2_score_figure(stage2_scores, figures_dir / "fig_raise_ids_stage2_scores.pdf")
    make_weight_sensitivity_figure(weight_summary, figures_dir / "fig_raise_ids_weight_sensitivity.pdf")

    print("Saved manuscript summary:")
    print(manuscript_summary_out)

    print("\nSaved LaTeX tables:")
    for name in [
        "table_raise_ids_component_definitions.tex",
        "table_raise_ids_selection_summary.tex",
        "table_raise_ids_stage2_scores.tex",
        "table_raise_ids_ablation_summary.tex",
        "table_raise_ids_weight_sensitivity.tex",
    ]:
        print(tables_dir / name)

    print("\nSaved figures:")
    for name in [
        "fig_raise_ids_rank_shift.pdf",
        "fig_raise_ids_stage2_scores.pdf",
        "fig_raise_ids_weight_sensitivity.pdf",
    ]:
        print(figures_dir / name)

    print("\nManuscript summary preview:")
    print(manuscript_summary.to_string(index=False))


if __name__ == "__main__":
    main()