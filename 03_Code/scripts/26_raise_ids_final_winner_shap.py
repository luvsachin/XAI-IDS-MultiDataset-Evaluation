from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

from lightgbm import LGBMClassifier
from xgboost import XGBClassifier


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------
DEFAULT_RANDOM_STATE = 42

FINAL_WINNER_ORDER = ["NSL-KDD", "UNSW-NB15", "CICIDS2017"]


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
    x_ok = all((dir_path / f).exists() for f in ["X_train_final.csv", "X_val_final.csv", "X_test_final.csv"])
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


def load_train_test(root: Path, dataset_name: str) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    proc_dir = find_processed_dataset_dir(root, dataset_name)
    print(f"Using processed directory for {dataset_name}: {proc_dir}")

    x_train = read_csv(proc_dir / "X_train_final.csv")
    x_test = read_csv(proc_dir / "X_test_final.csv")

    y_train_path = find_first_existing(proc_dir, ["y_train_binary.csv", "y_train.csv"])
    y_test_path = find_first_existing(proc_dir, ["y_test_binary.csv", "y_test.csv"])

    if y_train_path is None or y_test_path is None:
        raise FileNotFoundError(f"Could not find y_train/y_test files in {proc_dir}")

    y_train = read_csv(y_train_path).iloc[:, 0]
    y_test = read_csv(y_test_path).iloc[:, 0]

    return x_train, y_train, x_test, y_test


# ---------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------
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
    }

    return mapping.get(s, str(x).strip())


# ---------------------------------------------------------------------
# Model builders
# ---------------------------------------------------------------------
def build_model(model_name: str, random_state: int):
    if model_name == "LightGBM":
        return LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            random_state=random_state,
            n_jobs=-1,
        )

    if model_name == "XGBoost":
        return XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=random_state,
            n_jobs=-1,
        )

    raise ValueError(f"Unsupported final-winner model for this script: {model_name}")


# ---------------------------------------------------------------------
# SHAP helpers
# ---------------------------------------------------------------------
def extract_shap_array(shap_output) -> np.ndarray:
    if hasattr(shap_output, "values"):
        arr = shap_output.values
    else:
        arr = shap_output

    if isinstance(arr, list):
        if len(arr) == 2:
            arr = arr[1]
        else:
            arr = arr[0]

    arr = np.asarray(arr)

    if arr.ndim == 3:
        if arr.shape[-1] == 2:
            arr = arr[:, :, 1]
        elif arr.shape[0] == 2:
            arr = arr[1]
        else:
            raise ValueError(f"Unexpected 3D SHAP array shape: {arr.shape}")

    if arr.ndim != 2:
        raise ValueError(f"Unexpected SHAP array shape: {arr.shape}")

    return arr


def compute_shap_importance(model, x_shap: pd.DataFrame) -> pd.DataFrame:
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(x_shap)
    arr = extract_shap_array(shap_values)

    mean_abs = np.abs(arr).mean(axis=0)

    out = pd.DataFrame(
        {
            "feature": list(x_shap.columns),
            "mean_abs_shap": mean_abs,
        }
    ).sort_values("mean_abs_shap", ascending=False)

    out["rank"] = range(1, len(out) + 1)

    total = float(out["mean_abs_shap"].sum())
    if total > 0:
        out["relative_importance"] = out["mean_abs_shap"] / total
    else:
        out["relative_importance"] = np.nan

    return out


def save_bar_plot(df_top: pd.DataFrame, dataset: str, model_name: str, out_path: Path) -> None:
    plot_df = df_top.sort_values("mean_abs_shap", ascending=True)

    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    ax.barh(plot_df["feature"], plot_df["mean_abs_shap"])
    ax.set_xlabel("Mean |SHAP|")
    ax.set_title(f"{dataset}: top SHAP features for final RAISE-IDS winner ({model_name})")
    ax.grid(True, axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def save_combined_bar_plot(df_top_all: pd.DataFrame, out_path: Path) -> None:
    datasets = FINAL_WINNER_ORDER

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.0), sharex=False)

    for ax, dataset in zip(axes, datasets):
        grp = df_top_all[df_top_all["dataset"] == dataset].copy()
        if grp.empty:
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            ax.set_title(dataset)
            continue

        model_name = grp["model"].iloc[0]
        plot_df = grp.sort_values("mean_abs_shap", ascending=True)

        ax.barh(plot_df["feature"], plot_df["mean_abs_shap"])
        ax.set_title(f"{dataset}\n{model_name}")
        ax.set_xlabel("Mean |SHAP|")
        ax.grid(True, axis="x", alpha=0.25)

    fig.suptitle("Top SHAP features for final RAISE-IDS winners")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------
# LaTeX table
# ---------------------------------------------------------------------
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


def write_latex_top_features_table(df_top: pd.DataFrame, path: Path, top_n: int = 10) -> None:
    lines = []

    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Top SHAP-ranked features for the final RAISE-IDS winners. Unlike the earlier validation-reference SHAP table, this table explains the models selected after Stage-2 reliability refinement.}")
    lines.append(r"\label{tab:raise_ids_final_winner_shap_top_features}")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{llrll}")
    lines.append(r"\hline")
    lines.append(r"\textbf{Dataset} & \textbf{Final winner} & \textbf{Rank} & \textbf{Feature} & \textbf{Mean $|SHAP|$} \\")
    lines.append(r"\hline")

    for _, row in df_top.sort_values(["dataset_order", "rank"]).iterrows():
        lines.append(
            f"{latex_escape(row['dataset'])} & "
            f"{latex_escape(row['model'])} & "
            f"{int(row['rank'])} & "
            f"{latex_escape(row['feature'])} & "
            f"{float(row['mean_abs_shap']):.4f} \\\\"
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
    parser.add_argument("--max-train-rows", type=int, default=100000)
    parser.add_argument("--shap-rows", type=int, default=3000)
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--random-state", type=int, default=DEFAULT_RANDOM_STATE)
    args = parser.parse_args()

    root = project_root_from_script()
    metrics_dir = root / "04_Results" / "metrics"
    figures_dir = root / "05_Figures" / "final"
    tables_dir = root / "06_LaTeX" / "tables"

    metrics_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    summary_path = metrics_dir / "raise_ids_stage2_top_model_summary_formal_T.csv"
    summary = read_csv(summary_path)

    summary["dataset"] = summary["dataset"].map(normalize_dataset_name)
    summary["formal_T_stage2_winner"] = summary["formal_T_stage2_winner"].map(normalize_model_name)

    all_importance_rows = []
    top_rows = []

    for dataset in FINAL_WINNER_ORDER:
        row = summary[summary["dataset"] == dataset]
        if row.empty:
            raise ValueError(f"No final winner found for dataset {dataset}")

        model_name = row.iloc[0]["formal_T_stage2_winner"]

        print(f"\n=== {dataset}: final RAISE-IDS winner = {model_name} ===")

        x_train, y_train, x_test, y_test = load_train_test(root, dataset)

        if args.max_train_rows > 0 and len(x_train) > args.max_train_rows:
            x_train_use = x_train.sample(args.max_train_rows, random_state=args.random_state)
            y_train_use = y_train.loc[x_train_use.index]
            x_train_use = x_train_use.reset_index(drop=True)
            y_train_use = y_train_use.reset_index(drop=True)
        else:
            x_train_use = x_train.reset_index(drop=True)
            y_train_use = y_train.reset_index(drop=True)

        if args.shap_rows > 0 and len(x_test) > args.shap_rows:
            x_shap = x_test.sample(args.shap_rows, random_state=args.random_state).reset_index(drop=True)
        else:
            x_shap = x_test.reset_index(drop=True)

        print(f"Training rows used: {len(x_train_use)}, SHAP rows used: {len(x_shap)}")

        model = build_model(model_name, args.random_state)
        model.fit(x_train_use, y_train_use)

        importance = compute_shap_importance(model, x_shap)
        importance["dataset"] = dataset
        importance["model"] = model_name
        importance["shap_rows"] = len(x_shap)
        importance["training_rows_used"] = len(x_train_use)

        all_importance_rows.append(importance)

        top = importance.head(args.top_n).copy()
        top_rows.append(top)

        single_fig = figures_dir / f"fig_raise_ids_final_winner_shap_{dataset.lower().replace('-', '_')}.pdf"
        save_bar_plot(top, dataset, model_name, single_fig)

        print(f"Top {args.top_n} features:")
        print(top[["rank", "feature", "mean_abs_shap", "relative_importance"]].to_string(index=False))
        print(f"Saved figure: {single_fig}")

    df_importance = pd.concat(all_importance_rows, ignore_index=True)
    df_top = pd.concat(top_rows, ignore_index=True)

    dataset_order_map = {ds: i for i, ds in enumerate(FINAL_WINNER_ORDER)}
    df_importance["dataset_order"] = df_importance["dataset"].map(dataset_order_map)
    df_top["dataset_order"] = df_top["dataset"].map(dataset_order_map)

    out_all = metrics_dir / "raise_ids_final_winner_shap_importance_all_features.csv"
    out_top = metrics_dir / "raise_ids_final_winner_shap_top_features.csv"

    df_importance.sort_values(["dataset_order", "rank"]).to_csv(out_all, index=False)
    df_top.sort_values(["dataset_order", "rank"]).to_csv(out_top, index=False)

    combined_fig = figures_dir / "fig_raise_ids_final_winner_shap_top_features.pdf"
    save_combined_bar_plot(df_top, combined_fig)

    latex_table = tables_dir / "table_raise_ids_final_winner_shap_top_features.tex"
    write_latex_top_features_table(df_top, latex_table, top_n=args.top_n)

    print("\nSaved:")
    print(out_all)
    print(out_top)
    print(combined_fig)
    print(latex_table)

    print("\nFinal-winner SHAP top features:")
    print(
        df_top.sort_values(["dataset_order", "rank"])[
            ["dataset", "model", "rank", "feature", "mean_abs_shap", "relative_importance"]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()