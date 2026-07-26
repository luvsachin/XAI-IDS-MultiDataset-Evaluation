from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import brier_score_loss, log_loss
from sklearn.calibration import calibration_curve

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


def load_train_test(root: Path, dataset_name: str) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
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

    raise ValueError(f"Unsupported final-winner model for calibration script: {model_name}")


def predict_positive_probability(model, x: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(x)
        if proba.ndim == 2 and proba.shape[1] >= 2:
            return proba[:, 1]
        return proba.ravel()

    if hasattr(model, "decision_function"):
        scores = model.decision_function(x)
        return 1.0 / (1.0 + np.exp(-scores))

    raise ValueError("Model does not support probability or decision scores.")


# ---------------------------------------------------------------------
# Calibration metrics
# ---------------------------------------------------------------------
def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)

    for i in range(n_bins):
        low = bins[i]
        high = bins[i + 1]

        if i == n_bins - 1:
            mask = (y_prob >= low) & (y_prob <= high)
        else:
            mask = (y_prob >= low) & (y_prob < high)

        if not np.any(mask):
            continue

        bin_conf = float(np.mean(y_prob[mask]))
        bin_acc = float(np.mean(y_true[mask]))
        bin_frac = float(np.mean(mask))
        ece += bin_frac * abs(bin_acc - bin_conf)

    return float(ece)


def maximum_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    mce = 0.0

    for i in range(n_bins):
        low = bins[i]
        high = bins[i + 1]

        if i == n_bins - 1:
            mask = (y_prob >= low) & (y_prob <= high)
        else:
            mask = (y_prob >= low) & (y_prob < high)

        if not np.any(mask):
            continue

        bin_conf = float(np.mean(y_prob[mask]))
        bin_acc = float(np.mean(y_true[mask]))
        mce = max(mce, abs(bin_acc - bin_conf))

    return float(mce)


def calibration_bins(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    rows = []

    for i in range(n_bins):
        low = bins[i]
        high = bins[i + 1]

        if i == n_bins - 1:
            mask = (y_prob >= low) & (y_prob <= high)
        else:
            mask = (y_prob >= low) & (y_prob < high)

        count = int(np.sum(mask))

        if count == 0:
            rows.append(
                {
                    "bin": i + 1,
                    "bin_low": low,
                    "bin_high": high,
                    "count": 0,
                    "mean_predicted_probability": np.nan,
                    "observed_positive_fraction": np.nan,
                    "absolute_gap": np.nan,
                }
            )
            continue

        mean_prob = float(np.mean(y_prob[mask]))
        obs_frac = float(np.mean(y_true[mask]))

        rows.append(
            {
                "bin": i + 1,
                "bin_low": low,
                "bin_high": high,
                "count": count,
                "mean_predicted_probability": mean_prob,
                "observed_positive_fraction": obs_frac,
                "absolute_gap": abs(obs_frac - mean_prob),
            }
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Figures and tables
# ---------------------------------------------------------------------
def save_reliability_diagram(df_bins: pd.DataFrame, dataset: str, model_name: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 5.0))

    valid = df_bins.dropna(subset=["mean_predicted_probability", "observed_positive_fraction"])

    ax.plot([0, 1], [0, 1], linestyle="--", label="Perfect calibration")
    ax.plot(
        valid["mean_predicted_probability"],
        valid["observed_positive_fraction"],
        marker="o",
        label=f"{dataset} {model_name}",
    )

    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed positive fraction")
    ax.set_title(f"Reliability diagram: {dataset} ({model_name})")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def save_combined_reliability_diagram(all_bins: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.8, 5.6))

    ax.plot([0, 1], [0, 1], linestyle="--", label="Perfect calibration")

    for dataset in FINAL_WINNER_ORDER:
        grp = all_bins[all_bins["dataset"] == dataset].dropna(
            subset=["mean_predicted_probability", "observed_positive_fraction"]
        )
        if grp.empty:
            continue

        model_name = grp["model"].iloc[0]
        ax.plot(
            grp["mean_predicted_probability"],
            grp["observed_positive_fraction"],
            marker="o",
            label=f"{dataset} ({model_name})",
        )

    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed positive fraction")
    ax.set_title("Reliability diagrams for final RAISE-IDS winners")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


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


def write_latex_calibration_table(summary: pd.DataFrame, path: Path) -> None:
    lines = []

    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Calibration audit for final RAISE-IDS winners. ECE and MCE are computed using 10 equal-width probability bins. Calibration is reported as auxiliary operational evidence and is not used to train or tune the models.}")
    lines.append(r"\label{tab:raise_ids_calibration_audit}")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{llrrrr}")
    lines.append(r"\hline")
    lines.append(r"\textbf{Dataset} & \textbf{Winner} & \textbf{Brier} & \textbf{Log-loss} & \textbf{ECE} & \textbf{MCE} \\")
    lines.append(r"\hline")

    for _, row in summary.sort_values("dataset_order").iterrows():
        lines.append(
            f"{latex_escape(row['dataset'])} & "
            f"{latex_escape(row['model'])} & "
            f"{fmt(row['brier_score'])} & "
            f"{fmt(row['log_loss'])} & "
            f"{fmt(row['ece_10'])} & "
            f"{fmt(row['mce_10'])} \\\\"
        )

    lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-train-rows", type=int, default=100000)
    parser.add_argument("--max-test-rows", type=int, default=0)
    parser.add_argument("--bins", type=int, default=10)
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

    metric_rows = []
    all_bin_rows = []

    dataset_order_map = {ds: i for i, ds in enumerate(FINAL_WINNER_ORDER)}

    for dataset in FINAL_WINNER_ORDER:
        row = summary[summary["dataset"] == dataset]
        if row.empty:
            raise ValueError(f"No final winner found for dataset {dataset}")

        model_name = row.iloc[0]["formal_T_stage2_winner"]

        print(f"\n=== Calibration audit: {dataset}, final winner = {model_name} ===")

        x_train, y_train, x_test, y_test = load_train_test(root, dataset)

        if args.max_train_rows > 0 and len(x_train) > args.max_train_rows:
            x_train_use = x_train.sample(args.max_train_rows, random_state=args.random_state)
            y_train_use = y_train.loc[x_train_use.index]
            x_train_use = x_train_use.reset_index(drop=True)
            y_train_use = y_train_use.reset_index(drop=True)
        else:
            x_train_use = x_train.reset_index(drop=True)
            y_train_use = y_train.reset_index(drop=True)

        if args.max_test_rows > 0 and len(x_test) > args.max_test_rows:
            x_test_use = x_test.sample(args.max_test_rows, random_state=args.random_state)
            y_test_use = y_test.loc[x_test_use.index]
            x_test_use = x_test_use.reset_index(drop=True)
            y_test_use = y_test_use.reset_index(drop=True)
        else:
            x_test_use = x_test.reset_index(drop=True)
            y_test_use = y_test.reset_index(drop=True)

        print(f"Training rows used: {len(x_train_use)}, test rows used: {len(x_test_use)}")

        model = build_model(model_name, args.random_state)
        model.fit(x_train_use, y_train_use)

        y_prob = predict_positive_probability(model, x_test_use)
        y_prob = np.clip(y_prob, 1e-12, 1 - 1e-12)

        brier = brier_score_loss(y_test_use, y_prob)
        ll = log_loss(y_test_use, y_prob, labels=[0, 1])
        ece = expected_calibration_error(y_test_use.to_numpy(), y_prob, n_bins=args.bins)
        mce = maximum_calibration_error(y_test_use.to_numpy(), y_prob, n_bins=args.bins)

        bins = calibration_bins(y_test_use.to_numpy(), y_prob, n_bins=args.bins)
        bins["dataset"] = dataset
        bins["model"] = model_name
        bins["dataset_order"] = dataset_order_map[dataset]

        metric_rows.append(
            {
                "dataset": dataset,
                "dataset_order": dataset_order_map[dataset],
                "model": model_name,
                "train_rows_used": len(x_train_use),
                "test_rows_used": len(x_test_use),
                "brier_score": brier,
                "log_loss": ll,
                f"ece_{args.bins}": ece,
                f"mce_{args.bins}": mce,
            }
        )

        all_bin_rows.append(bins)

        fig_path = figures_dir / f"fig_raise_ids_calibration_{dataset.lower().replace('-', '_')}.pdf"
        save_reliability_diagram(bins, dataset, model_name, fig_path)

        print(
            f"Brier={brier:.6f}, Log-loss={ll:.6f}, "
            f"ECE-{args.bins}={ece:.6f}, MCE-{args.bins}={mce:.6f}"
        )
        print(f"Saved figure: {fig_path}")

    calibration_summary = pd.DataFrame(metric_rows)
    calibration_bins_all = pd.concat(all_bin_rows, ignore_index=True)

    out_summary = metrics_dir / "raise_ids_calibration_audit_summary.csv"
    out_bins = metrics_dir / "raise_ids_calibration_bins.csv"

    calibration_summary.sort_values("dataset_order").to_csv(out_summary, index=False)
    calibration_bins_all.sort_values(["dataset_order", "bin"]).to_csv(out_bins, index=False)

    combined_fig = figures_dir / "fig_raise_ids_calibration_reliability_diagrams.pdf"
    save_combined_reliability_diagram(calibration_bins_all, combined_fig)

    latex_table = tables_dir / "table_raise_ids_calibration_audit.tex"
    write_latex_calibration_table(calibration_summary, latex_table)

    print("\nSaved:")
    print(out_summary)
    print(out_bins)
    print(combined_fig)
    print(latex_table)

    print("\nCalibration audit summary:")
    print(calibration_summary.sort_values("dataset_order").to_string(index=False))

    print("\nCalibration bins:")
    print(calibration_bins_all.sort_values(["dataset_order", "bin"]).to_string(index=False))


if __name__ == "__main__":
    main()