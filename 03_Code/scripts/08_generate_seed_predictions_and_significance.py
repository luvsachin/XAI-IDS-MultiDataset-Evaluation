"""
Generate paired prediction vectors and statistical significance tests for Paper A.

Why this script exists
----------------------
The compact manuscript package does not include the large processed CSV files or
archived prediction vectors. Therefore p-values should not be invented inside the
manuscript. Run this script in the FULL Paper_A_XAI_IDS project folder after the
processed datasets are present under 02_Data/processed/.

Outputs
-------
04_Results/metrics/seed_wise_f1_results.csv
04_Results/metrics/mcnemar_test_results.csv
04_Results/metrics/wilcoxon_seed_f1_results.csv
04_Results/metrics/statistical_significance_summary.csv
06_LaTeX/tables/table_statistical_significance.tex
04_Results/predictions/*.npz  (paired predictions and probabilities)

Recommended final run
---------------------
python 03_Code/scripts/08_generate_seed_predictions_and_significance.py --max-train-rows 0 --seeds 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20

For a quick smoke test
----------------------
python 03_Code/scripts/08_generate_seed_predictions_and_significance.py --max-train-rows 30000
"""

from __future__ import annotations

from pathlib import Path
import argparse
import time
import warnings
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, roc_auc_score, average_precision_score

warnings.filterwarnings("ignore")

try:
    from scipy.stats import wilcoxon, binomtest
except Exception as exc:
    raise ImportError("This script requires scipy. Install with: pip install scipy") from exc

try:
    from lightgbm import LGBMClassifier
except Exception as exc:
    raise ImportError("This script requires lightgbm. Install with: pip install lightgbm") from exc

try:
    from xgboost import XGBClassifier
except Exception as exc:
    raise ImportError("This script requires xgboost. Install with: pip install xgboost") from exc

parser = argparse.ArgumentParser()
parser.add_argument("--max-train-rows", type=int, default=0,
                    help="0 means full training data. Use 30000 for quick smoke testing.")
parser.add_argument("--seeds", type=str, default=",".join(str(i) for i in range(1, 21)),
                    help="Comma-separated random seeds. Default is 20 seeds for reviewer-facing significance testing.")
parser.add_argument("--top-n-models", type=int, default=2,
                    help="Number of validation-ranked models to compare per dataset. Default 2.")
args = parser.parse_args()

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "02_Data" / "processed"
METRICS = ROOT / "04_Results" / "metrics"
PRED_DIR = ROOT / "04_Results" / "predictions"
TABLES = ROOT / "06_LaTeX" / "tables"
METRICS.mkdir(parents=True, exist_ok=True)
PRED_DIR.mkdir(parents=True, exist_ok=True)
TABLES.mkdir(parents=True, exist_ok=True)

SEEDS = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]

DATASETS = {
    "NSL-KDD": PROCESSED,
    "UNSW-NB15": PROCESSED / "UNSW-NB15",
    "CICIDS2017": PROCESSED / "CICIDS2017",
}


def read_y(path: Path) -> pd.Series:
    y_df = pd.read_csv(path)
    if y_df.shape[1] == 1:
        return y_df.iloc[:, 0].astype(int)
    if "label" in y_df.columns:
        return y_df["label"].astype(int)
    return y_df.iloc[:, -1].astype(int)


def load_dataset(dataset_path: Path):
    X_train = pd.read_csv(dataset_path / "X_train_final.csv").astype(np.float32)
    X_test = pd.read_csv(dataset_path / "X_test_final.csv").astype(np.float32)
    y_train = read_y(dataset_path / "y_train_binary.csv")
    y_test = read_y(dataset_path / "y_test_binary.csv")
    return X_train, X_test, y_train, y_test


def make_model(model_name: str, seed: int):
    if model_name == "LogisticRegression":
        return LogisticRegression(max_iter=1000, n_jobs=-1, class_weight="balanced", random_state=seed)
    if model_name == "RandomForest":
        return RandomForestClassifier(n_estimators=200, n_jobs=-1, class_weight="balanced", random_state=seed)
    if model_name == "ExtraTrees":
        return ExtraTreesClassifier(n_estimators=200, n_jobs=-1, class_weight="balanced", random_state=seed)
    if model_name == "LightGBM":
        return LGBMClassifier(n_estimators=500, learning_rate=0.05, num_leaves=64,
                              subsample=0.9, colsample_bytree=0.9, class_weight="balanced",
                              random_state=seed, n_jobs=-1, verbose=-1)
    if model_name == "XGBoost":
        return XGBClassifier(n_estimators=500, learning_rate=0.05, max_depth=8,
                             subsample=0.9, colsample_bytree=0.9,
                             objective="binary:logistic", eval_metric="logloss",
                             tree_method="hist", random_state=seed, n_jobs=-1)
    raise ValueError(f"Unsupported model: {model_name}")


def top_models_for_dataset(dataset: str) -> list[str]:
    val_path = METRICS / "multidataset_validation_results.csv"
    if not val_path.exists():
        raise FileNotFoundError(f"Missing validation results: {val_path}")
    val = pd.read_csv(val_path)
    ranked = (val[val["dataset"] == dataset]
              .sort_values("f1", ascending=False)["model"]
              .drop_duplicates()
              .tolist())
    if len(ranked) < args.top_n_models:
        raise ValueError(f"Need at least {args.top_n_models} models for {dataset}; found {ranked}")
    return ranked[:args.top_n_models]


def metric_record(dataset, model_name, seed, y_true, y_pred, y_score, train_rows, runtime):
    return {
        "dataset": dataset,
        "model": model_name,
        "seed": seed,
        "train_rows": train_rows,
        "test_rows": len(y_true),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_score),
        "pr_auc": average_precision_score(y_true, y_score),
        "runtime_sec": runtime,
    }


seed_metric_records = []
model_predictions = {}

for dataset, dataset_path in DATASETS.items():
    required = ["X_train_final.csv", "X_test_final.csv", "y_train_binary.csv", "y_test_binary.csv"]
    missing = [f for f in required if not (dataset_path / f).exists()]
    if missing:
        print(f"Skipping {dataset}: missing {missing}")
        continue

    models_to_compare = top_models_for_dataset(dataset)
    print(f"\nDataset: {dataset}")
    print(f"Top models selected from validation F1: {models_to_compare}")
    X_train, X_test, y_train, y_test = load_dataset(dataset_path)

    for seed in SEEDS:
        if args.max_train_rows > 0 and len(X_train) > args.max_train_rows:
            idx = X_train.sample(n=args.max_train_rows, random_state=seed).index
            X_fit = X_train.loc[idx].reset_index(drop=True)
            y_fit = y_train.loc[idx].reset_index(drop=True)
        else:
            X_fit = X_train.reset_index(drop=True)
            y_fit = y_train.reset_index(drop=True)

        for model_name in models_to_compare:
            print(f"  Training {model_name}, seed={seed}, rows={len(X_fit)}")
            start = time.time()
            model = make_model(model_name, seed)
            model.fit(X_fit, y_fit)
            runtime = time.time() - start
            y_pred = model.predict(X_test)
            if hasattr(model, "predict_proba"):
                y_score = model.predict_proba(X_test)[:, 1]
            else:
                y_score = y_pred.astype(float)

            seed_metric_records.append(metric_record(dataset, model_name, seed, y_test, y_pred, y_score, len(X_fit), runtime))
            key = (dataset, model_name, seed)
            model_predictions[key] = (y_pred.astype(np.int8), y_score.astype(np.float32))
            safe_dataset = dataset.lower().replace("-", "_")
            safe_model = model_name.lower().replace(" ", "_")
            np.savez_compressed(PRED_DIR / f"pred_{safe_dataset}_{safe_model}_seed{seed}.npz",
                                y_true=y_test.to_numpy(dtype=np.int8),
                                y_pred=y_pred.astype(np.int8),
                                y_score=y_score.astype(np.float32))

seed_df = pd.DataFrame(seed_metric_records)
seed_df.to_csv(METRICS / "seed_wise_f1_results.csv", index=False)

# Wilcoxon signed-rank test across seed-wise F1 values for top two models per dataset.
wilcoxon_records = []
for dataset in seed_df["dataset"].unique():
    models = top_models_for_dataset(dataset)
    if len(models) < 2:
        continue
    a, b = models[:2]
    a_scores = seed_df[(seed_df["dataset"] == dataset) & (seed_df["model"] == a)].sort_values("seed")["f1"].to_numpy()
    b_scores = seed_df[(seed_df["dataset"] == dataset) & (seed_df["model"] == b)].sort_values("seed")["f1"].to_numpy()
    try:
        stat, p_value = wilcoxon(a_scores, b_scores, zero_method="wilcox", alternative="two-sided")
    except ValueError:
        stat, p_value = np.nan, np.nan
    wilcoxon_records.append({
        "dataset": dataset,
        "model_a": a,
        "model_b": b,
        "test": "Wilcoxon signed-rank on seed-wise F1",
        "n_seeds": len(a_scores),
        "model_a_mean_f1": float(np.mean(a_scores)),
        "model_b_mean_f1": float(np.mean(b_scores)),
        "mean_f1_difference_a_minus_b": float(np.mean(a_scores - b_scores)),
        "statistic": stat,
        "p_value": p_value,
    })
wilcoxon_df = pd.DataFrame(wilcoxon_records)
wilcoxon_df.to_csv(METRICS / "wilcoxon_seed_f1_results.csv", index=False)

# Exact McNemar test on paired correctness vectors from seed 42 if available, otherwise first seed.
mcnemar_records = []
chosen_seed = 42 if 42 in SEEDS else SEEDS[0]
for dataset in seed_df["dataset"].unique():
    models = top_models_for_dataset(dataset)
    if len(models) < 2:
        continue
    a, b = models[:2]
    safe_dataset_path = DATASETS[dataset]
    y_true = read_y(safe_dataset_path / "y_test_binary.csv").to_numpy(dtype=np.int8)
    pred_a, _ = model_predictions[(dataset, a, chosen_seed)]
    pred_b, _ = model_predictions[(dataset, b, chosen_seed)]
    correct_a = pred_a == y_true
    correct_b = pred_b == y_true
    b_count = int(np.sum(correct_a & ~correct_b))
    c_count = int(np.sum(~correct_a & correct_b))
    n_discordant = b_count + c_count
    if n_discordant > 0:
        p_value = binomtest(min(b_count, c_count), n_discordant, p=0.5, alternative="two-sided").pvalue
    else:
        p_value = 1.0
    mcnemar_records.append({
        "dataset": dataset,
        "model_a": a,
        "model_b": b,
        "seed": chosen_seed,
        "test": "Exact McNemar on paired test predictions",
        "a_correct_b_wrong": b_count,
        "a_wrong_b_correct": c_count,
        "discordant_pairs": n_discordant,
        "p_value": p_value,
    })
mcnemar_df = pd.DataFrame(mcnemar_records)
mcnemar_df.to_csv(METRICS / "mcnemar_test_results.csv", index=False)

# Merge concise significance summary.
summary = pd.merge(
    wilcoxon_df[["dataset", "model_a", "model_b", "model_a_mean_f1", "model_b_mean_f1", "mean_f1_difference_a_minus_b", "p_value"]].rename(columns={"p_value":"wilcoxon_p_value"}),
    mcnemar_df[["dataset", "a_correct_b_wrong", "a_wrong_b_correct", "discordant_pairs", "p_value"]].rename(columns={"p_value":"mcnemar_p_value"}),
    on="dataset",
    how="outer"
)
summary.to_csv(METRICS / "statistical_significance_summary.csv", index=False)

latex = summary.copy()
for col in ["model_a_mean_f1", "model_b_mean_f1", "mean_f1_difference_a_minus_b", "wilcoxon_p_value", "mcnemar_p_value"]:
    if col in latex.columns:
        latex[col] = latex[col].astype(float).round(4)
latex_text = latex.to_latex(index=False,
    caption="Statistical comparison between the top two validation-ranked models for each dataset.",
    label="tab:statistical_significance",
    float_format="%.4f")
(TABLES / "table_statistical_significance.tex").write_text(latex_text, encoding="utf-8")

print("\nSaved statistical testing outputs:")
print(METRICS / "seed_wise_f1_results.csv")
print(METRICS / "wilcoxon_seed_f1_results.csv")
print(METRICS / "mcnemar_test_results.csv")
print(METRICS / "statistical_significance_summary.csv")
print(TABLES / "table_statistical_significance.tex")
print("\nSummary:")
print(summary.to_string(index=False))
