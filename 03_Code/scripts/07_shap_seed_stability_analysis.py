from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "02_Data" / "processed"
METRICS = ROOT / "04_Results" / "metrics"
FIG_FINAL = ROOT / "05_Figures" / "final"
LATEX_TABLES = ROOT / "06_LaTeX" / "tables"

METRICS.mkdir(parents=True, exist_ok=True)
FIG_FINAL.mkdir(parents=True, exist_ok=True)
LATEX_TABLES.mkdir(parents=True, exist_ok=True)

SEEDS = [11, 21, 42, 63, 84]
MAX_TRAIN_ROWS = 80000
MAX_SHAP_ROWS = 2000
TOP_K = 20

DATASETS = {
    "NSL-KDD": {
        "path": PROCESSED,
        "model_name": "LightGBM"
    },
    "UNSW-NB15": {
        "path": PROCESSED / "UNSW-NB15",
        "model_name": "XGBoost"
    },
    "CICIDS2017": {
        "path": PROCESSED / "CICIDS2017",
        "model_name": "LightGBM"
    }
}


def read_y(path):
    y_df = pd.read_csv(path)
    if y_df.shape[1] == 1:
        return y_df.iloc[:, 0].astype(int)
    if "label" in y_df.columns:
        return y_df["label"].astype(int)
    return y_df.iloc[:, -1].astype(int)


def load_data(dataset_path):
    X_train = pd.read_csv(dataset_path / "X_train_final.csv").astype(np.float32)
    X_test = pd.read_csv(dataset_path / "X_test_final.csv").astype(np.float32)
    y_train = read_y(dataset_path / "y_train_binary.csv")
    y_test = read_y(dataset_path / "y_test_binary.csv")
    return X_train, X_test, y_train, y_test


def make_model(model_name, seed):
    if model_name == "LightGBM":
        return LGBMClassifier(
            n_estimators=500,
            learning_rate=0.05,
            num_leaves=64,
            subsample=0.9,
            colsample_bytree=0.9,
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
            verbose=-1
        )

    if model_name == "XGBoost":
        return XGBClassifier(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=8,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            random_state=seed,
            n_jobs=-1
        )

    raise ValueError(f"Unsupported model: {model_name}")


def get_positive_class_shap_values(explainer, X_sample):
    shap_values = explainer.shap_values(X_sample)

    if isinstance(shap_values, list):
        return shap_values[1]

    if hasattr(shap_values, "values"):
        values = shap_values.values
        if values.ndim == 3:
            return values[:, :, 1]
        return values

    if isinstance(shap_values, np.ndarray):
        if shap_values.ndim == 3:
            return shap_values[:, :, 1]
        return shap_values

    return shap_values


seed_feature_records = []
pairwise_records = []
summary_records = []

for dataset_name, cfg in DATASETS.items():
    print(f"\nSeed-stability analysis for {dataset_name}...")

    dataset_path = cfg["path"]
    model_name = cfg["model_name"]

    X_train, X_test, y_train, y_test = load_data(dataset_path)

    # Fixed SHAP sample for fair comparison across seeds
    if len(X_test) > MAX_SHAP_ROWS:
        shap_idx = X_test.sample(n=MAX_SHAP_ROWS, random_state=42).index
        X_shap = X_test.loc[shap_idx].reset_index(drop=True)
    else:
        X_shap = X_test.reset_index(drop=True)

    top_sets = {}
    rank_maps = {}

    for seed in SEEDS:
        print(f"  Training {model_name}, seed={seed}")

        if len(X_train) > MAX_TRAIN_ROWS:
            train_idx = X_train.sample(n=MAX_TRAIN_ROWS, random_state=seed).index
            X_train_fit = X_train.loc[train_idx].reset_index(drop=True)
            y_train_fit = y_train.loc[train_idx].reset_index(drop=True)
        else:
            X_train_fit = X_train.reset_index(drop=True)
            y_train_fit = y_train.reset_index(drop=True)

        model = make_model(model_name, seed)
        model.fit(X_train_fit, y_train_fit)

        explainer = shap.TreeExplainer(model)
        shap_vals = get_positive_class_shap_values(explainer, X_shap)

        mean_abs = np.abs(shap_vals).mean(axis=0)

        imp = pd.DataFrame({
            "dataset": dataset_name,
            "model": model_name,
            "seed": seed,
            "feature": X_shap.columns,
            "mean_abs_shap": mean_abs
        }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

        imp["rank"] = np.arange(1, len(imp) + 1)

        top = imp.head(TOP_K).copy()
        seed_feature_records.append(top)

        features = top["feature"].tolist()
        top_sets[seed] = set(features)
        rank_maps[seed] = {f: r for f, r in zip(top["feature"], top["rank"])}

        print(f"    top feature: {features[0]}")

    # Pairwise Jaccard across seeds
    for i, s1 in enumerate(SEEDS):
        for s2 in SEEDS[i + 1:]:
            a = top_sets[s1]
            b = top_sets[s2]
            inter = a.intersection(b)
            union = a.union(b)
            jaccard = len(inter) / len(union) if len(union) else 0

            pairwise_records.append({
                "dataset": dataset_name,
                "model": model_name,
                "seed_a": s1,
                "seed_b": s2,
                "top_k": TOP_K,
                "intersection_count": len(inter),
                "union_count": len(union),
                "jaccard_similarity": jaccard,
                "common_features": " | ".join(sorted(inter))
            })

    dataset_pairwise = [r for r in pairwise_records if r["dataset"] == dataset_name]
    vals = [r["jaccard_similarity"] for r in dataset_pairwise]

    # Frequency: how often each feature appears in top-k across seeds
    feature_counts = {}
    for seed in SEEDS:
        for f in top_sets[seed]:
            feature_counts[f] = feature_counts.get(f, 0) + 1

    stable_features = [f for f, c in feature_counts.items() if c == len(SEEDS)]

    summary_records.append({
        "dataset": dataset_name,
        "model": model_name,
        "top_k": TOP_K,
        "num_seeds": len(SEEDS),
        "mean_pairwise_jaccard": float(np.mean(vals)),
        "min_pairwise_jaccard": float(np.min(vals)),
        "max_pairwise_jaccard": float(np.max(vals)),
        "features_stable_in_all_seeds": len(stable_features),
        "stable_feature_names": " | ".join(sorted(stable_features))
    })


seed_feature_df = pd.concat(seed_feature_records, axis=0, ignore_index=True)
pairwise_df = pd.DataFrame(pairwise_records)
summary_df = pd.DataFrame(summary_records)

seed_feature_df.to_csv(METRICS / "shap_seed_top_features.csv", index=False)
pairwise_df.to_csv(METRICS / "shap_seed_stability_pairwise.csv", index=False)
summary_df.to_csv(METRICS / "shap_seed_stability_summary.csv", index=False)

# LaTeX table
latex_summary = summary_df.copy()
for col in ["mean_pairwise_jaccard", "min_pairwise_jaccard", "max_pairwise_jaccard"]:
    latex_summary[col] = latex_summary[col].round(4)

latex_summary_short = latex_summary[
    [
        "dataset",
        "model",
        "top_k",
        "num_seeds",
        "mean_pairwise_jaccard",
        "min_pairwise_jaccard",
        "max_pairwise_jaccard",
        "features_stable_in_all_seeds"
    ]
]

latex_text = latex_summary_short.to_latex(
    index=False,
    caption="Random-seed stability of top-k SHAP explanations across repeated model training runs.",
    label="tab:shap_seed_stability",
    float_format="%.4f"
)

(LATEX_TABLES / "table_shap_seed_stability_summary.tex").write_text(latex_text, encoding="utf-8")

# Figure: mean Jaccard by dataset
plt.figure(figsize=(7, 4.5))
plt.bar(summary_df["dataset"], summary_df["mean_pairwise_jaccard"])
plt.ylim(0, 1.05)
plt.ylabel("Mean pairwise Jaccard similarity")
plt.xlabel("Dataset")
plt.title(f"Random-Seed Stability of Top-{TOP_K} SHAP Features")
plt.tight_layout()
plt.savefig(FIG_FINAL / "fig_shap_seed_stability_by_dataset.pdf", bbox_inches="tight")
plt.close()

print("\nSaved:")
print(METRICS / "shap_seed_top_features.csv")
print(METRICS / "shap_seed_stability_pairwise.csv")
print(METRICS / "shap_seed_stability_summary.csv")
print(LATEX_TABLES / "table_shap_seed_stability_summary.tex")
print(FIG_FINAL / "fig_shap_seed_stability_by_dataset.pdf")

print("\nSummary:")
print(summary_df.to_string(index=False))