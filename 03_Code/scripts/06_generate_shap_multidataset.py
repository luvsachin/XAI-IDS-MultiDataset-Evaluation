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
FIG_RAW = ROOT / "05_Figures" / "raw"
FIG_FINAL = ROOT / "05_Figures" / "final"
LATEX_TABLES = ROOT / "06_LaTeX" / "tables"

METRICS.mkdir(parents=True, exist_ok=True)
FIG_RAW.mkdir(parents=True, exist_ok=True)
FIG_FINAL.mkdir(parents=True, exist_ok=True)
LATEX_TABLES.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
MAX_TRAIN_ROWS = 100000
MAX_SHAP_ROWS = 3000
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


def make_model(model_name):
    if model_name == "LightGBM":
        return LGBMClassifier(
            n_estimators=500,
            learning_rate=0.05,
            num_leaves=64,
            subsample=0.9,
            colsample_bytree=0.9,
            class_weight="balanced",
            random_state=RANDOM_STATE,
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
            random_state=RANDOM_STATE,
            n_jobs=-1
        )

    raise ValueError(f"Unsupported model: {model_name}")


def get_positive_class_shap_values(explainer, X_sample):
    shap_values = explainer.shap_values(X_sample)

    # LightGBM/XGBoost binary models may return either:
    # array: (n_samples, n_features)
    # list: [class0, class1]
    if isinstance(shap_values, list):
        return shap_values[1]

    # SHAP Explanation object
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


all_top_records = []
top_feature_sets = {}

for dataset_name, cfg in DATASETS.items():
    print(f"\nProcessing SHAP for {dataset_name}...")

    dataset_path = cfg["path"]
    model_name = cfg["model_name"]

    X_train, X_test, y_train, y_test = load_data(dataset_path)

    if len(X_train) > MAX_TRAIN_ROWS:
        train_idx = X_train.sample(n=MAX_TRAIN_ROWS, random_state=RANDOM_STATE).index
        X_train_fit = X_train.loc[train_idx].reset_index(drop=True)
        y_train_fit = y_train.loc[train_idx].reset_index(drop=True)
    else:
        X_train_fit = X_train.reset_index(drop=True)
        y_train_fit = y_train.reset_index(drop=True)

    if len(X_test) > MAX_SHAP_ROWS:
        shap_idx = X_test.sample(n=MAX_SHAP_ROWS, random_state=RANDOM_STATE).index
        X_shap = X_test.loc[shap_idx].reset_index(drop=True)
        y_shap = y_test.loc[shap_idx].reset_index(drop=True)
    else:
        X_shap = X_test.reset_index(drop=True)
        y_shap = y_test.reset_index(drop=True)

    print(f"  Training rows used: {len(X_train_fit)}")
    print(f"  SHAP rows used: {len(X_shap)}")
    print(f"  Features: {X_train_fit.shape[1]}")

    model = make_model(model_name)
    model.fit(X_train_fit, y_train_fit)

    explainer = shap.TreeExplainer(model)
    shap_vals = get_positive_class_shap_values(explainer, X_shap)

    mean_abs = np.abs(shap_vals).mean(axis=0)
    feature_importance = pd.DataFrame({
        "dataset": dataset_name,
        "model": model_name,
        "feature": X_shap.columns,
        "mean_abs_shap": mean_abs
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

    feature_importance["rank"] = np.arange(1, len(feature_importance) + 1)
    feature_importance.to_csv(
        METRICS / f"shap_feature_importance_{dataset_name.lower().replace('-', '_')}.csv",
        index=False
    )

    top_df = feature_importance.head(TOP_K).copy()
    all_top_records.append(top_df)

    top_feature_sets[dataset_name] = set(top_df["feature"].tolist())

    # Bar plot top 15
    plot_df = feature_importance.head(15).iloc[::-1]

    plt.figure(figsize=(8, 6))
    plt.barh(plot_df["feature"], plot_df["mean_abs_shap"])
    plt.xlabel("Mean absolute SHAP value")
    plt.ylabel("Feature")
    plt.title(f"Top SHAP Features: {dataset_name} ({model_name})")
    plt.tight_layout()

    safe_name = dataset_name.lower().replace("-", "_")
    plt.savefig(FIG_RAW / f"fig_shap_top_features_{safe_name}.png", dpi=300)
    plt.savefig(FIG_FINAL / f"fig_shap_top_features_{safe_name}.pdf", bbox_inches="tight")
    plt.close()

    # SHAP beeswarm/summary plot for richer visual explanation
    shap.summary_plot(
        shap_vals,
        X_shap,
        show=False,
        max_display=15
    )
    plt.title(f"SHAP Summary Plot: {dataset_name} ({model_name})")
    plt.tight_layout()
    plt.savefig(FIG_RAW / f"fig_shap_summary_{safe_name}.png", dpi=300, bbox_inches="tight")
    plt.savefig(FIG_FINAL / f"fig_shap_summary_{safe_name}.pdf", bbox_inches="tight")
    plt.close()

    print(f"  Saved SHAP outputs for {dataset_name}")


# Combined top feature table
combined_top = pd.concat(all_top_records, axis=0, ignore_index=True)
combined_top.to_csv(METRICS / "shap_top_features_multidataset.csv", index=False)

latex_top = combined_top[["dataset", "model", "rank", "feature", "mean_abs_shap"]].copy()
latex_top["mean_abs_shap"] = latex_top["mean_abs_shap"].round(4)

latex_text = latex_top.to_latex(
    index=False,
    caption="Top SHAP-ranked features for the selected best model on each dataset.",
    label="tab:shap_top_features_multidataset",
    float_format="%.4f"
)
(LATEX_TABLES / "table_shap_top_features_multidataset.tex").write_text(latex_text, encoding="utf-8")


# Feature overlap / explanation stability
datasets = list(top_feature_sets.keys())
overlap_records = []
matrix = pd.DataFrame(index=datasets, columns=datasets, dtype=float)

for d1 in datasets:
    for d2 in datasets:
        s1 = top_feature_sets[d1]
        s2 = top_feature_sets[d2]
        jaccard = len(s1.intersection(s2)) / len(s1.union(s2)) if len(s1.union(s2)) > 0 else 0
        matrix.loc[d1, d2] = jaccard

        overlap_records.append({
            "dataset_a": d1,
            "dataset_b": d2,
            "top_k": TOP_K,
            "intersection_count": len(s1.intersection(s2)),
            "union_count": len(s1.union(s2)),
            "jaccard_similarity": jaccard,
            "common_features": " | ".join(sorted(s1.intersection(s2)))
        })

overlap_df = pd.DataFrame(overlap_records)
overlap_df.to_csv(METRICS / "shap_feature_overlap_matrix.csv", index=False)
matrix.to_csv(METRICS / "shap_feature_overlap_jaccard_matrix.csv")

# Heatmap without seaborn
plt.figure(figsize=(6, 5))
plt.imshow(matrix.astype(float).values, aspect="auto")
plt.colorbar(label="Jaccard similarity")
plt.xticks(range(len(datasets)), datasets, rotation=30, ha="right")
plt.yticks(range(len(datasets)), datasets)
plt.title(f"Top-{TOP_K} SHAP Feature Overlap Across Datasets")

for i in range(len(datasets)):
    for j in range(len(datasets)):
        plt.text(j, i, f"{matrix.iloc[i, j]:.2f}", ha="center", va="center")

plt.tight_layout()
plt.savefig(FIG_RAW / "fig_shap_feature_overlap_heatmap.png", dpi=300)
plt.savefig(FIG_FINAL / "fig_shap_feature_overlap_heatmap.pdf", bbox_inches="tight")
plt.close()

# Stability summary
stability_summary = pd.DataFrame([
    {
        "analysis": "Top-k SHAP feature overlap",
        "top_k": TOP_K,
        "mean_pairwise_jaccard_excluding_diagonal": matrix.where(~np.eye(len(matrix), dtype=bool)).stack().mean(),
        "min_pairwise_jaccard_excluding_diagonal": matrix.where(~np.eye(len(matrix), dtype=bool)).stack().min(),
        "max_pairwise_jaccard_excluding_diagonal": matrix.where(~np.eye(len(matrix), dtype=bool)).stack().max()
    }
])
stability_summary.to_csv(METRICS / "shap_explanation_stability_summary.csv", index=False)

print("\nSaved SHAP metrics:")
print(METRICS / "shap_top_features_multidataset.csv")
print(METRICS / "shap_feature_overlap_matrix.csv")
print(METRICS / "shap_explanation_stability_summary.csv")

print("\nSaved SHAP figures:")
for dataset_name in DATASETS:
    safe_name = dataset_name.lower().replace("-", "_")
    print(FIG_FINAL / f"fig_shap_top_features_{safe_name}.pdf")
    print(FIG_FINAL / f"fig_shap_summary_{safe_name}.pdf")
print(FIG_FINAL / "fig_shap_feature_overlap_heatmap.pdf")