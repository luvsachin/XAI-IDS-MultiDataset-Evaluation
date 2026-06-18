from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
METRICS = ROOT / "04_Results" / "metrics"
FIG_RAW = ROOT / "05_Figures" / "raw"
FIG_FINAL = ROOT / "05_Figures" / "final"
LATEX_TABLES = ROOT / "06_LaTeX" / "tables"

FIG_RAW.mkdir(parents=True, exist_ok=True)
FIG_FINAL.mkdir(parents=True, exist_ok=True)
LATEX_TABLES.mkdir(parents=True, exist_ok=True)

val_path = METRICS / "multidataset_validation_results.csv"
test_path = METRICS / "multidataset_test_results.csv"
best_path = METRICS / "multidataset_best_models.csv"

val = pd.read_csv(val_path)
test = pd.read_csv(test_path)
best = pd.read_csv(best_path)

# -------------------------
# Table 1: Best model summary
# -------------------------
best_table = best.copy()
for col in ["validation_f1", "test_f1", "test_roc_auc", "test_pr_auc"]:
    best_table[col] = best_table[col].round(4)

best_table.to_csv(METRICS / "paper_table_best_model_summary.csv", index=False)

best_latex = best_table.to_latex(
    index=False,
    caption="Best-performing model summary across the three intrusion-detection datasets.",
    label="tab:best_model_summary",
    float_format="%.4f"
)
(LATEX_TABLES / "table_best_model_summary.tex").write_text(best_latex, encoding="utf-8")

# -------------------------
# Table 2: Validation results
# -------------------------
val_cols = [
    "dataset", "model", "accuracy", "precision", "recall",
    "f1", "roc_auc", "pr_auc", "fpr", "fnr",
    "train_time_sec", "inference_ms_per_sample"
]
val_table = val[val_cols].copy()

metric_cols = [
    "accuracy", "precision", "recall", "f1", "roc_auc",
    "pr_auc", "fpr", "fnr", "train_time_sec", "inference_ms_per_sample"
]
for col in metric_cols:
    val_table[col] = val_table[col].round(4)

val_table.to_csv(METRICS / "paper_table_validation_all_models.csv", index=False)

val_latex = val_table.to_latex(
    index=False,
    caption="Validation performance of machine-learning models across NSL-KDD, UNSW-NB15, and CICIDS2017.",
    label="tab:validation_all_models",
    float_format="%.4f"
)
(LATEX_TABLES / "table_validation_all_models.tex").write_text(val_latex, encoding="utf-8")

# -------------------------
# Table 3: Independent test results
# -------------------------
test_cols = [
    "dataset", "model", "accuracy", "precision", "recall",
    "f1", "roc_auc", "pr_auc", "fpr", "fnr",
    "tn", "fp", "fn", "tp", "inference_ms_per_sample"
]
test_table = test[test_cols].copy()

for col in ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "fpr", "fnr", "inference_ms_per_sample"]:
    test_table[col] = test_table[col].round(4)

test_table.to_csv(METRICS / "paper_table_independent_test_results.csv", index=False)

test_latex = test_table.to_latex(
    index=False,
    caption="Independent test performance of the selected best model for each dataset.",
    label="tab:independent_test_results",
    float_format="%.4f"
)
(LATEX_TABLES / "table_independent_test_results.tex").write_text(test_latex, encoding="utf-8")

# -------------------------
# Figure 1: Test F1 by dataset
# -------------------------
plt.figure(figsize=(7, 4.5))
plt.bar(best["dataset"], best["test_f1"])
plt.ylim(0, 1.05)
plt.ylabel("Test F1-score")
plt.xlabel("Dataset")
plt.title("Independent Test F1-score Across Datasets")
plt.tight_layout()
plt.savefig(FIG_RAW / "fig_test_f1_by_dataset.png", dpi=300)
plt.savefig(FIG_FINAL / "fig_test_f1_by_dataset.pdf", bbox_inches="tight")
plt.close()

# -------------------------
# Figure 2: Validation vs Test F1
# -------------------------
plt.figure(figsize=(7, 4.5))
x = range(len(best))
width = 0.35
plt.bar([i - width/2 for i in x], best["validation_f1"], width=width, label="Validation F1")
plt.bar([i + width/2 for i in x], best["test_f1"], width=width, label="Test F1")
plt.xticks(list(x), best["dataset"])
plt.ylim(0, 1.05)
plt.ylabel("F1-score")
plt.xlabel("Dataset")
plt.title("Validation-to-Test Generalization Gap")
plt.legend()
plt.tight_layout()
plt.savefig(FIG_RAW / "fig_validation_test_f1_gap.png", dpi=300)
plt.savefig(FIG_FINAL / "fig_validation_test_f1_gap.pdf", bbox_inches="tight")
plt.close()

# -------------------------
# Figure 3: Test ROC-AUC and PR-AUC
# -------------------------
plt.figure(figsize=(7, 4.5))
x = range(len(best))
width = 0.35
plt.bar([i - width/2 for i in x], best["test_roc_auc"], width=width, label="ROC-AUC")
plt.bar([i + width/2 for i in x], best["test_pr_auc"], width=width, label="PR-AUC")
plt.xticks(list(x), best["dataset"])
plt.ylim(0, 1.05)
plt.ylabel("Score")
plt.xlabel("Dataset")
plt.title("Independent Test ROC-AUC and PR-AUC")
plt.legend()
plt.tight_layout()
plt.savefig(FIG_RAW / "fig_test_auc_comparison.png", dpi=300)
plt.savefig(FIG_FINAL / "fig_test_auc_comparison.pdf", bbox_inches="tight")
plt.close()

# -------------------------
# Figure 4: FPR and FNR
# -------------------------
plt.figure(figsize=(7, 4.5))
x = range(len(test))
width = 0.35
plt.bar([i - width/2 for i in x], test["fpr"], width=width, label="FPR")
plt.bar([i + width/2 for i in x], test["fnr"], width=width, label="FNR")
plt.xticks(list(x), test["dataset"])
plt.ylabel("Error rate")
plt.xlabel("Dataset")
plt.title("False Positive and False Negative Rates")
plt.legend()
plt.tight_layout()
plt.savefig(FIG_RAW / "fig_test_error_rates.png", dpi=300)
plt.savefig(FIG_FINAL / "fig_test_error_rates.pdf", bbox_inches="tight")
plt.close()

print("Generated publication tables:")
print(LATEX_TABLES / "table_best_model_summary.tex")
print(LATEX_TABLES / "table_validation_all_models.tex")
print(LATEX_TABLES / "table_independent_test_results.tex")

print("\nGenerated figures:")
print(FIG_FINAL / "fig_test_f1_by_dataset.pdf")
print(FIG_FINAL / "fig_validation_test_f1_gap.pdf")
print(FIG_FINAL / "fig_test_auc_comparison.pdf")
print(FIG_FINAL / "fig_test_error_rates.pdf")