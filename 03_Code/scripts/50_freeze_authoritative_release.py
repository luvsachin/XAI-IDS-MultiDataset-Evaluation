#!/usr/bin/env python3
"""Freeze the authoritative RAISE-IDS result lineage and regenerate derived outputs.

This script uses only numerical values already present in the accepted manuscript
artifacts. It does not retrain models or create new experimental observations.

Frozen decisions
----------------
1. Validation metrics: Table 4 / table_validation_all_models.tex.
2. Independent-test metrics: full all-model table (Table 31).
3. Stage-1 shortlist: top three models by Core score for each dataset.
4. Stage-2 explanation stability E: shortlisted-model values already reported in
   the Stage-2 table.
5. Statistical support T: practical-effect threshold delta_min = 0.0005 F1.
   - both corrected tests + effect >= threshold: favoured=1.00, comparator=0.15
   - one corrected test + effect >= threshold: favoured=0.75, comparator=0.30
   - negligible/no evidence/not directly tested: neutral=0.50
6. Weight sensitivity: 5,000 deterministic bounded perturbations, seed=42.
   Raw weights are sampled uniformly from Core [0.45,0.75], E [0.15,0.35],
   T [0.05,0.25], then normalized to sum to one.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv
import hashlib
import json
import math

import matplotlib.pyplot as plt
import numpy as np

SCRIPT_PATH = Path(__file__).resolve()
ROOT = next(
    parent for parent in (SCRIPT_PATH.parents[1], SCRIPT_PATH.parents[2], SCRIPT_PATH.parents[3])
    if (parent / "results_summary").exists() or (parent / "tables").exists()
)
if (ROOT / "04_Results" / "frozen_release").exists():
    TABLES = ROOT / "04_Results" / "frozen_release" / "tables"
    FIGURES = ROOT / "04_Results" / "frozen_release" / "figures"
else:
    TABLES = ROOT / "tables"
    FIGURES = ROOT / "figures"
RESULTS = ROOT / "results_summary"
for p in (TABLES, FIGURES, RESULTS):
    p.mkdir(parents=True, exist_ok=True)

DELTA_MIN = 0.0005
WEIGHT_SEED = 42
WEIGHT_DRAWS = 5000
WEIGHT_RANGES = {
    "Core": (0.45, 0.75),
    "E": (0.15, 0.35),
    "T": (0.05, 0.25),
}

@dataclass(frozen=True)
class Metrics:
    val_f1: float
    test_f1: float
    pr_auc: float
    roc_auc: float
    fpr: float
    fnr: float

METRICS: dict[str, dict[str, Metrics]] = {
    "CICIDS2017": {
        "LightGBM": Metrics(0.9995, 0.9995, 1.0000, 1.0000, 0.0000, 0.0006),
        "XGBoost": Metrics(0.9995, 0.9994, 0.9999, 1.0000, 0.0000, 0.0007),
        "Random Forest": Metrics(0.9989, 0.9991, 0.9998, 0.9999, 0.0000, 0.0014),
        "Extra Trees": Metrics(0.9987, 0.9988, 0.9997, 0.9999, 0.0001, 0.0015),
        "MLP": Metrics(0.9822, 0.9835, 0.9984, 0.9998, 0.0007, 0.0263),
        "Logistic Regression": Metrics(0.7245, 0.7256, 0.9495, 0.9927, 0.0862, 0.0117),
    },
    "NSL-KDD": {
        "LightGBM": Metrics(0.9994, 0.7680, 0.9642, 0.9589, 0.0276, 0.3637),
        "XGBoost": Metrics(0.9991, 0.7844, 0.9708, 0.9671, 0.0277, 0.3412),
        "Random Forest": Metrics(0.9989, 0.7461, 0.9645, 0.9612, 0.0266, 0.3930),
        "Extra Trees": Metrics(0.9988, 0.7791, 0.9595, 0.9560, 0.0281, 0.3482),
        "MLP": Metrics(0.9683, 0.7473, 0.9016, 0.8603, 0.0334, 0.3884),
        "Logistic Regression": Metrics(0.8686, 0.6670, 0.8980, 0.8692, 0.0960, 0.4633),
    },
    "UNSW-NB15": {
        "Random Forest": Metrics(0.9706, 0.8942, 0.9846, 0.9807, 0.2711, 0.0125),
        "Extra Trees": Metrics(0.9697, 0.8894, 0.9715, 0.9730, 0.2791, 0.0167),
        "LightGBM": Metrics(0.9692, 0.8971, 0.9892, 0.9853, 0.2580, 0.0152),
        "XGBoost": Metrics(0.9687, 0.8984, 0.9888, 0.9849, 0.2573, 0.0133),
        "MLP": Metrics(0.9628, 0.8813, 0.9808, 0.9740, 0.2912, 0.0250),
        "Logistic Regression": Metrics(0.9504, 0.8622, 0.9671, 0.9560, 0.2787, 0.0698),
    },
}

# Stage-2 stability values already reported for the shortlisted candidates.
E_VALUES = {
    "CICIDS2017": {"LightGBM": 1.0000, "XGBoost": 0.8701, "Random Forest": 0.8449},
    "NSL-KDD": {"XGBoost": 0.9247, "Extra Trees": 0.9429, "LightGBM": 1.0000},
    "UNSW-NB15": {"XGBoost": 0.8039, "LightGBM": 1.0000, "Random Forest": 0.8442},
}

# Final T values after applying the frozen practical-effect rule.
T_VALUES = {
    "CICIDS2017": {"LightGBM": 0.50, "XGBoost": 0.50, "Random Forest": 0.50},
    "NSL-KDD": {"XGBoost": 1.00, "Extra Trees": 0.50, "LightGBM": 0.15},
    "UNSW-NB15": {"XGBoost": 1.00, "LightGBM": 0.50, "Random Forest": 0.15},
}

DISPLAY_ORDER = {
    "CICIDS2017": ["LightGBM", "XGBoost", "Random Forest"],
    "NSL-KDD": ["XGBoost", "Extra Trees", "LightGBM"],
    "UNSW-NB15": ["XGBoost", "LightGBM", "Random Forest"],
}

VALIDATION_WINNER = {ds: max(models, key=lambda m: models[m].val_f1) for ds, models in METRICS.items()}
TEST_WINNER = {ds: max(models, key=lambda m: models[m].test_f1) for ds, models in METRICS.items()}


def core_score(metric: Metrics, lambda_fnr: float = 0.70) -> tuple[float, float, float, float]:
    q = math.sqrt(metric.test_f1 * metric.pr_auc)
    o = 1.0 - (lambda_fnr * metric.fnr + (1.0 - lambda_fnr) * metric.fpr)
    g = 1.0 - max(0.0, metric.val_f1 - metric.test_f1)
    core = (q ** 0.45) * (o ** 0.35) * (g ** 0.20)
    return core, q, o, g


def stage2_score(core: float, e: float, t: float, weights: tuple[float, float, float] = (0.60, 0.25, 0.15)) -> float:
    wc, we, wt = weights
    return (core ** wc) * (e ** we) * (t ** wt)


def shortlist() -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for ds, models in METRICS.items():
        ranked = sorted(models, key=lambda m: core_score(models[m])[0], reverse=True)
        out[ds] = ranked[:3]
    return out

SHORTLIST = shortlist()


def fmt_model(model: str) -> str:
    return model


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def tex_escape(s: str) -> str:
    return s.replace("_", r"\_")

# ---------------------------------------------------------------------------
# Freeze data lineage tables.
# ---------------------------------------------------------------------------
lineage_rows: list[dict[str, object]] = []
for ds, models in METRICS.items():
    for model, m in models.items():
        c, q, o, g = core_score(m)
        lineage_rows.append({
            "dataset": ds,
            "model": model,
            "validation_f1": f"{m.val_f1:.4f}",
            "test_f1": f"{m.test_f1:.4f}",
            "test_pr_auc": f"{m.pr_auc:.4f}",
            "test_roc_auc": f"{m.roc_auc:.4f}",
            "fpr": f"{m.fpr:.4f}",
            "fnr": f"{m.fnr:.4f}",
            "Q": f"{q:.6f}",
            "O": f"{o:.6f}",
            "G": f"{g:.6f}",
            "core": f"{c:.6f}",
            "authoritative_source": "Table 4 validation + Table 31 independent test",
        })
write_csv(RESULTS / "authoritative_metric_lineage.csv", list(lineage_rows[0].keys()), lineage_rows)

# Validation winners promoted to the authoritative test matrix.
lines = [
    r"\begin{table}[!htbp]",
    r"\centering",
    r"\caption{Validation winners evaluated on the frozen authoritative independent-test result set. All values are drawn from the single frozen all-model result generation documented in Table~\ref{tab:result_generation_provenance}.}",
    r"\label{tab:best_model_summary}",
    r"\small",
    r"\begin{tabular}{llrrrr}",
    r"\toprule",
    r"Dataset & Validation winner & Validation F1 & Test F1 & Test ROC-AUC & Test PR-AUC \\",
    r"\midrule",
]
for ds in ("NSL-KDD", "UNSW-NB15", "CICIDS2017"):
    model = VALIDATION_WINNER[ds]
    m = METRICS[ds][model]
    lines.append(f"{ds} & {model} & {m.val_f1:.4f} & {m.test_f1:.4f} & {m.roc_auc:.4f} & {m.pr_auc:.4f} " + r"\\")
lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
(TABLES / "table_best_model_summary.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

# Final winner top-five SHAP table, taken from existing final-winner top-ten table.
FINAL_SHAP = {
    "NSL-KDD": ("XGBoost", [
        ("src_bytes", 2.7568), ("dst_host_srv_count", 1.1014),
        ("dst_bytes", 0.9918), ("count", 0.8643),
        ("dst_host_same_src_port_rate", 0.5333),
    ]),
    "UNSW-NB15": ("XGBoost", [
        ("sttl", 2.8264), ("ct_dst_sport_ltm", 0.6840),
        ("ct_state_ttl", 0.5012), ("sbytes", 0.4918),
        ("smean", 0.4290),
    ]),
    "CICIDS2017": ("LightGBM", [
        ("Destination Port", 1.8230), ("Init_Win_bytes_forward", 1.5713),
        ("Fwd Packet Length Mean", 1.4051),
        ("Total Length of Fwd Packets", 0.8365), ("Source Port", 0.7304),
    ]),
}
lines = [
    r"\begin{table}[!htbp]", r"\centering",
    r"\caption{Top five SHAP-ranked features for the final Stage-2 RAISE-IDS winner on each dataset. These values are a compact subset of Table~\ref{tab:raise_ids_final_winner_shap_top_features}.}",
    r"\label{tab:shap_top5}", r"\small", r"\begin{tabularx}{\textwidth}{llcXr}", r"\toprule",
    r"Dataset & Final winner & Rank & Feature & Mean $|\mathrm{SHAP}|$ \\", r"\midrule",
]
for ds in ("NSL-KDD", "UNSW-NB15", "CICIDS2017"):
    model, feats = FINAL_SHAP[ds]
    for rank, (feat, value) in enumerate(feats, 1):
        feature_tex = r"\texttt{" + tex_escape(feat) + "}" if " " not in feat else feat
        lines.append(f"{ds} & {model} & {rank} & {feature_tex} & {value:.4f} " + r"\\")
lines += [r"\bottomrule", r"\end{tabularx}", r"\end{table}"]
(TABLES / "table_shap_top5_multidataset.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

# Stage-2 table.
stage_rows: list[dict[str, object]] = []
lines = [
    r"\begin{table}[!htbp]", r"\centering",
    r"\caption{Frozen Stage-2 RAISE-IDS scores for the top-three Stage-1 candidates. Core is computed from the authoritative test matrix; $E$ is the shortlisted-model seed-wise SHAP stability; $T$ follows Table~\ref{tab:raise_ids_t_mapping}.}",
    r"\label{tab:raise_ids_stage2_scores}", r"\small", r"\begin{tabular}{llrrrrr}", r"\toprule",
    r"Dataset & Model & Core & $E$ & $T$ & Stage 2 & Rank \\", r"\midrule",
]
for ds in ("CICIDS2017", "NSL-KDD", "UNSW-NB15"):
    scores = []
    for model in DISPLAY_ORDER[ds]:
        c = core_score(METRICS[ds][model])[0]
        s = stage2_score(c, E_VALUES[ds][model], T_VALUES[ds][model])
        scores.append((model, c, E_VALUES[ds][model], T_VALUES[ds][model], s))
    scores.sort(key=lambda x: x[4], reverse=True)
    for rank, (model, c, e, t, s) in enumerate(scores, 1):
        lines.append(f"{ds} & {model} & {c:.4f} & {e:.4f} & {t:.2f} & {s:.4f} & {rank} " + r"\\")
        stage_rows.append({"dataset": ds, "model": model, "core": f"{c:.6f}", "E": f"{e:.6f}", "T": f"{t:.2f}", "stage2": f"{s:.6f}", "rank": rank})
lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
(TABLES / "table_raise_ids_stage2_scores.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
write_csv(RESULTS / "frozen_stage2_scores.csv", list(stage_rows[0].keys()), stage_rows)

# T mapping table with explicit threshold.
(TABLES / "table_raise_ids_t_mapping.tex").write_text(r"""\begin{table}[!htbp]
\centering
\caption{Frozen mapping from corrected statistical evidence to the RAISE-IDS statistical-support coefficient $T$. The practical-effect threshold is fixed at $\delta_{\min}=0.0005$ absolute F1. Candidates not directly involved in the designated leading comparison receive neutral support.}
\label{tab:raise_ids_t_mapping}
\small
\begin{tabularx}{\textwidth}{Xcc}
\toprule
Evidence condition & Favored model & Comparator \\
\midrule
Favored by both Holm-corrected Wilcoxon and McNemar tests, with $|\Delta F1|\geq 0.0005$ & 1.00 & 0.15 \\
Favored by one corrected test, with $|\Delta F1|\geq 0.0005$ & 0.75 & 0.30 \\
No corrected statistical support or statistically indistinguishable & 0.50 & 0.50 \\
Statistically detectable but $|\Delta F1|<0.0005$ & 0.50 & 0.50 \\
Candidate not directly tested in the designated leading comparison & 0.50 & 0.50 \\
\bottomrule
\end{tabularx}
\end{table}
""", encoding="utf-8")

# Weight sensitivity.
rng = np.random.default_rng(WEIGHT_SEED)
raw = np.column_stack([
    rng.uniform(*WEIGHT_RANGES["Core"], size=WEIGHT_DRAWS),
    rng.uniform(*WEIGHT_RANGES["E"], size=WEIGHT_DRAWS),
    rng.uniform(*WEIGHT_RANGES["T"], size=WEIGHT_DRAWS),
])
weights = raw / raw.sum(axis=1, keepdims=True)
weight_rows = []
frequencies: dict[str, dict[str, float]] = {}
mean_margins: dict[str, float] = {}
for ds in ("NSL-KDD", "UNSW-NB15", "CICIDS2017"):
    models = DISPLAY_ORDER[ds]
    matrix = np.zeros((WEIGHT_DRAWS, len(models)))
    for j, model in enumerate(models):
        c = core_score(METRICS[ds][model])[0]
        e = E_VALUES[ds][model]
        t = T_VALUES[ds][model]
        matrix[:, j] = (c ** weights[:, 0]) * (e ** weights[:, 1]) * (t ** weights[:, 2])
    winner_idx = matrix.argmax(axis=1)
    frequencies[ds] = {model: float(np.mean(winner_idx == j)) for j, model in enumerate(models)}
    sorted_scores = np.sort(matrix, axis=1)
    mean_margins[ds] = float(np.mean(sorted_scores[:, -1] - sorted_scores[:, -2]))
    for model in models:
        weight_rows.append({"dataset": ds, "model": model, "top_rank_frequency": f"{frequencies[ds][model]:.4f}"})
write_csv(RESULTS / "frozen_weight_sensitivity_frequencies.csv", list(weight_rows[0].keys()), weight_rows)

lines = [
    r"\begin{table}[!htbp]", r"\centering",
    r"\caption{Weight-sensitivity analysis of frozen RAISE-IDS Stage-2 selection under 5,000 deterministic bounded perturbations (seed 42). Raw weights are sampled from Core $[0.45,0.75]$, $E$ $[0.15,0.35]$, and $T$ $[0.05,0.25]$, then normalized.}",
    r"\label{tab:raise_ids_weight_sensitivity}", r"\small", r"\begin{tabular}{llrr}", r"\toprule",
    r"Dataset & Dominant model & Top-rank frequency & Mean margin \\", r"\midrule",
]
for ds in ("CICIDS2017", "NSL-KDD", "UNSW-NB15"):
    dominant = max(frequencies[ds], key=frequencies[ds].get)
    lines.append(f"{ds} & {dominant} & {frequencies[ds][dominant]:.4f} & {mean_margins[ds]:.4f} " + r"\\")
lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
(TABLES / "table_raise_ids_weight_sensitivity.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

# Operational-cost sensitivity.
cost_rows = []
lines = [
    r"\begin{table}[!htbp]", r"\centering",
    r"\caption{Operational-cost sensitivity of frozen RAISE-IDS Stage-2 winners under alternative false-negative weights $\lambda$ in $O_\lambda=1-(\lambda FNR+(1-\lambda)FPR)$.}",
    r"\label{tab:raise_ids_o_cost_sensitivity}", r"\small", r"\begin{tabular}{lrlr}", r"\toprule",
    r"Dataset & $\lambda$ & Stage-2 winner & Margin \\", r"\midrule",
]
for ds in ("CICIDS2017", "NSL-KDD", "UNSW-NB15"):
    for lam in (0.50, 0.60, 0.70, 0.80, 0.90):
        scores = {}
        for model in DISPLAY_ORDER[ds]:
            c = core_score(METRICS[ds][model], lambda_fnr=lam)[0]
            scores[model] = stage2_score(c, E_VALUES[ds][model], T_VALUES[ds][model])
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        margin = ranked[0][1] - ranked[1][1]
        lines.append(f"{ds} & {lam:.2f} & {ranked[0][0]} & {margin:.4f} " + r"\\")
        cost_rows.append({"dataset": ds, "lambda": f"{lam:.2f}", "winner": ranked[0][0], "margin": f"{margin:.6f}"})
lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
(TABLES / "table_raise_ids_o_cost_sensitivity.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
write_csv(RESULTS / "frozen_operational_cost_sensitivity.csv", list(cost_rows[0].keys()), cost_rows)

# ---------------------------------------------------------------------------
# Figures.
# ---------------------------------------------------------------------------
plt.rcParams.update({"font.size": 9})

# Stage-2 score figure.
fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.6))
for ax, ds in zip(axes, ("NSL-KDD", "UNSW-NB15", "CICIDS2017")):
    models = DISPLAY_ORDER[ds]
    vals = [stage2_score(core_score(METRICS[ds][m])[0], E_VALUES[ds][m], T_VALUES[ds][m]) for m in models]
    ax.bar(models, vals)
    ax.set_title(ds)
    ax.set_ylabel("RAISE-IDS Stage-2 score")
    ax.set_ylim(0, 1.02)
    ax.tick_params(axis="x", rotation=25)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.015, f"{v:.3f}", ha="center", fontsize=8)
fig.suptitle("Frozen RAISE-IDS Stage-2 scores for shortlisted models")
fig.tight_layout()
fig.savefig(FIGURES / "fig_raise_ids_stage2_scores.pdf", bbox_inches="tight")
plt.close(fig)

# Weight sensitivity figure.
fig, ax = plt.subplots(figsize=(8.2, 4.6))
datasets = ["NSL-KDD", "UNSW-NB15", "CICIDS2017"]
all_models = ["Extra Trees", "LightGBM", "Random Forest", "XGBoost"]
x = np.arange(len(datasets))
width = 0.18
for idx, model in enumerate(all_models):
    vals = [frequencies[ds].get(model, 0.0) for ds in datasets]
    ax.bar(x + (idx - 1.5) * width, vals, width=width, label=model)
ax.set_xticks(x)
ax.set_xticklabels(datasets)
ax.set_ylabel("Top-rank frequency")
ax.set_ylim(0, 1.05)
ax.set_title("Weight sensitivity of frozen RAISE-IDS Stage-2 selection")
ax.legend()
fig.tight_layout()
fig.savefig(FIGURES / "fig_raise_ids_weight_sensitivity.pdf", bbox_inches="tight")
plt.close(fig)

# Validation-winner generalization figure.
fig, ax = plt.subplots(figsize=(7.2, 4.8))
for ds in ("NSL-KDD", "UNSW-NB15", "CICIDS2017"):
    model = VALIDATION_WINNER[ds]
    m = METRICS[ds][model]
    ax.plot([0, 1], [m.val_f1, m.test_f1], marker="o", label=f"{ds} ({model})")
ax.set_xticks([0, 1])
ax.set_xticklabels(["Validation F1", "Independent-test F1"])
ax.set_ylabel("F1-score")
ax.set_ylim(0.70, 1.02)
ax.set_title("Validation-to-test generalization gap for validation winners")
ax.legend()
fig.tight_layout()
fig.savefig(FIGURES / "fig_generalization_slope.pdf", bbox_inches="tight")
plt.close(fig)

# Validation-winner FPR/FNR figure.
fig, ax = plt.subplots(figsize=(7.2, 4.8))
labels = ["NSL-KDD", "UNSW-NB15", "CICIDS2017"]
fpr = [METRICS[ds][VALIDATION_WINNER[ds]].fpr for ds in labels]
fnr = [METRICS[ds][VALIDATION_WINNER[ds]].fnr for ds in labels]
x = np.arange(len(labels))
ax.bar(x - 0.18, fpr, width=0.36, label="FPR")
ax.bar(x + 0.18, fnr, width=0.36, label="FNR")
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylabel("Error rate")
ax.set_title("Validation-winner error rates on the authoritative test set")
ax.legend()
fig.tight_layout()
fig.savefig(FIGURES / "fig_test_error_rates.pdf", bbox_inches="tight")
plt.close(fig)

# Operational cost sensitivity figure.
fig, axes = plt.subplots(1, 3, figsize=(10.6, 3.6))
for ax, ds in zip(axes, ("NSL-KDD", "UNSW-NB15", "CICIDS2017")):
    lambdas = np.array([0.50, 0.60, 0.70, 0.80, 0.90])
    winner_vals, runner_vals = [], []
    for lam in lambdas:
        scores = {}
        for model in DISPLAY_ORDER[ds]:
            c = core_score(METRICS[ds][model], lambda_fnr=float(lam))[0]
            scores[model] = stage2_score(c, E_VALUES[ds][model], T_VALUES[ds][model])
        ranked = sorted(scores.values(), reverse=True)
        winner_vals.append(ranked[0])
        runner_vals.append(ranked[1])
    ax.plot(lambdas, winner_vals, marker="o", label="Winner score")
    ax.plot(lambdas, runner_vals, marker="s", label="Second score")
    ax.set_title(ds)
    ax.set_xlabel(r"$\lambda$ in $O_\lambda$")
    ax.set_ylabel("Stage-2 score")
    ax.legend(fontsize=7)
fig.suptitle("Frozen RAISE-IDS sensitivity to false-negative operational-cost weight")
fig.tight_layout()
fig.savefig(FIGURES / "fig_raise_ids_o_cost_sensitivity.pdf", bbox_inches="tight")
plt.close(fig)

# Release metadata.
release = {
    "release": "v1.0.0-raise-ids-candidate",
    "delta_min_f1": DELTA_MIN,
    "weight_sensitivity": {
        "draws": WEIGHT_DRAWS,
        "seed": WEIGHT_SEED,
        "raw_ranges": WEIGHT_RANGES,
        "normalization": "divide each raw vector by its sum",
    },
    "validation_winners": VALIDATION_WINNER,
    "test_winners": TEST_WINNER,
    "shortlist": SHORTLIST,
    "stage2_winners": {},
    "weight_frequencies": frequencies,
}
for ds in DISPLAY_ORDER:
    winner = max(DISPLAY_ORDER[ds], key=lambda m: stage2_score(core_score(METRICS[ds][m])[0], E_VALUES[ds][m], T_VALUES[ds][m]))
    release["stage2_winners"][ds] = winner
(RESULTS / "frozen_release_metadata.json").write_text(json.dumps(release, indent=2), encoding="utf-8")

# Manifest of files generated by this script.
generated = [
    RESULTS / "authoritative_metric_lineage.csv",
    RESULTS / "frozen_stage2_scores.csv",
    RESULTS / "frozen_weight_sensitivity_frequencies.csv",
    RESULTS / "frozen_operational_cost_sensitivity.csv",
    RESULTS / "frozen_release_metadata.json",
    TABLES / "table_best_model_summary.tex",
    TABLES / "table_shap_top5_multidataset.tex",
    TABLES / "table_raise_ids_stage2_scores.tex",
    TABLES / "table_raise_ids_t_mapping.tex",
    TABLES / "table_raise_ids_weight_sensitivity.tex",
    TABLES / "table_raise_ids_o_cost_sensitivity.tex",
    FIGURES / "fig_raise_ids_stage2_scores.pdf",
    FIGURES / "fig_raise_ids_weight_sensitivity.pdf",
    FIGURES / "fig_generalization_slope.pdf",
    FIGURES / "fig_test_error_rates.pdf",
    FIGURES / "fig_raise_ids_o_cost_sensitivity.pdf",
]
manifest_lines = ["sha256  relative_path"]
for path in generated:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest_lines.append(f"{digest}  {path.relative_to(ROOT)}")
(RESULTS / "frozen_generated_manifest.sha256").write_text("\n".join(manifest_lines) + "\n", encoding="utf-8")

print("Frozen authoritative release generated.")
print(json.dumps(release, indent=2))
