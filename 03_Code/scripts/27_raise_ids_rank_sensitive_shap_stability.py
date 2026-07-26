from __future__ import annotations

from pathlib import Path
from itertools import combinations
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------
K_VALUES = [10, 20, 30]
DEFAULT_K = 20


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


def find_col(df: pd.DataFrame, candidates: List[str], required: bool = True) -> Optional[str]:
    cols = list(df.columns)

    for c in candidates:
        if c in cols:
            return c

    for c in candidates:
        tokens = c.split("_")
        for col in cols:
            if all(t in col for t in tokens):
                return col

    if required:
        raise KeyError(f"Could not find any of {candidates}. Available columns: {cols}")
    return None


def jaccard(a: set, b: set) -> float:
    union = a | b
    if not union:
        return np.nan
    return len(a & b) / len(union)


def weighted_jaccard(rank_a: Dict[str, int], rank_b: Dict[str, int], k: int) -> float:
    """
    Rank-weighted Jaccard.

    Feature weight = 1 / rank.
    Shared features contribute min(weight_a, weight_b).
    Union features contribute max(weight_a, weight_b).
    """
    features = set(rank_a) | set(rank_b)
    if not features:
        return np.nan

    num = 0.0
    den = 0.0

    for f in features:
        wa = 1.0 / rank_a[f] if f in rank_a and rank_a[f] <= k else 0.0
        wb = 1.0 / rank_b[f] if f in rank_b and rank_b[f] <= k else 0.0
        num += min(wa, wb)
        den += max(wa, wb)

    return num / den if den > 0 else np.nan


def rank_biased_overlap(rank_a: List[str], rank_b: List[str], p: float = 0.90) -> float:
    """
    Finite-depth Rank-Biased Overlap approximation.

    This gives high weight to agreement near the top of the ranking.
    """
    k = min(len(rank_a), len(rank_b))
    if k == 0:
        return np.nan

    seen_a = set()
    seen_b = set()
    score = 0.0

    for d in range(1, k + 1):
        seen_a.add(rank_a[d - 1])
        seen_b.add(rank_b[d - 1])
        agreement = len(seen_a & seen_b) / d
        score += (1 - p) * (p ** (d - 1)) * agreement

    # Residual extrapolation using agreement at depth k
    final_agreement = len(set(rank_a[:k]) & set(rank_b[:k])) / k
    score += (p ** k) * final_agreement

    return float(score)


def spearman_footrule_similarity(rank_a: Dict[str, int], rank_b: Dict[str, int], k: int) -> float:
    """
    Top-k rank similarity based on normalized Spearman footrule distance.

    Missing features receive rank k+1. Similarity lies approximately in [0,1],
    where 1 means identical top-k ranking.
    """
    features = set(rank_a) | set(rank_b)
    if not features:
        return np.nan

    max_rank = k + 1
    distance = 0.0

    for f in features:
        ra = rank_a.get(f, max_rank)
        rb = rank_b.get(f, max_rank)
        distance += abs(ra - rb)

    # Conservative normalizer for union up to 2k features.
    max_distance = max(1.0, 2 * k * k)
    sim = 1.0 - (distance / max_distance)
    return float(np.clip(sim, 0.0, 1.0))


def summarize_values(values: List[float]) -> Dict[str, float]:
    arr = np.array([v for v in values if pd.notna(v)], dtype=float)

    if arr.size == 0:
        return {
            "mean": np.nan,
            "std": np.nan,
            "min": np.nan,
            "max": np.nan,
            "median": np.nan,
        }

    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        "min": float(arr.min()),
        "max": float(arr.max()),
        "median": float(np.median(arr)),
    }


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


# ---------------------------------------------------------------------
# Input loading
# ---------------------------------------------------------------------
def load_seed_top_features(metrics_dir: Path) -> pd.DataFrame:
    """
    Tries to use existing seed-level top-feature outputs from Script 19.

    Expected file:
    raise_ids_shortlist_seed_top_features.csv

    The script tries to be robust to column-name variations.
    """
    path = metrics_dir / "raise_ids_shortlist_seed_top_features.csv"
    df = read_csv(path)

    # Normalize column names but keep original values.
    original_cols = list(df.columns)
    df.columns = [c.strip().lower().replace(" ", "_").replace("-", "_") for c in df.columns]

    dataset_col = find_col(df, ["dataset"])
    model_col = find_col(df, ["model"])
    seed_col = find_col(df, ["seed", "random_seed", "random_state"])
    feature_col = find_col(df, ["feature", "feature_name"])
    rank_col = find_col(df, ["rank", "feature_rank", "shap_rank"], required=False)
    importance_col = find_col(
        df,
        ["mean_abs_shap", "importance", "shap_importance", "mean_abs_shap_value"],
        required=False,
    )

    out = pd.DataFrame()
    out["dataset"] = df[dataset_col].map(normalize_dataset_name)
    out["model"] = df[model_col].map(normalize_model_name)
    out["seed"] = df[seed_col]
    out["feature"] = df[feature_col].astype(str)

    if rank_col is not None:
        out["rank"] = pd.to_numeric(df[rank_col], errors="coerce")
    else:
        # If rank absent but feature rows are ordered, infer rank per dataset-model-seed.
        out["rank"] = (
            out.groupby(["dataset", "model", "seed"]).cumcount() + 1
        )

    if importance_col is not None:
        out["importance"] = pd.to_numeric(df[importance_col], errors="coerce")
    else:
        out["importance"] = np.nan

    out = out.dropna(subset=["rank"])
    out["rank"] = out["rank"].astype(int)

    print(f"Loaded seed top features from: {path}")
    print(f"Original columns: {original_cols}")
    print(f"Normalized rows: {len(out)}")

    return out


def load_final_winners(metrics_dir: Path) -> pd.DataFrame:
    path = metrics_dir / "raise_ids_stage2_top_model_summary_formal_T.csv"
    df = read_csv(path)

    df["dataset"] = df["dataset"].map(normalize_dataset_name)
    df["formal_T_stage2_winner"] = df["formal_T_stage2_winner"].map(normalize_model_name)

    return df[["dataset", "formal_T_stage2_winner"]].rename(
        columns={"formal_T_stage2_winner": "model"}
    )


# ---------------------------------------------------------------------
# Stability computation
# ---------------------------------------------------------------------
def compute_pairwise_rank_metrics(seed_features: pd.DataFrame, k_values: List[int]) -> pd.DataFrame:
    rows = []

    for (dataset, model), grp in seed_features.groupby(["dataset", "model"]):
        seeds = sorted(grp["seed"].unique())

        if len(seeds) < 2:
            continue

        rankings_by_seed: Dict[object, pd.DataFrame] = {}
        for seed in seeds:
            g = grp[grp["seed"] == seed].sort_values("rank")
            rankings_by_seed[seed] = g

        for k in k_values:
            for seed_a, seed_b in combinations(seeds, 2):
                ga = rankings_by_seed[seed_a].head(k)
                gb = rankings_by_seed[seed_b].head(k)

                list_a = ga["feature"].tolist()
                list_b = gb["feature"].tolist()

                set_a = set(list_a)
                set_b = set(list_b)

                rank_a = {f: i + 1 for i, f in enumerate(list_a)}
                rank_b = {f: i + 1 for i, f in enumerate(list_b)}

                rows.append(
                    {
                        "dataset": dataset,
                        "model": model,
                        "k": k,
                        "seed_a": seed_a,
                        "seed_b": seed_b,
                        "jaccard": jaccard(set_a, set_b),
                        "weighted_jaccard": weighted_jaccard(rank_a, rank_b, k),
                        "rank_biased_overlap_p90": rank_biased_overlap(list_a, list_b, p=0.90),
                        "spearman_footrule_similarity": spearman_footrule_similarity(rank_a, rank_b, k),
                        "intersection_count": len(set_a & set_b),
                        "union_count": len(set_a | set_b),
                    }
                )

    return pd.DataFrame(rows)


def summarize_pairwise_metrics(pairwise: pd.DataFrame) -> pd.DataFrame:
    rows = []

    metrics = [
        "jaccard",
        "weighted_jaccard",
        "rank_biased_overlap_p90",
        "spearman_footrule_similarity",
    ]

    for (dataset, model, k), grp in pairwise.groupby(["dataset", "model", "k"]):
        row = {
            "dataset": dataset,
            "model": model,
            "k": k,
            "pair_count": len(grp),
            "mean_intersection_count": grp["intersection_count"].mean(),
        }

        for metric in metrics:
            stats = summarize_values(grp[metric].tolist())
            for stat_name, value in stats.items():
                row[f"{metric}_{stat_name}"] = value

        rows.append(row)

    return pd.DataFrame(rows)


def compute_magnitude_stability(seed_features: pd.DataFrame, k_values: List[int]) -> pd.DataFrame:
    """
    Computes SHAP magnitude stability if importance is available.
    For each dataset-model-k, considers the union of top-k features across seeds,
    then computes coefficient of variation across seeds for each feature's importance.
    Missing feature in a seed receives zero importance.
    """
    if seed_features["importance"].isna().all():
        return pd.DataFrame()

    rows = []

    for (dataset, model), grp in seed_features.groupby(["dataset", "model"]):
        seeds = sorted(grp["seed"].unique())
        if len(seeds) < 2:
            continue

        for k in k_values:
            top_grp = grp[grp["rank"] <= k]
            features = sorted(top_grp["feature"].unique())

            if not features:
                continue

            matrix = []
            for seed in seeds:
                seed_g = top_grp[top_grp["seed"] == seed]
                imp_map = dict(zip(seed_g["feature"], seed_g["importance"]))
                matrix.append([float(imp_map.get(f, 0.0)) for f in features])

            arr = np.array(matrix, dtype=float)
            feature_means = arr.mean(axis=0)
            feature_stds = arr.std(axis=0, ddof=1)

            cv = np.divide(
                feature_stds,
                feature_means + 1e-12,
                out=np.zeros_like(feature_stds),
                where=(feature_means + 1e-12) != 0,
            )

            rows.append(
                {
                    "dataset": dataset,
                    "model": model,
                    "k": k,
                    "seed_count": len(seeds),
                    "union_feature_count": len(features),
                    "mean_importance_cv": float(np.mean(cv)),
                    "median_importance_cv": float(np.median(cv)),
                    "max_importance_cv": float(np.max(cv)),
                    "features_present_all_seeds": int(
                        sum(
                            (arr[:, j] > 0).sum() == len(seeds)
                            for j in range(arr.shape[1])
                        )
                    ),
                }
            )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# LaTeX tables
# ---------------------------------------------------------------------
def write_rank_stability_latex(summary: pd.DataFrame, final_winners: pd.DataFrame, path: Path) -> None:
    df = summary.merge(final_winners.assign(is_final_winner=True), on=["dataset", "model"], how="left")
    df["is_final_winner"] = df["is_final_winner"].fillna(False)

    # Keep final winners only for main manuscript table.
    df = df[df["is_final_winner"]].copy()

    lines = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Rank-sensitive SHAP stability for final RAISE-IDS winners under alternative top-$k$ cutoffs. Jaccard measures set overlap, weighted Jaccard and rank-biased overlap give more weight to highly ranked features, and footrule similarity penalizes rank displacement.}")
    lines.append(r"\label{tab:raise_ids_rank_sensitive_shap_stability}")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{llrrrrr}")
    lines.append(r"\hline")
    lines.append(r"\textbf{Dataset} & \textbf{Model} & \textbf{$k$} & \textbf{Jaccard} & \textbf{Weighted Jac.} & \textbf{RBO} & \textbf{Footrule sim.} \\")
    lines.append(r"\hline")

    for _, row in df.sort_values(["dataset", "k"]).iterrows():
        lines.append(
            f"{latex_escape(row['dataset'])} & "
            f"{latex_escape(row['model'])} & "
            f"{int(row['k'])} & "
            f"{fmt(row['jaccard_mean'])} & "
            f"{fmt(row['weighted_jaccard_mean'])} & "
            f"{fmt(row['rank_biased_overlap_p90_mean'])} & "
            f"{fmt(row['spearman_footrule_similarity_mean'])} \\\\"
        )

    lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table*}")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def write_topk_sensitivity_latex(summary: pd.DataFrame, final_winners: pd.DataFrame, path: Path) -> None:
    df = summary.merge(final_winners.assign(is_final_winner=True), on=["dataset", "model"], how="left")
    df["is_final_winner"] = df["is_final_winner"].fillna(False)
    df = df[df["is_final_winner"]].copy()

    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Top-$k$ sensitivity of SHAP Jaccard stability for the final RAISE-IDS winners.}")
    lines.append(r"\label{tab:raise_ids_topk_shap_sensitivity}")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{llrrrr}")
    lines.append(r"\hline")
    lines.append(r"\textbf{Dataset} & \textbf{Model} & \textbf{$k=10$} & \textbf{$k=20$} & \textbf{$k=30$} & \textbf{Range} \\")
    lines.append(r"\hline")

    for (dataset, model), grp in df.groupby(["dataset", "model"]):
        vals = {}
        for k in K_VALUES:
            sub = grp[grp["k"] == k]
            vals[k] = float(sub["jaccard_mean"].iloc[0]) if not sub.empty else np.nan

        val_range = np.nanmax(list(vals.values())) - np.nanmin(list(vals.values()))
        lines.append(
            f"{latex_escape(dataset)} & "
            f"{latex_escape(model)} & "
            f"{fmt(vals.get(10))} & "
            f"{fmt(vals.get(20))} & "
            f"{fmt(vals.get(30))} & "
            f"{fmt(val_range)} \\\\"
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
    root = project_root_from_script()
    metrics_dir = root / "04_Results" / "metrics"
    tables_dir = root / "06_LaTeX" / "tables"

    metrics_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    seed_features = load_seed_top_features(metrics_dir)
    final_winners = load_final_winners(metrics_dir)

    pairwise = compute_pairwise_rank_metrics(seed_features, K_VALUES)
    summary = summarize_pairwise_metrics(pairwise)
    magnitude = compute_magnitude_stability(seed_features, K_VALUES)

    # Add final-winner flag to output summaries for easier manuscript use.
    summary_flagged = summary.merge(
        final_winners.assign(is_final_winner=True),
        on=["dataset", "model"],
        how="left",
    )
    summary_flagged["is_final_winner"] = summary_flagged["is_final_winner"].fillna(False)

    if not magnitude.empty:
        magnitude_flagged = magnitude.merge(
            final_winners.assign(is_final_winner=True),
            on=["dataset", "model"],
            how="left",
        )
        magnitude_flagged["is_final_winner"] = magnitude_flagged["is_final_winner"].fillna(False)
    else:
        magnitude_flagged = magnitude

    out_pairwise = metrics_dir / "raise_ids_rank_sensitive_shap_pairwise.csv"
    out_summary = metrics_dir / "raise_ids_rank_sensitive_shap_summary.csv"
    out_magnitude = metrics_dir / "raise_ids_shap_magnitude_stability_summary.csv"

    pairwise.to_csv(out_pairwise, index=False)
    summary_flagged.to_csv(out_summary, index=False)
    magnitude_flagged.to_csv(out_magnitude, index=False)

    latex_rank = tables_dir / "table_raise_ids_rank_sensitive_shap_stability.tex"
    latex_topk = tables_dir / "table_raise_ids_topk_shap_sensitivity.tex"

    write_rank_stability_latex(summary, final_winners, latex_rank)
    write_topk_sensitivity_latex(summary, final_winners, latex_topk)

    print("Saved:")
    print(out_pairwise)
    print(out_summary)
    print(out_magnitude)
    print(latex_rank)
    print(latex_topk)

    print("\nRank-sensitive SHAP stability summary:")
    print(
        summary_flagged.sort_values(["dataset", "model", "k"])[
            [
                "dataset",
                "model",
                "k",
                "is_final_winner",
                "pair_count",
                "jaccard_mean",
                "weighted_jaccard_mean",
                "rank_biased_overlap_p90_mean",
                "spearman_footrule_similarity_mean",
                "mean_intersection_count",
            ]
        ].to_string(index=False)
    )

    if not magnitude_flagged.empty:
        print("\nSHAP magnitude stability summary:")
        print(
            magnitude_flagged.sort_values(["dataset", "model", "k"])[
                [
                    "dataset",
                    "model",
                    "k",
                    "is_final_winner",
                    "union_feature_count",
                    "mean_importance_cv",
                    "median_importance_cv",
                    "features_present_all_seeds",
                ]
            ].to_string(index=False)
        )
    else:
        print("\nSHAP magnitude stability was not computed because no importance column was available.")

    print("\nFinal-winner rank-sensitive stability only:")
    final_only = summary_flagged[summary_flagged["is_final_winner"]].copy()
    print(
        final_only.sort_values(["dataset", "k"])[
            [
                "dataset",
                "model",
                "k",
                "jaccard_mean",
                "weighted_jaccard_mean",
                "rank_biased_overlap_p90_mean",
                "spearman_footrule_similarity_mean",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()