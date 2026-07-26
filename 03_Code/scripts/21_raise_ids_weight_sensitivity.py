from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


N_SAMPLES = 5000
RANDOM_SEED = 42


def project_root_from_script() -> Path:
    return Path(__file__).resolve().parents[2]


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
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
        "random forest": "RandomForest",
        "randomforest": "RandomForest",
        "extra trees": "ExtraTrees",
        "extratrees": "ExtraTrees",
        "lightgbm": "LightGBM",
        "xgboost": "XGBoost",
        "mlp": "MLP",
        "logistic regression": "LogisticRegression",
        "logisticregression": "LogisticRegression",
    }
    return mapping.get(s, str(x).strip())


def weighted_geom(values: Dict[str, float], weights: Dict[str, float]) -> float:
    prod = 1.0
    total_w = sum(weights.values())
    for k, v in values.items():
        w = weights[k] / total_w
        v = float(np.clip(v, 1e-6, 1.0))
        prod *= v ** w
    return prod


def main() -> None:
    root = project_root_from_script()
    metrics_dir = root / "04_Results" / "metrics"

    in_path = metrics_dir / "raise_ids_stage2_refined_scores.csv"
    df = read_csv(in_path)

    df["dataset"] = df["dataset"].map(normalize_dataset_name)
    df["model"] = df["model"].map(normalize_model_name)

    rng = np.random.default_rng(RANDOM_SEED)

    sample_rows: List[Dict[str, float]] = []
    summary_rows: List[Dict[str, float]] = []
    rank_rows: List[Dict[str, float]] = []

    for dataset, grp in df.groupby("dataset"):
        grp = grp.copy()

        model_names = grp["model"].tolist()
        component_map = {
            row["model"]: {
                "core": float(row["raise_ids_core_score"]),
                "E": float(row["E_seed_stability"]),
                "T": float(row["T_statistical_support"]),
            }
            for _, row in grp.iterrows()
        }

        # Dirichlet-sampled weights over (core, E, T)
        weights = rng.dirichlet(alpha=np.array([3.0, 2.0, 2.0]), size=N_SAMPLES)

        win_counts = {m: 0 for m in model_names}
        first_second_margin = []

        for i, (w_core, w_E, w_T) in enumerate(weights, start=1):
            w = {"core": float(w_core), "E": float(w_E), "T": float(w_T)}

            scores = {}
            for model_name in model_names:
                scores[model_name] = weighted_geom(component_map[model_name], w)

            ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

            top_model, top_score = ranked[0]
            second_model, second_score = ranked[1]

            win_counts[top_model] += 1
            first_second_margin.append(top_score - second_score)

            sample_rows.append(
                {
                    "dataset": dataset,
                    "sample_id": i,
                    "w_core": w["core"],
                    "w_E": w["E"],
                    "w_T": w["T"],
                    "top_model": top_model,
                    "top_score": top_score,
                    "second_model": second_model,
                    "second_score": second_score,
                    "margin_top_minus_second": top_score - second_score,
                }
            )

            # rank rows for each sample/model (optional richer output)
            for rank_idx, (model_name, model_score) in enumerate(ranked, start=1):
                rank_rows.append(
                    {
                        "dataset": dataset,
                        "sample_id": i,
                        "model": model_name,
                        "rank": rank_idx,
                        "score": model_score,
                    }
                )

        total = float(N_SAMPLES)
        for model_name in model_names:
            summary_rows.append(
                {
                    "dataset": dataset,
                    "model": model_name,
                    "top_rank_frequency": win_counts[model_name] / total,
                    "top_rank_count": win_counts[model_name],
                }
            )

        # dataset-level overall summary
        dominant_model = max(win_counts, key=win_counts.get)
        summary_rows.append(
            {
                "dataset": dataset,
                "model": "__DATASET_SUMMARY__",
                "top_rank_frequency": win_counts[dominant_model] / total,
                "top_rank_count": win_counts[dominant_model],
                "dominant_model": dominant_model,
                "mean_margin_top_minus_second": float(np.mean(first_second_margin)),
                "min_margin_top_minus_second": float(np.min(first_second_margin)),
                "max_margin_top_minus_second": float(np.max(first_second_margin)),
            }
        )

    df_samples = pd.DataFrame(sample_rows)
    df_summary = pd.DataFrame(summary_rows)
    df_ranks = pd.DataFrame(rank_rows)

    out_samples = metrics_dir / "raise_ids_weight_sensitivity_samples.csv"
    out_summary = metrics_dir / "raise_ids_weight_sensitivity_summary.csv"
    out_ranks = metrics_dir / "raise_ids_weight_sensitivity_rank_distribution.csv"

    df_samples.to_csv(out_samples, index=False)
    df_summary.to_csv(out_summary, index=False)
    df_ranks.to_csv(out_ranks, index=False)

    print("Saved:")
    print(out_samples)
    print(out_summary)
    print(out_ranks)

    print("\nWeight-sensitivity summary:")
    print(df_summary.to_string(index=False))


if __name__ == "__main__":
    main()