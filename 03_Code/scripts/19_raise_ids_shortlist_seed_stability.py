from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import shap

from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier


DEFAULT_SEEDS = [42, 123, 2024, 31415, 27182]


# -----------------------------------
# Path helpers
# -----------------------------------
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
    y_val = find_first_existing(dir_path, ["y_val_binary.csv", "y_val.csv"])
    y_test = find_first_existing(dir_path, ["y_test_binary.csv", "y_test.csv"])
    return x_ok and y_train is not None and y_val is not None and y_test is not None


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
        raise FileNotFoundError(
            f"No processed dataset directories found under: {search_root}"
        )

    aliases = dataset_aliases(dataset_name)
    scored: List[Tuple[int, Path]] = []

    for c in candidates:
        path_str = str(c).lower().replace("\\", "/")
        score = 0
        for alias in aliases:
            alias_norm = alias.replace(" ", "").replace("-", "").replace("_", "")
            path_norm = path_str.replace(" ", "").replace("-", "").replace("_", "")
            if alias_norm in path_norm:
                score += 10
        if "processed" in path_str:
            score += 2
        scored.append((score, c))

    scored.sort(key=lambda x: (-x[0], str(x[1])))
    best_score, best_dir = scored[0]

    if best_score <= 0:
        raise FileNotFoundError(
            f"Could not confidently match processed directory for dataset '{dataset_name}'"
        )
    return best_dir


def load_dataset_splits(root: Path, dataset_name: str) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    proc_dir = find_processed_dataset_dir(root, dataset_name)

    x_train = read_csv(proc_dir / "X_train_final.csv")
    x_test = read_csv(proc_dir / "X_test_final.csv")

    y_train_path = find_first_existing(proc_dir, ["y_train_binary.csv", "y_train.csv"])
    y_test_path = find_first_existing(proc_dir, ["y_test_binary.csv", "y_test.csv"])

    if y_train_path is None or y_test_path is None:
        raise FileNotFoundError(f"Missing y files in processed directory: {proc_dir}")

    y_train = read_csv(y_train_path).iloc[:, 0]
    y_test = read_csv(y_test_path).iloc[:, 0]

    return x_train, y_train, x_test, y_test


# -----------------------------------
# Normalization helpers
# -----------------------------------
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
    }
    return mapping.get(s, str(x).strip())


# -----------------------------------
# Models
# -----------------------------------
def build_model(model_name: str, seed: int):
    if model_name == "RandomForest":
        return RandomForestClassifier(
            n_estimators=300,
            random_state=seed,
            n_jobs=-1,
            class_weight="balanced_subsample",
        )
    if model_name == "ExtraTrees":
        return ExtraTreesClassifier(
            n_estimators=300,
            random_state=seed,
            n_jobs=-1,
            class_weight="balanced",
        )
    if model_name == "LightGBM":
        return LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            random_state=seed,
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
            random_state=seed,
            n_jobs=-1,
        )
    raise ValueError(f"Unsupported shortlist model: {model_name}")


# -----------------------------------
# SHAP helpers
# -----------------------------------
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
        # possible shapes:
        # (n_samples, n_features, 2) or (2, n_samples, n_features)
        if arr.shape[-1] == 2:
            arr = arr[:, :, 1]
        elif arr.shape[0] == 2:
            arr = arr[1]
        else:
            raise ValueError(f"Unexpected 3D SHAP array shape: {arr.shape}")

    if arr.ndim != 2:
        raise ValueError(f"Unexpected SHAP array shape after normalization: {arr.shape}")

    return arr


def top_k_features_by_seed(
    model,
    x_shap: pd.DataFrame,
    feature_names: List[str],
    top_k: int,
) -> Tuple[List[str], Dict[str, float]]:
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(x_shap)
    arr = extract_shap_array(shap_values)

    mean_abs = np.abs(arr).mean(axis=0)
    importance = dict(zip(feature_names, mean_abs))

    top_features = (
        pd.Series(importance)
        .sort_values(ascending=False)
        .head(top_k)
        .index.tolist()
    )
    return top_features, importance


def jaccard(a: List[str], b: List[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / len(sa | sb)


# -----------------------------------
# Main
# -----------------------------------
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--max-train-rows", type=int, default=100000)
    parser.add_argument("--shap-rows", type=int, default=1000)
    args = parser.parse_args()

    root = project_root_from_script()
    metrics_dir = root / "04_Results" / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    shortlist_path = metrics_dir / "raise_ids_stage2_shortlist.csv"
    shortlist = read_csv(shortlist_path)
    shortlist["dataset"] = shortlist["dataset"].map(normalize_dataset_name)
    shortlist["model"] = shortlist["model"].map(normalize_model_name)

    seeds = DEFAULT_SEEDS

    pairwise_rows = []
    summary_rows = []
    top_feature_rows = []

    for dataset, grp in shortlist.groupby("dataset"):
        models = grp.sort_values("shortlist_rank")["model"].tolist()

        print(f"\n=== Dataset: {dataset} ===")
        print(f"Shortlist models: {models}")

        x_train, y_train, x_test, _ = load_dataset_splits(root, dataset)
        print(f"Using processed train/test shapes: {x_train.shape}, {x_test.shape}")

        if args.max_train_rows > 0 and len(x_train) > args.max_train_rows:
            x_train_use = x_train.sample(args.max_train_rows, random_state=42)
            y_train_use = y_train.loc[x_train_use.index]
            x_train_use = x_train_use.reset_index(drop=True)
            y_train_use = y_train_use.reset_index(drop=True)
        else:
            x_train_use = x_train.reset_index(drop=True)
            y_train_use = y_train.reset_index(drop=True)

        if args.shap_rows > 0 and len(x_test) > args.shap_rows:
            x_shap = x_test.sample(args.shap_rows, random_state=42).reset_index(drop=True)
        else:
            x_shap = x_test.reset_index(drop=True)

        print(f"Training rows used: {len(x_train_use)}, SHAP rows used: {len(x_shap)}")

        for model_name in models:
            print(f"  Processing {model_name}...")

            per_seed_features: Dict[int, List[str]] = {}

            for seed in seeds:
                model = build_model(model_name, seed)
                model.fit(x_train_use, y_train_use)

                top_features, importance = top_k_features_by_seed(
                    model=model,
                    x_shap=x_shap,
                    feature_names=list(x_shap.columns),
                    top_k=args.top_k,
                )
                per_seed_features[seed] = top_features

                for rank_idx, feat in enumerate(top_features, start=1):
                    top_feature_rows.append(
                        {
                            "dataset": dataset,
                            "model": model_name,
                            "seed": seed,
                            "top_k": args.top_k,
                            "rank": rank_idx,
                            "feature": feat,
                            "mean_abs_shap": float(importance[feat]),
                        }
                    )

            # Pairwise Jaccard
            pair_scores = []
            for s1, s2 in combinations(seeds, 2):
                j = jaccard(per_seed_features[s1], per_seed_features[s2])
                pair_scores.append(j)
                pairwise_rows.append(
                    {
                        "dataset": dataset,
                        "model": model_name,
                        "top_k": args.top_k,
                        "seed_a": s1,
                        "seed_b": s2,
                        "pairwise_jaccard": j,
                    }
                )

            intersection = set(per_seed_features[seeds[0]])
            for s in seeds[1:]:
                intersection &= set(per_seed_features[s])

            summary_rows.append(
                {
                    "dataset": dataset,
                    "model": model_name,
                    "top_k": args.top_k,
                    "num_seeds": len(seeds),
                    "mean_pairwise_jaccard": float(np.mean(pair_scores)),
                    "min_pairwise_jaccard": float(np.min(pair_scores)),
                    "max_pairwise_jaccard": float(np.max(pair_scores)),
                    "features_stable_in_all_seeds": len(intersection),
                    "stable_feature_names": " | ".join(sorted(intersection)),
                }
            )

            print(
                f"    mean Jaccard={np.mean(pair_scores):.4f}, "
                f"min={np.min(pair_scores):.4f}, max={np.max(pair_scores):.4f}, "
                f"stable_all_seeds={len(intersection)}"
            )

    df_pairwise = pd.DataFrame(pairwise_rows)
    df_summary = pd.DataFrame(summary_rows)
    df_top = pd.DataFrame(top_feature_rows)

    pairwise_out = metrics_dir / "raise_ids_shortlist_seed_stability_pairwise.csv"
    summary_out = metrics_dir / "raise_ids_shortlist_seed_stability_summary.csv"
    top_out = metrics_dir / "raise_ids_shortlist_seed_top_features.csv"

    df_pairwise.to_csv(pairwise_out, index=False)
    df_summary.to_csv(summary_out, index=False)
    df_top.to_csv(top_out, index=False)

    print("\nSaved:")
    print(pairwise_out)
    print(summary_out)
    print(top_out)

    print("\nSeed stability summary:")
    print(df_summary.sort_values(["dataset", "mean_pairwise_jaccard"], ascending=[True, False]).to_string(index=False))


if __name__ == "__main__":
    main()