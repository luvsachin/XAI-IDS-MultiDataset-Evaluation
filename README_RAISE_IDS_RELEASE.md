# RAISE-IDS Frozen Reliability Release v1.0.0

This overlay extends the public `XAI-IDS-MultiDataset-Evaluation` repository with the frozen RAISE-IDS evidence lineage used for the submission manuscript.

## What this release freezes

- One authoritative all-model validation and held-out metric matrix.
- Stage-1 Core scores and top-three shortlists.
- A final statistical-support rule with `delta_min = 0.0005` F1.
- Stage-2 scores using fixed Core/E/T weights of 0.60/0.25/0.15.
- A deterministic 5,000-draw weight-sensitivity audit using seed 42.
- Operational-cost sensitivity for false-negative weights 0.50 to 0.90.

The release regenerates derived scores and figures from measured values already present in the manuscript artifacts. It does **not** retrain predictive models or recompute raw SHAP values.

## Installation

```bash
python -m pip install numpy matplotlib
```

## Regenerate derived outputs

Run from the repository root after applying this overlay:

```bash
python 03_Code/scripts/50_freeze_authoritative_release.py
python 03_Code/scripts/51_validate_frozen_release.py
```

The validator must return `"status": "PASS"`.

## Frozen decisions

| Dataset | Validation winner | Held-out F1 winner | RAISE-IDS Stage-2 winner |
|---|---|---|---|
| NSL-KDD | LightGBM | XGBoost | XGBoost |
| UNSW-NB15 | Random Forest | XGBoost | XGBoost |
| CICIDS2017 | LightGBM | LightGBM | LightGBM |

## Important evidence boundary

The shortlisted-model explanation-stability values are measured project inputs. This compact release verifies all derived RAISE-IDS calculations, ranks, sensitivity outputs, and hashes, but it does not contain the raw datasets or large seed-wise SHAP arrays required for full model retraining.

See `results_summary/PROVENANCE_AND_FREEZE.md` and `docs/RELEASE_NOTES.md`. This file is intentionally named `README_RAISE_IDS_RELEASE.md` so the existing repository README is preserved when the overlay is applied.
