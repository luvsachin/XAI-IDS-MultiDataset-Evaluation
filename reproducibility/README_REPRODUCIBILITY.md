# RAISE-IDS Reproducibility Package

Generated: 2026-07-11 10:49:32

## Purpose

This package supports reproducibility for the manuscript:

**Beyond Benchmark Accuracy: A Reliability-Aware Framework for Explainable Intrusion Detection under Dataset Shift and Explanation Instability**

The artifact contains scripts, generated tables, generated figures, and environment descriptors for the RAISE-IDS evaluation workflow.

## System information used when generating this package

- Python executable: environment-dependent; use the Python version specified below.
- Python version: `3.13.5`
- Platform: `Windows-11-10.0.26200-SP0`

## Key reproducibility files

- `reproducibility/requirements_raise_ids.txt`
- `reproducibility/environment_raise_ids.yml`
- `Dockerfile`
- `.dockerignore`
- `reproducibility/RUN_ORDER_RAISE_IDS.md`
- `reproducibility/MANIFEST_RAISE_IDS_ARTIFACTS.csv`

## Data availability note

Raw datasets are not bundled in this artifact because public IDS datasets can be large and may have redistribution restrictions.
The scripts assume that processed datasets are available under the project structure documented in the run-order file.

## Important methodological notes

1. RAISE-IDS Stage 2 uses formalized statistical support after Holm correction.
2. Operational-cost sensitivity is evaluated across false-negative weights from 0.50 to 0.90.
3. MCDA and simple-selection baselines are reported for transparency.
4. Final-winner SHAP explanations are generated for the final RAISE-IDS selected models.
5. Rank-sensitive SHAP stability is reported using multiple top-k values and rank-aware metrics.
6. Calibration analysis is reported as an auxiliary operational audit.
7. The CICIDS2017 leave-one-file-out audit is a stress test and is not used to tune or select models.
