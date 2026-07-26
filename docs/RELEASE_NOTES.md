# Release notes: v1.0.0 candidate

## Corrected scientific lineage

1. UNSW-NB15 validation selection is frozen to Random Forest (validation F1 0.9706).
2. The all-model held-out matrix is the sole predictive source for Q, O, G, and Core.
3. The Stage-1 shortlist is the top three models by Core for each dataset.
4. The final practical-effect threshold is 0.0005 absolute F1.
5. CICIDS2017 LightGBM and XGBoost receive neutral statistical support because their reported difference is practically negligible.
6. Every Stage-2 table, operational-cost table, and weight-sensitivity result is regenerated under this rule.
7. Historical validation-reference SHAP generations are excluded from final scoring.

## Frozen Stage-2 winners

- NSL-KDD: XGBoost
- UNSW-NB15: XGBoost
- CICIDS2017: LightGBM

## Weight-sensitivity winner frequencies

- NSL-KDD XGBoost: 1.0000
- UNSW-NB15 XGBoost: 0.8646
- CICIDS2017 LightGBM: 1.0000

## Limitations

This release is a deterministic derived-result freeze. Raw benchmark datasets and large intermediate/seed-wise arrays are excluded. Full end-to-end retraining therefore still depends on the original repository pipeline and locally acquired public datasets.
