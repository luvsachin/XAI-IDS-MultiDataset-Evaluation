# RAISE-IDS authoritative evidence freeze

Release candidate: `v1.0.0-raise-ids`

## Frozen decisions

1. **Validation metrics**: the all-model validation matrix (manuscript Table 4).
2. **Independent-test metrics**: the all-model test matrix (formerly Table 31; supplied in the supplementary material).
3. **Stage-1 shortlist**: the three highest Core scores per dataset.
4. **Explanation stability (`E`)**: the shortlisted-model seed audit, with the same 20 seeds, top-20 cutoff, and fixed 3,000-row explanation sample for every shortlisted candidate.
5. **Statistical support (`T`)**: Holm-corrected Wilcoxon and McNemar evidence with `delta_min = 0.0005` absolute F1. Models not directly compared receive neutral support `T=0.50`.
6. **Stage-2 weights**: Core 0.60, E 0.25, T 0.15.
7. **Weight sensitivity**: 5,000 deterministic draws, seed 42; raw weights drawn from Core [0.45, 0.75], E [0.15, 0.35], T [0.05, 0.25], then normalized.

## Authoritative winners

| Dataset | Validation winner | Test F1 winner | Stage-2 winner |
|---|---|---|---|
| NSL-KDD | LightGBM | XGBoost | XGBoost |
| UNSW-NB15 | Random Forest | XGBoost | XGBoost |
| CICIDS2017 | LightGBM | LightGBM | LightGBM |

## Generation policy

The all-model matrix is the sole source for validation/test metrics and Core scores. Earlier single-model promotion outputs and validation-reference SHAP runs are retained as supplementary historical diagnostics but are not used in final scoring. The CICIDS2017 credibility audits are independent stress tests and do not enter RAISE-IDS.

No model was retrained by the release-freeze script. It deterministically recomputes derived RAISE-IDS tables and figures from measured values already present in the project artifacts.
