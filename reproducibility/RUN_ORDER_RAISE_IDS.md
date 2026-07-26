# RAISE-IDS Reproducibility Run Order

This document lists the recommended execution order for reproducing the RAISE-IDS evidence package.

## Core training and baseline metrics

1. Prepare processed datasets under:
   - `02_Data/processed`
   - `02_Data/processed/UNSW-NB15`
   - `02_Data/processed/CICIDS2017`

2. Run the original preprocessing/training scripts already supplied in `03_Code/scripts`.

3. Ensure these core metric files exist:
   - `04_Results/metrics/multidataset_validation_results_full.csv`
   - `04_Results/metrics/multidataset_test_results_full.csv`
   - `04_Results/metrics/statistical_significance_summary.csv`
   - `04_Results/metrics/raise_ids_stage2_refined_scores.csv`
   - `04_Results/metrics/raise_ids_stage2_top_model_summary.csv`
   - `04_Results/metrics/raise_ids_shortlist_seed_top_features.csv`

## RAISE-IDS strengthening scripts

Run the following scripts in order:

```bash
python 03_Code/scripts/23_raise_ids_formal_statistical_support.py
python 03_Code/scripts/24_raise_ids_o_cost_sensitivity.py
python 03_Code/scripts/25_raise_ids_mcda_baseline_comparison.py
python 03_Code/scripts/26_raise_ids_final_winner_shap.py
python 03_Code/scripts/27_raise_ids_rank_sensitive_shap_stability.py
python 03_Code/scripts/28_raise_ids_statistical_and_full_model_tables.py
python 03_Code/scripts/29_raise_ids_calibration_analysis.py
python 03_Code/scripts/30_raise_ids_tuned_mlp_baseline.py
python 03_Code/scripts/31_cicids2017_leave_one_file_out_audit.py --max-rows-per-file 30000
python 03_Code/scripts/32_generate_reproducibility_package.py
```

## Notes

- Script 31 requires raw CICIDS2017 CSV files.
- Raw datasets are not redistributed in this artifact.
- Leave-one-file-out CICIDS2017 audit is a stress test, not a RAISE-IDS scoring component.
- Calibration audit is auxiliary operational evidence, not a model-selection component.
