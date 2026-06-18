#!/usr/bin/env python3
"""Quick reproducibility/status check for Paper A.
Run from the repository root:
    python 03_Code/scripts/00_reproducibility_check.py
This does not retrain models; it validates that the committed inputs/results exist and are internally consistent.
"""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
required = [
    '02_Data/raw/NSL-KDD/KDDTrain+.txt',
    '02_Data/raw/NSL-KDD/KDDTest+.txt',
    '02_Data/processed/X_train_final.csv',
    '02_Data/processed/X_val_final.csv',
    '02_Data/processed/X_test_final.csv',
    '02_Data/processed/y_train_binary.csv',
    '02_Data/processed/y_val_binary.csv',
    '02_Data/processed/y_test_binary.csv',
    '04_Results/metrics/baseline_validation_results.csv',
    '04_Results/metrics/best_model_test_results.csv',
    '04_Results/metrics/shap_feature_importance_lightgbm.csv',
    '04_Results/metrics/shap_feature_importance_xgboost.csv',
]
missing = [p for p in required if not (ROOT / p).exists()]
if missing:
    raise SystemExit('Missing required files:\n' + '\n'.join(missing))

summary = pd.read_csv(ROOT/'02_Data/processed/preprocessing_summary.csv')
print('Processed split summary:')
print(summary.to_string(index=False))

val = pd.read_csv(ROOT/'04_Results/metrics/baseline_validation_results.csv')
print('\nValidation models ranked by F1:')
print(val.sort_values('f1', ascending=False).to_string(index=False))

test = pd.read_csv(ROOT/'04_Results/metrics/best_model_test_results.csv')
print('\nSelected model independent test performance:')
print(test.to_string(index=False))

lgb = pd.read_csv(ROOT/'04_Results/metrics/shap_feature_importance_lightgbm.csv')
xgb = pd.read_csv(ROOT/'04_Results/metrics/shap_feature_importance_xgboost.csv')
print('\nTop 10 LightGBM SHAP features:')
print(lgb.head(10).to_string(index=False))
print('\nTop 10 XGBoost SHAP features:')
print(xgb.head(10).to_string(index=False))
print('\nStatus check complete. This confirms stored artifacts only; retraining must be run from notebooks/scripts for full reproducibility.')
