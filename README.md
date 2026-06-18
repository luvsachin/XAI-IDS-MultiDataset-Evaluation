# Multi-Dataset XAI-IDS Evaluation Protocol

This repository provides code and reproducibility artifacts for a multi-dataset intrusion detection evaluation study using explainable artificial intelligence (XAI) techniques.

---

## Overview

The study investigates the generalization behavior, interpretability, and reliability of machine learning models for intrusion detection across multiple benchmark datasets. It integrates predictive evaluation with SHAP-based explanation analysis and statistical significance testing to produce reproducible scientific evidence.

---

## Datasets

The following publicly available benchmark datasets are used:

* **NSL-KDD**
* **UNSW-NB15**
* **CICIDS2017**

Due to licensing and size constraints, datasets are not included in this repository. Please download them from their official sources.

---

## Methods

The experimental pipeline includes:

### Machine Learning Models

* Logistic Regression (LR)
* Random Forest (RF)
* Extra Trees (ET)
* LightGBM (LGBM)
* XGBoost (XGB)
* Multi-Layer Perceptron (MLP)

### Explainability

* SHAP (global and local explanations)

### Statistical Analysis

* Wilcoxon signed-rank test
* McNemar’s test
* Seed-wise stability analysis

---

## Key Contributions

* A unified **multi-dataset evaluation protocol** for IDS research
* **Cross-dataset explainability analysis** using SHAP
* **Explanation stability analysis** across multiple random seeds
* **File-wise holdout evaluation** for CICIDS2017 to assess external validity
* Integration of **statistical significance testing** for robust comparison

---

## Repository Structure

```
03_Code/
 ├── scripts/                # Final experimental pipeline (00–15)
 ├── notebooks_exploration/  # Early-stage EDA and prototyping

results_summary/             # Selected key outputs and metrics

requirements.txt             # Python dependencies
README.md                    # Documentation
```

---

## Reproducibility

All experiments are implemented as a sequential pipeline in:

```
03_Code/scripts/
```

### Execution Order

Run scripts in order:

```
00 → 15
```

This includes:

* dataset preparation
* model training
* evaluation
* SHAP analysis
* statistical testing
* robustness and credibility audits

---

## Results Summary

The `results_summary/` directory contains:

* Multi-dataset performance results
* Statistical significance tests
* SHAP stability metrics
* CICIDS2017 credibility and multi-holdout evaluations
* Reproducibility metadata (environment and system details)

---

## Notebooks

The `notebooks_exploration/` folder contains early-stage exploratory analysis and prototyping.

**Note:**
The final results reported in the paper are generated exclusively using the scripts in `03_Code/scripts/`.

---

## Data Availability

This study uses publicly available benchmark datasets.
Users are expected to obtain datasets from official sources and place them in the appropriate directory structure before execution.

---

## Reproducibility Note

This repository provides a **compact reproducibility package** including:

* complete experimental pipeline
* key evaluation outputs
* statistical validation results

Raw datasets and intermediate large files are excluded due to size constraints.

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

---

### 2. Prepare datasets

Download the datasets and place them in the expected directory structure:

```
02_Data/
 ├── NSL-KDD/
 ├── UNSW-NB15/
 ├── CICIDS2017/
```

---

### 3. Run the full pipeline

Execute scripts sequentially:

```bash
python 03_Code/scripts/00_reproducibility_check.py
python 03_Code/scripts/01_dataset_audit.py
...
python 03_Code/scripts/15_cicids2017_multi_holdout_audit.py
```

---

### 4. Outputs

Results will be generated in:

```
04_Results/
```

Key summary outputs are provided in:

```
results_summary/
```


---

## Contact

For questions or clarifications, please contact the corresponding author.
