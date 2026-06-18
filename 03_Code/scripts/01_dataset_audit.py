from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "02_Data" / "raw"
OUT = ROOT / "04_Results" / "metrics"
OUT.mkdir(parents=True, exist_ok=True)

records = []

def audit_csv(path, dataset, nrows_preview=5):
    print(f"\nAuditing: {path}")
    try:
        df_head = pd.read_csv(path, nrows=nrows_preview)
        df_full_info = pd.read_csv(path, nrows=10000)
        
        label_candidates = [c for c in df_head.columns if c.strip().lower() in ["label", "class", "attack_cat"]]
        
        record = {
            "dataset": dataset,
            "file": path.name,
            "size_mb": round(path.stat().st_size / (1024 * 1024), 2),
            "n_columns": len(df_head.columns),
            "columns": " | ".join([str(c) for c in df_head.columns]),
            "label_candidates": " | ".join(label_candidates),
            "preview_rows_checked": len(df_full_info),
            "missing_values_in_10k": int(df_full_info.isna().sum().sum()),
            "inf_values_possible": "manual_check_needed"
        }
        
        for lab in label_candidates:
            try:
                record[f"{lab}_value_counts_10k"] = str(df_full_info[lab].value_counts(dropna=False).head(20).to_dict())
            except Exception as e:
                record[f"{lab}_value_counts_10k"] = f"error: {e}"
        
        records.append(record)
        
    except Exception as e:
        records.append({
            "dataset": dataset,
            "file": path.name,
            "error": str(e)
        })

# UNSW-NB15
unsw_dir = RAW / "UNSW-NB15"
for p in sorted(unsw_dir.glob("*.csv")):
    audit_csv(p, "UNSW-NB15")

# CICIDS2017
cic_dir = RAW / "CICIDS2017"
for p in sorted(cic_dir.glob("*.csv")):
    audit_csv(p, "CICIDS2017")

audit_df = pd.DataFrame(records)
audit_path = OUT / "dataset_audit_summary.csv"
audit_df.to_csv(audit_path, index=False)

print("\nSaved audit summary to:")
print(audit_path)

print("\nCompact audit view:")
cols = [c for c in ["dataset", "file", "size_mb", "n_columns", "label_candidates", "missing_values_in_10k"] if c in audit_df.columns]
print(audit_df[cols].to_string(index=False))