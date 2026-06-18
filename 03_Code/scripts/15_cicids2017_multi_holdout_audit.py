"""
Additional CICIDS2017 file-wise holdout sensitivity audit.

This script extends the reviewer-facing credibility audit by testing multiple
whole-file holdout combinations. It is intended to be run from the full project
folder where raw CICIDS2017 CSV files are available.

Example:
python 03_Code/scripts/15_cicids2017_multi_holdout_audit.py --rows-per-file 75000 --chunk-size 25000

Outputs:
04_Results/metrics/cicids2017_multi_holdout_audit_results.csv
06_LaTeX/tables/table_cicids2017_multi_holdout_audit.tex
"""
from __future__ import annotations
from pathlib import Path
import argparse, gc, warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, confusion_matrix
warnings.filterwarnings("ignore")
try:
    from lightgbm import LGBMClassifier
except Exception as exc:
    raise ImportError("Install lightgbm before running this script: pip install lightgbm") from exc

parser = argparse.ArgumentParser()
parser.add_argument("--rows-per-file", type=int, default=75000)
parser.add_argument("--chunk-size", type=int, default=25000)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--holdout-groups", type=str, default="Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv|Wednesday-workingHours.pcap_ISCX.csv;Monday-WorkingHours.pcap_ISCX.csv;Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv|Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv;Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv|Tuesday-WorkingHours.pcap_ISCX.csv", help="Semicolon-separated holdout groups; files inside a group are pipe-separated.")
args = parser.parse_args()

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "02_Data" / "raw" / "CICIDS2017"
METRICS = ROOT / "04_Results" / "metrics"
TABLES = ROOT / "06_LaTeX" / "tables"
METRICS.mkdir(parents=True, exist_ok=True); TABLES.mkdir(parents=True, exist_ok=True)
FEATURE_DROP_MISSING_RATIO = 0.50

def clean_chunk(df: pd.DataFrame, source_file: str, global_cols: list[str] | None = None):
    df = df.copy(); df.columns = [str(c).strip() for c in df.columns]
    if "Label" not in df.columns: raise ValueError(f"Label column not found in {source_file}")
    label_text = df["Label"].astype(str).str.strip(); y = (label_text.str.upper() != "BENIGN").astype(np.int8)
    X = df.drop(columns=["Label"]).apply(pd.to_numeric, errors="coerce").replace([np.inf,-np.inf], np.nan)
    miss = X.isna().mean(); X = X.drop(columns=miss[miss > FEATURE_DROP_MISSING_RATIO].index.tolist(), errors="ignore")
    X = X.fillna(X.median(numeric_only=True).fillna(0)).fillna(0)
    if global_cols is not None: X = X.reindex(columns=global_cols, fill_value=0)
    out = X.copy(); out["binary_label"] = y.values; out["source_file"] = source_file; out["original_label"] = label_text.values
    return out

def collect_filewise_data(seed: int):
    csv_files = sorted(RAW.glob("*.csv"))
    if not csv_files: raise FileNotFoundError(f"No CICIDS2017 CSV files found in {RAW}")
    frames=[]; global_cols=None
    for path in csv_files:
        file_frames=[]; file_rows=0
        for chunk in pd.read_csv(path, chunksize=args.chunk_size, low_memory=False, encoding="latin1"):
            clean = clean_chunk(chunk, path.name, global_cols)
            if global_cols is None:
                global_cols = [c for c in clean.columns if c not in ["binary_label","source_file","original_label"]]
            remaining = args.rows_per_file - file_rows
            if remaining <= 0: break
            if len(clean) > remaining: clean = clean.sample(n=remaining, random_state=seed)
            file_frames.append(clean); file_rows += len(clean)
            if file_rows >= args.rows_per_file: break
        if file_frames: frames.append(pd.concat(file_frames, ignore_index=True))
        gc.collect()
    return pd.concat(frames, ignore_index=True)

def split_xy(data):
    y = data["binary_label"].astype(int)
    X = data.drop(columns=["binary_label","source_file","original_label"], errors="ignore")
    return X,y

def evaluate(X_train, X_test, y_train, y_test, seed):
    scaler=StandardScaler(); Xtr=scaler.fit_transform(X_train).astype(np.float32); Xte=scaler.transform(X_test).astype(np.float32)
    model=LGBMClassifier(n_estimators=500, learning_rate=0.05, num_leaves=64, subsample=0.9, colsample_bytree=0.9, class_weight="balanced", random_state=seed, n_jobs=-1, verbose=-1)
    model.fit(Xtr,y_train); pred=model.predict(Xte); score=model.predict_proba(Xte)[:,1]
    tn,fp,fn,tp=confusion_matrix(y_test,pred).ravel()
    return {"accuracy":accuracy_score(y_test,pred),"precision":precision_score(y_test,pred,zero_division=0),"recall":recall_score(y_test,pred,zero_division=0),"f1":f1_score(y_test,pred,zero_division=0),"roc_auc":roc_auc_score(y_test,score),"pr_auc":average_precision_score(y_test,score),"tn":int(tn),"fp":int(fp),"fn":int(fn),"tp":int(tp),"fpr":fp/(fp+tn) if fp+tn else 0,"fnr":fn/(fn+tp) if fn+tp else 0}

file_data = collect_filewise_data(args.seed)
rows=[]
for group_id, group in enumerate([g for g in args.holdout_groups.split(';') if g.strip()], start=1):
    holdout = {x.strip() for x in group.split('|') if x.strip()}
    train_data=file_data[~file_data["source_file"].isin(holdout)].reset_index(drop=True)
    test_data=file_data[file_data["source_file"].isin(holdout)].reset_index(drop=True)
    Xtr,ytr=split_xy(train_data); Xte,yte=split_xy(test_data)
    rec={"protocol":"file_holdout_sensitivity","group_id":group_id,"seed":args.seed,"holdout_files":" | ".join(sorted(holdout)),"train_rows":len(Xtr),"test_rows":len(Xte),"train_attack_ratio":float(ytr.mean()),"test_attack_ratio":float(yte.mean())}
    rec.update(evaluate(Xtr,Xte,ytr,yte,args.seed)); rows.append(rec)
    print(f"Completed holdout group {group_id}: {rec['holdout_files']} F1={rec['f1']:.4f} FNR={rec['fnr']:.4f}")
res=pd.DataFrame(rows)
res.to_csv(METRICS/"cicids2017_multi_holdout_audit_results.csv", index=False)
tex=res.copy()
for c in ["train_attack_ratio","test_attack_ratio","accuracy","precision","recall","f1","roc_auc","pr_auc","fpr","fnr"]:
    tex[c]=tex[c].astype(float).round(4)
cols=["group_id","holdout_files","test_attack_ratio","f1","roc_auc","pr_auc","fpr","fnr"]
latex=tex[cols].to_latex(index=False, caption="Additional CICIDS2017 file-wise holdout sensitivity audit across multiple held-out file groups.", label="tab:cicids_multi_holdout", float_format="%.4f")
(TABLES/"table_cicids2017_multi_holdout_audit.tex").write_text(latex, encoding="utf-8")
print("Saved", METRICS/"cicids2017_multi_holdout_audit_results.csv")
