"""
NSL-KDD class-level error analysis.

This script addresses reviewer requests to explain the NSL-KDD validation-to-test gap.
It attempts to recover original attack labels from raw NSL-KDD/KDDTest files if they are
available in 02_Data/raw/ or from existing processed metadata if present. It then joins the
selected model predictions and reports class/family-level recall and false-negative rates.

Outputs:
04_Results/metrics/nsl_kdd_class_level_error_analysis.csv
06_LaTeX/tables/table_nsl_kdd_class_error_analysis.tex
05_Figures/final/fig_nsl_kdd_class_fnr.pdf

Recommended prerequisite:
Run 08_generate_seed_predictions_and_significance.py first so paired predictions exist in
04_Results/predictions/.
"""
from __future__ import annotations
from pathlib import Path
import argparse, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

warnings.filterwarnings("ignore")
parser=argparse.ArgumentParser()
parser.add_argument("--prediction-file", type=str, default="", help="Optional path to an NPZ prediction file. If omitted, uses LightGBM seed42 for NSL-KDD if present.")
parser.add_argument("--raw-test-file", type=str, default="", help="Optional raw NSL-KDD test file with original attack labels.")
args=parser.parse_args()
ROOT=Path(__file__).resolve().parents[2]
METRICS=ROOT/"04_Results"/"metrics"; PRED=ROOT/"04_Results"/"predictions"; TABLES=ROOT/"06_LaTeX"/"tables"; FIG=ROOT/"05_Figures"/"final"
for p in [METRICS,TABLES,FIG]: p.mkdir(parents=True, exist_ok=True)

ATTACK_FAMILY={
    "normal":"normal","neptune":"dos","smurf":"dos","back":"dos","teardrop":"dos","pod":"dos","land":"dos","apache2":"dos","udpstorm":"dos","processtable":"dos","worm":"dos",
    "satan":"probe","ipsweep":"probe","portsweep":"probe","nmap":"probe","mscan":"probe","saint":"probe",
    "guess_passwd":"r2l","ftp_write":"r2l","imap":"r2l","phf":"r2l","multihop":"r2l","warezmaster":"r2l","warezclient":"r2l","spy":"r2l","xlock":"r2l","xsnoop":"r2l","snmpguess":"r2l","snmpgetattack":"r2l","httptunnel":"r2l","sendmail":"r2l","named":"r2l",
    "buffer_overflow":"u2r","loadmodule":"u2r","rootkit":"u2r","perl":"u2r","sqlattack":"u2r","xterm":"u2r","ps":"u2r"
}

def find_raw_test():
    if args.raw_test_file:
        p=Path(args.raw_test_file); return p if p.exists() else None
    candidates=list((ROOT/"02_Data"/"raw").rglob("*Test*"))+list((ROOT/"02_Data"/"raw").rglob("*test*"))+list((ROOT/"02_Data").rglob("*KDDTest*"))
    candidates=[p for p in candidates if p.is_file()]
    return candidates[0] if candidates else None

def load_original_labels(n_expected:int):
    p=find_raw_test()
    if p and p.exists():
        try:
            # NSL-KDD raw is usually comma-separated without header. Label is second-last if difficulty column is present, otherwise last.
            df=pd.read_csv(p, header=None)
            if len(df)==n_expected:
                label_col = df.columns[-2] if df.shape[1] >= 43 else df.columns[-1]
                labels=df[label_col].astype(str).str.strip().str.replace(".","",regex=False)
                return labels, f"raw:{p}"
        except Exception as exc:
            print(f"Could not parse raw NSL-KDD labels from {p}: {exc}")
    # Fallback: binary only.
    ypath=ROOT/"02_Data"/"processed"/"y_test_binary.csv"
    if ypath.exists():
        y=pd.read_csv(ypath).iloc[:,0].astype(int)
        labels=y.map({0:"normal",1:"attack_unknown"})
        return labels, "binary_fallback"
    raise FileNotFoundError("Could not locate raw NSL-KDD test labels or processed y_test_binary.csv")

def find_pred_file():
    if args.prediction_file:
        p=Path(args.prediction_file); return p if p.exists() else None
    candidates=list(PRED.glob("pred_nsl_kdd_lightgbm_seed42.npz"))+list(PRED.glob("*nsl*kdd*lightgbm*.npz"))
    return candidates[0] if candidates else None

pf=find_pred_file()
if pf is None:
    raise FileNotFoundError("No NSL-KDD prediction NPZ found. Run script 08 first or pass --prediction-file.")
data=np.load(pf)
y_true=data["y_true"].astype(int); y_pred=data["y_pred"].astype(int)
labels,source=load_original_labels(len(y_true))
if len(labels)!=len(y_true):
    raise ValueError(f"Original label count {len(labels)} does not match y_true length {len(y_true)}")
family=labels.str.lower().map(ATTACK_FAMILY).fillna("other_attack")
df=pd.DataFrame({"original_label":labels,"attack_family":family,"y_true":y_true,"y_pred":y_pred})
records=[]
for group_col in ["attack_family","original_label"]:
    for name,g in df.groupby(group_col):
        if len(g)<5: continue
        positives=int((g.y_true==1).sum()); negatives=int((g.y_true==0).sum())
        tp=int(((g.y_true==1)&(g.y_pred==1)).sum()); fn=int(((g.y_true==1)&(g.y_pred==0)).sum()); tn=int(((g.y_true==0)&(g.y_pred==0)).sum()); fp=int(((g.y_true==0)&(g.y_pred==1)).sum())
        rec={"group_type":group_col,"group":name,"rows":len(g),"positives":positives,"negatives":negatives,"tp":tp,"fn":fn,"tn":tn,"fp":fp,
             "recall_attack":tp/(tp+fn) if (tp+fn)>0 else np.nan,"fnr_attack":fn/(tp+fn) if (tp+fn)>0 else np.nan,"fpr_benign":fp/(fp+tn) if (fp+tn)>0 else np.nan}
        records.append(rec)
res=pd.DataFrame(records).sort_values(["group_type","fnr_attack"], ascending=[True,False])
res.to_csv(METRICS/"nsl_kdd_class_level_error_analysis.csv", index=False)
tex=res[res.group_type=="attack_family"].copy()
for c in ["recall_attack","fnr_attack","fpr_benign"]: tex[c]=tex[c].astype(float).round(4)
(TABLES/"table_nsl_kdd_class_error_analysis.tex").write_text(tex.to_latex(index=False, caption="NSL-KDD test-set error analysis by attack family.", label="tab:nsl_class_error", float_format="%.4f"), encoding="utf-8")
plot=tex[tex["positives"]>0].sort_values("fnr_attack")
plt.figure(figsize=(7,4)); plt.barh(plot["group"], plot["fnr_attack"]); plt.xlabel("False negative rate"); plt.title("NSL-KDD attack-family false negative rate"); plt.tight_layout(); plt.savefig(FIG/"fig_nsl_kdd_class_fnr.pdf", bbox_inches="tight"); plt.savefig(FIG/"fig_nsl_kdd_class_fnr.png", dpi=300, bbox_inches="tight"); plt.close()
print(f"Saved NSL-KDD class-level error analysis using labels from {source} and predictions {pf}")
print(res.head(20).to_string(index=False))
