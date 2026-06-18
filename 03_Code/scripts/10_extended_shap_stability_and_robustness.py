"""
Extended SHAP stability and SHAP subsample robustness analysis.

Addresses reviewer requests for:
- more than 5 random seeds for explanation stability;
- confidence intervals for top-k Jaccard stability;
- robustness of SHAP top features across multiple fixed-size test subsamples;
- one local SHAP explanation figure per dataset.

Outputs include:
04_Results/metrics/shap_seed_stability_summary_extended.csv
04_Results/metrics/shap_subsample_robustness_summary.csv
04_Results/metrics/local_shap_case_index.csv
06_LaTeX/tables/table_shap_seed_stability_extended.tex
06_LaTeX/tables/table_shap_subsample_robustness.tex
05_Figures/final/fig_local_shap_<dataset>.pdf
"""
from __future__ import annotations
from pathlib import Path
import argparse, warnings, gc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

warnings.filterwarnings("ignore")
try:
    from lightgbm import LGBMClassifier
    from xgboost import XGBClassifier
except Exception as exc:
    raise ImportError("Install lightgbm and xgboost before running this script.") from exc

parser = argparse.ArgumentParser()
parser.add_argument("--seeds", type=str, default=",".join(str(i) for i in range(1, 21)))
parser.add_argument("--max-train-rows", type=int, default=100000)
parser.add_argument("--shap-rows", type=int, default=3000)
parser.add_argument("--subsample-draws", type=int, default=5)
parser.add_argument("--top-k", type=int, default=20)
args = parser.parse_args()

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "02_Data" / "processed"
METRICS = ROOT / "04_Results" / "metrics"
FIG = ROOT / "05_Figures" / "final"
TABLES = ROOT / "06_LaTeX" / "tables"
for p in [METRICS, FIG, TABLES]: p.mkdir(parents=True, exist_ok=True)
SEEDS = [int(s) for s in args.seeds.split(',') if s.strip()]

DATASETS = {
    "NSL-KDD": {"path": PROCESSED, "model":"LightGBM"},
    "UNSW-NB15": {"path": PROCESSED / "UNSW-NB15", "model":"XGBoost"},
    "CICIDS2017": {"path": PROCESSED / "CICIDS2017", "model":"LightGBM"},
}

def read_y(path):
    df = pd.read_csv(path)
    return df.iloc[:,0].astype(int)

def load(path):
    Xtr = pd.read_csv(path/"X_train_final.csv").astype(np.float32)
    Xte = pd.read_csv(path/"X_test_final.csv").astype(np.float32)
    ytr = read_y(path/"y_train_binary.csv")
    yte = read_y(path/"y_test_binary.csv")
    return Xtr, Xte, ytr, yte

def make_model(name, seed):
    if name == "LightGBM":
        return LGBMClassifier(n_estimators=500, learning_rate=0.05, num_leaves=64, subsample=0.9, colsample_bytree=0.9, class_weight="balanced", random_state=seed, n_jobs=-1, verbose=-1)
    if name == "XGBoost":
        return XGBClassifier(n_estimators=500, learning_rate=0.05, max_depth=8, subsample=0.9, colsample_bytree=0.9, objective="binary:logistic", eval_metric="logloss", tree_method="hist", random_state=seed, n_jobs=-1)
    raise ValueError(name)

def get_shap(explainer, X):
    sv = explainer.shap_values(X)
    if isinstance(sv, list): sv = sv[1]
    if hasattr(sv, "values"): sv = sv.values
    if isinstance(sv, np.ndarray) and sv.ndim == 3: sv = sv[:,:,1]
    return sv

def jaccard(a,b):
    a,b = set(a), set(b)
    return len(a & b)/len(a | b) if a | b else 0

def ci_boot(vals, n=2000):
    vals = np.asarray(vals, dtype=float)
    if len(vals) == 0: return (np.nan,np.nan)
    rng = np.random.default_rng(42)
    means = [np.mean(rng.choice(vals, size=len(vals), replace=True)) for _ in range(n)]
    return tuple(np.quantile(means, [0.025,0.975]))

seed_records=[]; pair_records=[]; subsample_records=[]; local_records=[]
for ds,cfg in DATASETS.items():
    print(f"\n{ds}: extended SHAP stability")
    Xtr,Xte,ytr,yte = load(cfg["path"])
    fixed_idx = Xte.sample(n=min(args.shap_rows,len(Xte)), random_state=42).index
    Xfixed = Xte.loc[fixed_idx].reset_index(drop=True)
    top_sets={}
    last_model=None; last_explainer=None
    for seed in SEEDS:
        idx = Xtr.sample(n=min(args.max_train_rows,len(Xtr)), random_state=seed).index
        Xfit = Xtr.loc[idx].reset_index(drop=True); yfit = ytr.loc[idx].reset_index(drop=True)
        model = make_model(cfg["model"], seed); model.fit(Xfit,yfit)
        explainer = shap.TreeExplainer(model)
        sv = get_shap(explainer, Xfixed)
        mean_abs = np.abs(sv).mean(axis=0)
        imp = pd.DataFrame({"feature":Xfixed.columns,"mean_abs_shap":mean_abs}).sort_values("mean_abs_shap", ascending=False)
        top = imp.head(args.top_k)["feature"].tolist()
        top_sets[seed] = set(top)
        for rank,(feat,val) in enumerate(zip(imp.head(args.top_k)["feature"], imp.head(args.top_k)["mean_abs_shap"]), start=1):
            seed_records.append({"dataset":ds,"model":cfg["model"],"seed":seed,"rank":rank,"feature":feat,"mean_abs_shap":float(val)})
        last_model, last_explainer = model, explainer
        del Xfit,yfit,sv,imp
        gc.collect()
    vals=[]
    for i,s1 in enumerate(SEEDS):
        for s2 in SEEDS[i+1:]:
            val = jaccard(top_sets[s1], top_sets[s2]); vals.append(val)
            pair_records.append({"dataset":ds,"model":cfg["model"],"seed_a":s1,"seed_b":s2,"top_k":args.top_k,"jaccard":val,"common_features":" | ".join(sorted(top_sets[s1]&top_sets[s2]))})
    lo,hi = ci_boot(vals)
    stable_all = set.intersection(*top_sets.values()) if top_sets else set()
    seed_summary = {"dataset":ds,"model":cfg["model"],"top_k":args.top_k,"num_seeds":len(SEEDS),"mean_jaccard":float(np.mean(vals)),"ci95_low":float(lo),"ci95_high":float(hi),"min_jaccard":float(np.min(vals)),"max_jaccard":float(np.max(vals)),"features_stable_all_seeds":len(stable_all),"stable_features":" | ".join(sorted(stable_all))}
    subsample_sets=[]
    for draw in range(args.subsample_draws):
        idx = Xte.sample(n=min(args.shap_rows,len(Xte)), random_state=1000+draw).index
        Xs = Xte.loc[idx].reset_index(drop=True)
        sv = get_shap(last_explainer, Xs)
        imp = pd.DataFrame({"feature":Xs.columns,"mean_abs_shap":np.abs(sv).mean(axis=0)}).sort_values("mean_abs_shap", ascending=False)
        subsample_sets.append(set(imp.head(args.top_k)["feature"]))
    sub_vals=[]
    for i in range(len(subsample_sets)):
        for j in range(i+1,len(subsample_sets)):
            sub_vals.append(jaccard(subsample_sets[i], subsample_sets[j]))
    slo,shi = ci_boot(sub_vals) if sub_vals else (np.nan,np.nan)
    subsample_records.append({"dataset":ds,"model":cfg["model"],"top_k":args.top_k,"draws":args.subsample_draws,"shap_rows_per_draw":args.shap_rows,"mean_jaccard":float(np.mean(sub_vals)),"ci95_low":float(slo),"ci95_high":float(shi),"min_jaccard":float(np.min(sub_vals)),"max_jaccard":float(np.max(sub_vals))})
    seed_summary.update({"subsample_mean_jaccard":subsample_records[-1]["mean_jaccard"]})
    # local explanation for first correctly detected attack sample
    scores = last_model.predict_proba(Xte)[:,1]
    preds = (scores >= 0.5).astype(int)
    candidates = np.where((yte.to_numpy()==1) & (preds==1))[0]
    if len(candidates)==0: candidates = np.where(preds==1)[0]
    if len(candidates)>0:
        loc = int(candidates[0]); xone = Xte.iloc[[loc]].reset_index(drop=True)
        svone = get_shap(last_explainer, xone)[0]
        top_local = pd.DataFrame({"feature":xone.columns,"shap_value":svone,"feature_value":xone.iloc[0].to_numpy()}).assign(abs_shap=lambda d: d.shap_value.abs()).sort_values("abs_shap", ascending=True).tail(12)
        plt.figure(figsize=(8,5.5)); plt.barh(top_local["feature"], top_local["shap_value"]); plt.axvline(0, color="black", linewidth=0.8); plt.xlabel("SHAP value"); plt.title(f"Local SHAP explanation: {ds}"); plt.tight_layout()
        safe = ds.lower().replace("-","_")
        plt.savefig(FIG/f"fig_local_shap_{safe}.pdf", bbox_inches="tight"); plt.savefig(FIG/f"fig_local_shap_{safe}.png", dpi=300, bbox_inches="tight"); plt.close()
        local_records.append({"dataset":ds,"model":cfg["model"],"test_index":loc,"true_label":int(yte.iloc[loc]),"predicted_label":int(preds[loc]),"predicted_attack_probability":float(scores[loc]),"figure":f"fig_local_shap_{safe}.pdf"})
    seed_records.append({"dataset":ds,"model":cfg["model"],"seed":"SUMMARY","rank":0,"feature":"__SUMMARY__","mean_abs_shap":seed_summary["mean_jaccard"]})

pd.DataFrame(seed_records).to_csv(METRICS/"shap_seed_top_features_extended.csv", index=False)
pd.DataFrame(pair_records).to_csv(METRICS/"shap_seed_stability_pairwise_extended.csv", index=False)
summary_rows=[]
for ds in DATASETS:
    vals=[r["jaccard"] for r in pair_records if r["dataset"]==ds]
    lo,hi=ci_boot(vals)
    rows=[r for r in seed_records if r["dataset"]==ds and r["seed"]!="SUMMARY"]
    # stable count recalc
    seed_to_set={}
    for row in rows:
        seed_to_set.setdefault(row["seed"],set()).add(row["feature"])
    stable_all=set.intersection(*seed_to_set.values()) if seed_to_set else set()
    summary_rows.append({"dataset":ds,"model":DATASETS[ds]["model"],"top_k":args.top_k,"num_seeds":len(SEEDS),"mean_pairwise_jaccard":np.mean(vals),"ci95_low":lo,"ci95_high":hi,"min_pairwise_jaccard":np.min(vals),"max_pairwise_jaccard":np.max(vals),"features_stable_in_all_seeds":len(stable_all),"stable_features":" | ".join(sorted(stable_all))})
summary_df=pd.DataFrame(summary_rows)
sub_df=pd.DataFrame(subsample_records)
loc_df=pd.DataFrame(local_records)
summary_df.to_csv(METRICS/"shap_seed_stability_summary_extended.csv", index=False)
sub_df.to_csv(METRICS/"shap_subsample_robustness_summary.csv", index=False)
loc_df.to_csv(METRICS/"local_shap_case_index.csv", index=False)
for out_df,name,caption,label in [(summary_df,"table_shap_seed_stability_extended.tex","Extended random-seed stability of top-k SHAP explanations with bootstrap confidence intervals.","tab:shap_seed_stability_extended"),(sub_df,"table_shap_subsample_robustness.tex","Robustness of top-k SHAP explanations across repeated fixed-size test subsamples.","tab:shap_subsample_robustness")]:
    tex=out_df.copy()
    for c in tex.select_dtypes(include=[float]).columns: tex[c]=tex[c].round(4)
    (TABLES/name).write_text(tex.to_latex(index=False, caption=caption, label=label, float_format="%.4f"), encoding="utf-8")
print("Saved extended SHAP stability, subsample robustness, and local explanation artifacts.")
