"""
Semantic grouping robustness and runtime/environment report.

Creates:
04_Results/metrics/semantic_grouping_robustness.csv
04_Results/metrics/system_hardware_report.csv
04_Results/metrics/python_package_versions.csv
06_LaTeX/tables/table_semantic_grouping_robustness.tex
06_LaTeX/tables/table_system_environment.tex
"""
from pathlib import Path
import platform, subprocess, sys, re
import pandas as pd, numpy as np
ROOT=Path(__file__).resolve().parents[2]; METRICS=ROOT/"04_Results"/"metrics"; TABLES=ROOT/"06_LaTeX"/"tables"; METRICS.mkdir(parents=True,exist_ok=True); TABLES.mkdir(parents=True,exist_ok=True)

def group_primary(f):
    s=f.lower()
    if any(k in s for k in ["byte","bytes","sbytes","dbytes","length","pkt","packet"]): return "traffic_volume_packet_length"
    if any(k in s for k in ["port","service","proto","protocol","flag","state"]): return "service_protocol_port_state"
    if any(k in s for k in ["count","srv","host","ct_"]): return "connection_host_counts"
    if any(k in s for k in ["ttl","iat","rate","mean","std","win","header","seg"]): return "timing_header_statistics"
    return "other"

def group_alt(f):
    s=f.lower()
    if any(k in s for k in ["src","dst","sbytes","dbytes","byte"]): return "endpoint_volume"
    if any(k in s for k in ["service","proto","port","flag","state"]): return "communication_context"
    if any(k in s for k in ["count","ct_","host","srv"]): return "aggregation_behavior"
    if any(k in s for k in ["length","packet","iat","ttl","win","header","seg","mean","std"]): return "flow_shape_timing"
    return "other"

rows=[]
for path in [METRICS/"shap_top_features_multidataset.csv", METRICS/"shap_seed_top_features.csv"]:
    if path.exists():
        df=pd.read_csv(path)
        if "feature" in df.columns:
            for _,r in df.iterrows():
                feat=str(r["feature"]); rows.append({"source":path.name,"dataset":r.get("dataset","unknown"),"feature":feat,"primary_group":group_primary(feat),"alternative_group":group_alt(feat)})
res=pd.DataFrame(rows).drop_duplicates()
if len(res):
    res["agreement_exact_group_name"]=(res.primary_group==res.alternative_group)
    summary=res.groupby(["dataset","primary_group"]).size().reset_index(name="feature_count")
else:
    summary=pd.DataFrame(columns=["dataset","primary_group","feature_count"])
res.to_csv(METRICS/"semantic_grouping_robustness.csv", index=False)
summary.to_csv(METRICS/"semantic_grouping_summary_primary.csv", index=False)
(TABLES/"table_semantic_grouping_robustness.tex").write_text(summary.to_latex(index=False, caption="Semantic feature-group distribution under the primary expert-defined grouping scheme.", label="tab:semantic_group_robustness"), encoding="utf-8")
# environment
sys_rows=[{"item":"platform","value":platform.platform()}, {"item":"processor","value":platform.processor()}, {"item":"python","value":sys.version.split()[0]}]
try:
    import psutil
    sys_rows.append({"item":"physical_memory_gb","value":round(psutil.virtual_memory().total/(1024**3),2)})
    sys_rows.append({"item":"cpu_count_logical","value":psutil.cpu_count(logical=True)})
except Exception:
    pass
pd.DataFrame(sys_rows).to_csv(METRICS/"system_hardware_report.csv", index=False)
# package versions
packages=["numpy","pandas","sklearn","scipy","xgboost","lightgbm","shap","matplotlib"]
vers=[]
for p in packages:
    try:
        mod=__import__(p if p!="sklearn" else "sklearn")
        vers.append({"package":p,"version":getattr(mod,"__version__","unknown")})
    except Exception:
        vers.append({"package":p,"version":"not installed"})
pd.DataFrame(vers).to_csv(METRICS/"python_package_versions.csv", index=False)
tex=pd.DataFrame(sys_rows)
(TABLES/"table_system_environment.tex").write_text(tex.to_latex(index=False, caption="Execution environment used for runtime and reproducibility reporting.", label="tab:system_environment"), encoding="utf-8")
print("Saved semantic grouping robustness and system environment reports.")
