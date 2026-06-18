"""
Add a lightweight deep-learning-style MLP baseline using scikit-learn MLPClassifier.
This addresses reviewer concerns about excluding neural baselines without evidence.

Outputs:
04_Results/metrics/mlp_baseline_results.csv
06_LaTeX/tables/table_mlp_baseline_results.tex
"""
from pathlib import Path
import argparse, warnings, time
import numpy as np, pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, confusion_matrix
warnings.filterwarnings("ignore")
parser=argparse.ArgumentParser(); parser.add_argument("--max-train-rows", type=int, default=0); parser.add_argument("--random-state", type=int, default=42); args=parser.parse_args()
ROOT=Path(__file__).resolve().parents[2]; PROCESSED=ROOT/"02_Data"/"processed"; METRICS=ROOT/"04_Results"/"metrics"; TABLES=ROOT/"06_LaTeX"/"tables"; METRICS.mkdir(parents=True, exist_ok=True); TABLES.mkdir(parents=True, exist_ok=True)
DATASETS={"NSL-KDD":PROCESSED,"UNSW-NB15":PROCESSED/"UNSW-NB15","CICIDS2017":PROCESSED/"CICIDS2017"}
def read_y(path): return pd.read_csv(path).iloc[:,0].astype(int)
def eval_model(model,X,y):
    pred=model.predict(X); score=model.predict_proba(X)[:,1]; tn,fp,fn,tp=confusion_matrix(y,pred).ravel()
    return {"accuracy":accuracy_score(y,pred),"precision":precision_score(y,pred,zero_division=0),"recall":recall_score(y,pred,zero_division=0),"f1":f1_score(y,pred,zero_division=0),"roc_auc":roc_auc_score(y,score),"pr_auc":average_precision_score(y,score),"tn":tn,"fp":fp,"fn":fn,"tp":tp,"fpr":fp/(fp+tn) if fp+tn else 0,"fnr":fn/(fn+tp) if fn+tp else 0}
records=[]
for ds,path in DATASETS.items():
    missing=[f for f in ["X_train_final.csv","X_val_final.csv","X_test_final.csv","y_train_binary.csv","y_val_binary.csv","y_test_binary.csv"] if not (path/f).exists()]
    if missing: print(f"Skipping {ds}: {missing}"); continue
    Xtr=pd.read_csv(path/"X_train_final.csv").astype(np.float32); Xval=pd.read_csv(path/"X_val_final.csv").astype(np.float32); Xte=pd.read_csv(path/"X_test_final.csv").astype(np.float32)
    ytr=read_y(path/"y_train_binary.csv"); yval=read_y(path/"y_val_binary.csv"); yte=read_y(path/"y_test_binary.csv")
    if args.max_train_rows and len(Xtr)>args.max_train_rows:
        idx=Xtr.sample(n=args.max_train_rows, random_state=args.random_state).index; Xtr=Xtr.loc[idx].reset_index(drop=True); ytr=ytr.loc[idx].reset_index(drop=True)
    clf=MLPClassifier(hidden_layer_sizes=(128,64), activation="relu", alpha=1e-4, batch_size=256, learning_rate_init=1e-3, max_iter=80, early_stopping=True, validation_fraction=0.1, random_state=args.random_state)
    print(f"Training MLP on {ds}: {Xtr.shape}")
    t=time.time(); clf.fit(Xtr,ytr); train_time=time.time()-t
    for split,X,y in [("validation",Xval,yval),("test",Xte,yte)]:
        rec={"dataset":ds,"model":"MLP","split":split,"train_rows":len(Xtr),"features":Xtr.shape[1],"train_time_sec":train_time}; rec.update(eval_model(clf,X,y)); records.append(rec)
res=pd.DataFrame(records); res.to_csv(METRICS/"mlp_baseline_results.csv", index=False)
tex=res.copy();
for c in tex.select_dtypes(include=[float]).columns: tex[c]=tex[c].round(4)
(TABLES/"table_mlp_baseline_results.tex").write_text(tex.to_latex(index=False, caption="Lightweight MLP baseline results for reviewer-requested neural comparison.", label="tab:mlp_baseline", float_format="%.4f"), encoding="utf-8")
print(res.to_string(index=False))
