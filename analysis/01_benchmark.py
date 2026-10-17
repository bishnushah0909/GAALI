import os as _os
_D = _os.environ.get("GAALI_DATA_DIR", _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "data"))
import itertools, re
import numpy as np, pandas as pd
from collections import defaultdict
from scipy.stats import binomtest, chi2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
from sklearn.svm import LinearSVC

SEED, B = 42, 1000
d = pd.read_csv(_os.path.join(_D, "corrected_cleaned_combined_data_kaggle_stratified.csv"))
p = pd.read_csv(_os.path.join(_D, "test_set_model_predictions.csv"))
tr = d[d.split=='train'].reset_index(drop=True)
dv = d[d.split=='dev'].reset_index(drop=True)
te = d[d.split=='test'].reset_index(drop=True)
ytr, ydv, yte = tr.Label.to_numpy(), dv.Label.to_numpy(), te.Label.to_numpy()
Xtr, Xdv, Xte = tr.Comment.astype(str), dv.Comment.astype(str), te.Comment.astype(str)
assert (p.Comment.astype(str).values == Xte.values).all() and (p.true_label.values==yte).all()

# ---------- reproducible reference SVC (Section 5.2.1 config) ----------
def fit_svc(txt_tr, y_tr, txt_ev):
    v = TfidfVectorizer(max_features=5000)
    A = v.fit_transform(txt_tr)
    return LinearSVC(C=1, random_state=SEED).fit(A, y_tr).predict(v.transform(txt_ev))

svc = fit_svc(Xtr, ytr, Xte)

# ---------- lexicon: select k on DEV, evaluate on TEST ----------
cv = TfidfVectorizer(max_features=5000, use_idf=False, norm=None, binary=True).fit(Xtr)
terms = np.array(cv.get_feature_names_out())
Btr = (cv.transform(Xtr) > 0).toarray()
lo = np.log(((Btr[ytr==1].sum(0)+1)/(ytr==1).sum()) / ((Btr[ytr==0].sum(0)+1)/(ytr==0).sum()))
order = np.argsort(-lo)
Bdv = (cv.transform(Xdv) > 0).toarray(); Bte = (cv.transform(Xte) > 0).toarray()

print("="*80); print("A. LEXICON SIZE SELECTED ON DEV (test untouched)"); print("="*80)
print(f"{'k':>5}{'DEV macroF1':>14}{'TEST macroF1':>14}{'test coverage':>15}")
sweep = {}
for k in (1,5,10,25,50,100):
    rd = (Bdv[:,order[:k]].sum(1)>0).astype(int)
    rt = (Bte[:,order[:k]].sum(1)>0).astype(int)
    sweep[k] = (f1_score(ydv,rd,average='macro'), f1_score(yte,rt,average='macro'), rt.mean())
    print(f"{k:>5}{sweep[k][0]:>14.4f}{sweep[k][1]:>14.4f}{sweep[k][2]:>15.3f}")
k_star = max(sweep, key=lambda k: sweep[k][0])
print(f"\nDEV-selected k* = {k_star}  (dev F1 {sweep[k_star][0]:.4f}) -> test F1 {sweep[k_star][1]:.4f}")
lex = (Bte[:,order[:k_star]].sum(1)>0).astype(int)
print("\nFULL LEXICON (k*={}):".format(k_star))
print("  " + ", ".join(terms[order[:k_star]]))

# ---------- systems ----------
preds = {"mBERT":p.mbert_pred.to_numpy(), f"Keyword lexicon ({k_star})":lex,
         "MuRIL":p.muril_pred.to_numpy(), "XLM-RoBERTa":p.xlm_roberta_pred.to_numpy(),
         "Linear SVC":svc, "IndicBERT":p.indicbert_pred.to_numpy(),
         "Multinomial NB":p.mnb_pred.to_numpy()}

print("\n"+"="*80); print("B. TABLE 3 (reference SVC substituted)"); print("="*80)
rows=[]
for s,q in preds.items():
    rows.append(dict(system=s,
        mP=precision_score(yte,q,average='macro'), mR=recall_score(yte,q,average='macro'),
        mF1=f1_score(yte,q,average='macro'), wF1=f1_score(yte,q,average='weighted'),
        acc=(q==yte).mean(), cm=confusion_matrix(yte,q).tolist()))
mt = pd.DataFrame(rows).sort_values('mF1',ascending=False).reset_index(drop=True)
print(mt.to_string(index=False, float_format=lambda x:f"{x:.4f}"))

# ---------- bootstrap ----------
rng = np.random.default_rng(SEED); bi = rng.integers(0,len(yte),size=(B,len(yte)))
boot = {s: np.array([f1_score(yte[i],preds[s][i],average='macro',zero_division=0) for i in bi]) for s in preds}
print("\n95% CI:")
ci={}
for s in mt.system:
    l,h = np.percentile(boot[s],[2.5,97.5]); ci[s]=(l,h)
    print(f"  {s:<22}{f1_score(yte,preds[s],average='macro'):.4f}  [{l:.4f}, {h:.4f}]")

def mc(a,b):
    ao,bo = preds[a]==yte, preds[b]==yte
    n10,n01 = int((ao&~bo).sum()), int((~ao&bo).sum()); nd=n10+n01
    if nd==0: return n10,n01,1.0,"none"
    if nd<25: return n10,n01,float(binomtest(min(n01,n10),nd,0.5).pvalue),"exact"
    return n10,n01,float(chi2.sf((abs(n01-n10)-1)**2/nd,1)),"chi2"
def holm(pv):
    m,o,adj,run=len(pv),np.argsort(pv),np.empty(len(pv)),0.
    for r,i in enumerate(o): run=max(run,(m-r)*pv[i]); adj[i]=min(run,1.)
    return adj
rws,pvs=[],[]
for a,b in itertools.combinations(list(mt.system),2):
    n10,n01,pv,kind = mc(a,b); dd = boot[a]-boot[b]
    rws.append(dict(a=a,b=b,delta=f1_score(yte,preds[a],average='macro')-f1_score(yte,preds[b],average='macro'),
                    lo=np.percentile(dd,2.5),hi=np.percentile(dd,97.5),n10=n10,n01=n01,kind=kind)); pvs.append(pv)
pw=pd.DataFrame(rws); pw['p_holm']=holm(np.array(pvs)); pw['sig']=pw.p_holm<0.05
print("\n"+"="*80); print("C. TABLE 4 (all 21 pairs)"); print("="*80)
for _,r in pw.iterrows():
    print(f" {'*' if r.sig else ' '} {r.a:<22}vs {r.b:<22}D={r.delta:+.4f} [{r.lo:+.4f},{r.hi:+.4f}] n10={r.n10:<3} n01={r.n01:<3} p={r.p_holm:.4f} ({r.kind})")
pw.to_csv('pairwise_final.csv',index=False); mt.to_csv('metrics_final.csv',index=False)
