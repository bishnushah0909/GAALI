import os as _os
_D = _os.environ.get("GAALI_DATA_DIR", _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "data"))
import numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn.svm import LinearSVC
SEED=42
d=pd.read_csv(_os.path.join(_D, "corrected_cleaned_combined_data_kaggle_stratified.csv"))
p=pd.read_csv(_os.path.join(_D, "test_set_model_predictions.csv"))
tr=d[d.split=='train'].reset_index(drop=True); te=d[d.split=='test'].reset_index(drop=True)
ytr,yte=tr.Label.to_numpy(),te.Label.to_numpy(); Xtr,Xte=tr.Comment.astype(str),te.Comment.astype(str)
v=TfidfVectorizer(max_features=5000); A=v.fit_transform(Xtr)
svc=LinearSVC(C=1,random_state=SEED).fit(A,ytr).predict(v.transform(Xte))
cv=TfidfVectorizer(max_features=5000,use_idf=False,norm=None,binary=True).fit(Xtr)
Btr=(cv.transform(Xtr)>0).toarray()
lo=np.log(((Btr[ytr==1].sum(0)+1)/(ytr==1).sum())/((Btr[ytr==0].sum(0)+1)/(ytr==0).sum()))
o=np.argsort(-lo); lex=((cv.transform(Xte)>0).toarray()[:,o[:25]].sum(1)>0).astype(int)
P={"mBERT":p.mbert_pred.to_numpy(),"Keyword lexicon":lex,"Linear SVC":svc,"MuRIL":p.muril_pred.to_numpy(),
   "XLM-RoBERTa":p.xlm_roberta_pred.to_numpy(),"IndicBERT":p.indicbert_pred.to_numpy(),"Multinomial NB":p.mnb_pred.to_numpy()}
cl=np.load('clean_mask.npy')
print("="*76);print(f"LEAKAGE-CONTROLLED RE-EVALUATION (n={cl.sum()})");print("="*76)
for s,q in P.items():
    f=f1_score(yte,q,average='macro'); c=f1_score(yte[cl],q[cl],average='macro')
    print(f"  {s:<18} full {f:.4f}   controlled {c:.4f}   change {c-f:+.4f}")
print("\n"+"="*76);print("TARGET-GROUP RECALL ON OFFENSIVE COMMENTS");print("="*76)
G={"dhoti":r'(?i)\bdhoti',"madhesi":r'(?i)\bmadhesi|\bmadesi|\bmadhesh',"muslim":r'(?i)\bmuslim|\bmusalman',
   "newar":r'(?i)\bnewar',"tharu":r'(?i)\btharu',"dalit":r'(?i)\bdalit'}
hdr=f"{'group':<10}{'n':>5}{'n_off':>7}{'%off':>7}"+"".join(f"{s[:11]:>13}" for s in P)
print(hdr)
for g,pat in G.items():
    m=Xte.str.contains(pat,regex=True).to_numpy(); pos=m&(yte==1)
    if pos.sum()==0: continue
    line=f"{g:<10}{m.sum():>5}{pos.sum():>7}{yte[m].mean()*100:>6.1f}%"
    for s in P: line+=f"{(P[s][pos]==1).mean():>13.3f}"
    print(line)
e1,e2=(P['mBERT']!=yte),(P['Keyword lexicon']!=yte)
print(f"\nmBERT errors {e1.sum()} | lexicon errors {e2.sum()} | shared {(e1&e2).sum()} | Jaccard {(e1&e2).sum()/(e1|e2).sum():.3f}")
allw=np.ones(len(yte),bool)
for s in P:
    if s!='Keyword lexicon': allw &= (P[s]!=yte)
print(f"misclassified by all six trained systems: {allw.sum()} -> oracle ceiling {1-allw.mean():.4f}")
maj=(np.stack([P[s] for s in P if s!='Keyword lexicon']).sum(0)>=3).astype(int)
print(f"majority vote macro F1: {f1_score(yte,maj,average='macro'):.4f}")
