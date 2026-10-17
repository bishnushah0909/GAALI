import os as _os
_D = _os.environ.get("GAALI_DATA_DIR", _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "data"))
import re, numpy as np, pandas as pd
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn.svm import LinearSVC
SEED=42
d = pd.read_csv(_os.path.join(_D, "corrected_cleaned_combined_data_kaggle_stratified.csv"))
tr=d[d.split=='train'].reset_index(drop=True); te=d[d.split=='test'].reset_index(drop=True)
ytr,yte=tr.Label.to_numpy(),te.Label.to_numpy()
Xtr,Xte=tr.Comment.astype(str),te.Comment.astype(str)

def run(txt_tr,y_tr,txt_te,y_te):
    v=TfidfVectorizer(max_features=5000); A=v.fit_transform(txt_tr)
    c=LinearSVC(C=1,random_state=SEED).fit(A,y_tr)
    return f1_score(y_te,c.predict(v.transform(txt_te)),average='macro')

print("="*78); print("DENOMINATORS for 'dhoti'"); print("="*78)
pat=r'(?i)\bdhoti'
m_all=d.Comment.astype(str).str.contains(pat,regex=True).to_numpy()
m_tr=Xtr.str.contains(pat,regex=True).to_numpy(); m_te=Xte.str.contains(pat,regex=True).to_numpy()
for nm,msk,lab in [("CORPUS (n=6773)",m_all,d.Label.to_numpy()),
                   ("TRAIN  (n=5418)",m_tr,ytr),("TEST   (n=678) ",m_te,yte)]:
    print(f"  {nm}: {msk.sum():>5} comments ({msk.mean()*100:>5.2f}%)  offensive prevalence {lab[msk].mean()*100:.1f}%")

print("\n"+"="*78); print("ABLATION, all anchored to the SAME reference SVC"); print("="*78)
ref_full = run(Xtr,ytr,Xte,yte)
print(f"  [R0] reference SVC, full test (n={len(yte)})                 : {ref_full:.4f}")

# MASKING: same test population, token withheld
mk=lambda s: s.str.replace(r'(?i)dhoti\w*','<MASK>',regex=True)
mask_full = run(mk(Xtr),ytr,mk(Xte),yte)
print(f"  [R1] masked, full test (n={len(yte)}, SAME population)       : {mask_full:.4f}   D vs R0 = {mask_full-ref_full:+.4f}")

# REMOVAL: smaller test population -> needs its own control
rem = run(Xtr[~m_tr],ytr[~m_tr],Xte[~m_te],yte[~m_te])
print(f"  [R2] removed, reduced test (n={(~m_te).sum()})                : {rem:.4f}")
# control for population change: reference SVC scored on the SAME reduced subset
v=TfidfVectorizer(max_features=5000); A=v.fit_transform(Xtr)
c=LinearSVC(C=1,random_state=SEED).fit(A,ytr); full_pred=c.predict(v.transform(Xte))
ref_sub = f1_score(yte[~m_te],full_pred[~m_te],average='macro')
print(f"  [R3] reference SVC scored on that SAME reduced subset        : {ref_sub:.4f}")
print(f"\n  population effect  (R3 - R0) = {ref_sub-ref_full:+.4f}   <- caused by changing the test set, not the term")
print(f"  retraining effect  (R2 - R3) = {rem-ref_sub:+.4f}   <- effect of removing the term from TRAINING")
print(f"  masking effect     (R1 - R0) = {mask_full-ref_full:+.4f}   <- clean estimate, population held fixed")

# lexicon with/without dhoti, same population
cv=TfidfVectorizer(max_features=5000,use_idf=False,norm=None,binary=True).fit(Xtr)
terms=np.array(cv.get_feature_names_out()); Btr=(cv.transform(Xtr)>0).toarray()
lo=np.log(((Btr[ytr==1].sum(0)+1)/(ytr==1).sum())/((Btr[ytr==0].sum(0)+1)/(ytr==0).sum()))
o=np.argsort(-lo); Bte=(cv.transform(Xte)>0).toarray()
lex=(Bte[:,o[:25]].sum(1)>0).astype(int)
keep=[i for i in o[:25] if terms[i]!='dhoti']
lex_nod=(Bte[:,keep].sum(1)>0).astype(int)
print(f"\n  lexicon-25 full test                 : {f1_score(yte,lex,average='macro'):.4f}")
print(f"  lexicon-25 minus 'dhoti' (24 terms)  : {f1_score(yte,lex_nod,average='macro'):.4f}")
print(f"  'dhoti' alone as sole rule           : {f1_score(yte,m_te.astype(int),average='macro'):.4f}")

# ---- leakage: what the controlled subset does and does NOT establish ----
def sh(t,k=5):
    t=re.sub(r'\s+',' ',t.strip().lower()); return {t[i:i+k] for i in range(max(1,len(t)-k+1))} or {t}
trs=[sh(t) for t in Xtr]; inv=defaultdict(set)
for i,s in enumerate(trs):
    for g in s: inv[g].add(i)
mj=[]
for t in Xte:
    s=sh(t); cnt=defaultdict(int)
    for g in s:
        for i in inv.get(g,()): cnt[i]+=1
    mj.append(max((v/(len(s)+len(trs[i])-v) for i,v in cnt.items()),default=0.))
mj=np.array(mj)
print("\n"+"="*78); print("LEAKAGE AUDIT thresholds"); print("="*78)
for t in (0.7,0.8,0.9):
    print(f"  J* >= {t}: {(mj>=t).sum():>4} of {len(mj)} test comments ({(mj>=t).mean()*100:.2f}%)")
print(f"  controlled subset (J* < 0.8): n = {(mj<0.8).sum()}")
np.save('clean_mask.npy', mj<0.8)
