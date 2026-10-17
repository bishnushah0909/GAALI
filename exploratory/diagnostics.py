#!/usr/bin/env python3
"""How much of GAALI is solvable by keyword lookup alone?"""
import os as _os
_D = _os.environ.get("GAALI_DATA_DIR", _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "data"))
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn.svm import LinearSVC

CSV = _os.path.join(_D, "corrected_cleaned_combined_data_kaggle_stratified.csv")
df = pd.read_csv(CSV)
tr = df[df.split == "train"].reset_index(drop=True)
te = df[df.split == "test"].reset_index(drop=True)
ytr, yte = tr.Label.to_numpy(), te.Label.to_numpy()

# ---- 1. single-keyword rule --------------------------------------------
print("=" * 76)
print("1. SINGLE-KEYWORD RULES on the test set")
print("=" * 76)
print(f"{'keyword':<14}{'n_test':>8}{'P(off|kw)':>11}{'macroF1 as sole rule':>22}")
for kw in ["dhoti", "muji", "randi", "chor", "kukur", "chutiya", "sala"]:
    m = te.Comment.astype(str).str.contains(rf"\b{kw}", case=False).to_numpy()
    if m.sum() == 0:
        continue
    rule = m.astype(int)
    print(f"{kw:<14}{m.sum():>8}{yte[m].mean():>11.3f}"
          f"{f1_score(yte, rule, average='macro'):>22.4f}")

# ---- 2. top-k lexicon rule ---------------------------------------------
print()
print("=" * 76)
print("2. LEXICON RULE: top-k train terms by offensive log-odds")
print("=" * 76)
vec = TfidfVectorizer(max_features=5000)
vec.fit(tr.Comment.astype(str))
vocab = np.array(vec.get_feature_names_out())

cv = TfidfVectorizer(max_features=5000, use_idf=False, norm=None,
                     binary=True).fit(tr.Comment.astype(str))
Btr = (cv.transform(tr.Comment.astype(str)) > 0).toarray()
n1 = Btr[ytr == 1].sum(0) + 1
n0 = Btr[ytr == 0].sum(0) + 1
logodds = np.log((n1 / (ytr == 1).sum()) / (n0 / (ytr == 0).sum()))
terms = np.array(cv.get_feature_names_out())

Bte = (cv.transform(te.Comment.astype(str)) > 0).toarray()
for k in (1, 5, 10, 25, 50, 100):
    top = np.argsort(-logodds)[:k]
    rule = (Bte[:, top].sum(1) > 0).astype(int)
    print(f"  k={k:<4} macro F1 = {f1_score(yte, rule, average='macro'):.4f}"
          f"   coverage = {rule.mean():.3f}")
print("\n  top 15 terms:", ", ".join(terms[np.argsort(-logodds)[:15]]))

# ---- 3. ablate the dominant slur ---------------------------------------
print()
print("=" * 76)
print("3. ABLATION: performance once 'dhoti' comments are removed")
print("=" * 76)
has_tr = tr.Comment.astype(str).str.contains("dhoti", case=False).to_numpy()
has_te = te.Comment.astype(str).str.contains("dhoti", case=False).to_numpy()
print(f"train containing 'dhoti': {has_tr.sum()} ({has_tr.mean()*100:.1f}%), "
      f"offensive share {ytr[has_tr].mean():.3f}")
print(f"test  containing 'dhoti': {has_te.sum()} ({has_te.mean()*100:.1f}%), "
      f"offensive share {yte[has_te].mean():.3f}")


def fit_eval(tr_txt, tr_y, te_txt, te_y):
    v = TfidfVectorizer(max_features=5000)
    A = v.fit_transform(tr_txt)
    clf = LinearSVC(C=1, random_state=42).fit(A, tr_y)
    return f1_score(te_y, clf.predict(v.transform(te_txt)), average="macro")


full = fit_eval(tr.Comment.astype(str), ytr, te.Comment.astype(str), yte)
nodh = fit_eval(tr.Comment[~has_tr].astype(str), ytr[~has_tr],
                te.Comment[~has_te].astype(str), yte[~has_te])
masked_tr = tr.Comment.astype(str).str.replace(r"(?i)dhoti\w*", "<MASK>", regex=True)
masked_te = te.Comment.astype(str).str.replace(r"(?i)dhoti\w*", "<MASK>", regex=True)
mask = fit_eval(masked_tr, ytr, masked_te, yte)

print(f"\n  macro F1, full test                      : {full:.4f}")
print(f"  macro F1, 'dhoti' comments removed       : {nodh:.4f}   "
      f"(n_test = {(~has_te).sum()})")
print(f"  macro F1, 'dhoti' masked to <MASK>       : {mask:.4f}")

# ---- 4. learning curve --------------------------------------------------
print()
print("=" * 76)
print("4. LEARNING CURVE (LinearSVC, word unigrams)")
print("=" * 76)
rng = np.random.default_rng(42)
for frac in (0.05, 0.1, 0.25, 0.5, 1.0):
    k = int(len(tr) * frac)
    idx = rng.permutation(len(tr))[:k]
    s = fit_eval(tr.Comment.astype(str).iloc[idx], ytr[idx],
                 te.Comment.astype(str), yte)
    print(f"  {frac*100:>5.0f}% of train (n={k:>5}) -> macro F1 {s:.4f}")
