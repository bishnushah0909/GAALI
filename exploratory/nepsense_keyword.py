#!/usr/bin/env python3
"""Keyword-baseline saturation test on NepSense (Devanagari Nepali).

Mirrors the GAALI protocol: stratified 80/10/10, lexicon fitted on train
only, TF-IDF baselines, bootstrap CIs, Holm-corrected McNemar.
"""
import os as _os
_D = _os.environ.get("GAALI_DATA_DIR", _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "data"))
import itertools
import re
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.stats import binomtest, chi2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

SEED, B = 42, 1000
# Devanagari vowel signs are combining marks; sklearn's default \w\w+ pattern
# treats them as token boundaries and shatters every word. Always override.
DEVA = r"[\u0900-\u097F]+|[A-Za-z]{2,}|[0-9]+"
XL = _os.path.join(_D, "NepSense_Data_Version_1.xlsx")

df = pd.read_excel(XL)
df["text"] = df.Nepali_Text.astype(str).str.strip()
y_all = df.Offensive.to_numpy()

print("=" * 84)
print("0. CORPUS")
print("=" * 84)
print(f"rows: {len(df)}   offensive: {y_all.sum()} ({y_all.mean()*100:.1f}%)")
print("Label x Profanity cross-tab:")
print(pd.crosstab(df.Label, df.Profanity))
print("\nLabel semantics inferred: 0 = non-offensive, 1 = offensive w/o profanity, "
      "2 = profane")
tok = df.text.str.split().map(len)
print(f"tokens total {tok.sum():,}   mean/comment {tok.mean():.1f}   "
      f"median {tok.median():.0f}   max {tok.max()}")
vocab = {t for r in df.text.str.split() for t in r}
print(f"vocabulary (whitespace types): {len(vocab):,}   "
      f"TTR {len(vocab)/tok.sum():.4f}")

# ---------------- stratified 80/10/10 ----------------
idx = np.arange(len(df))
tr_i, tmp_i = train_test_split(idx, test_size=0.20, stratify=y_all,
                               random_state=SEED)
dv_i, te_i = train_test_split(tmp_i, test_size=0.50, stratify=y_all[tmp_i],
                              random_state=SEED)
Xtr, Xte = df.text.iloc[tr_i], df.text.iloc[te_i]
ytr, yte = y_all[tr_i], y_all[te_i]
lab_te = df.Label.to_numpy()[te_i]
print(f"\nsplit: train {len(tr_i)}  dev {len(dv_i)}  test {len(te_i)}   "
      f"offensive share {ytr.mean():.3f}/{y_all[dv_i].mean():.3f}/{yte.mean():.3f}")

# ---------------- lexicon from train only ----------------
cv = TfidfVectorizer(max_features=5000, use_idf=False, norm=None,
                     binary=True, token_pattern=DEVA).fit(Xtr)
Btr = (cv.transform(Xtr) > 0).toarray()
Bte = (cv.transform(Xte) > 0).toarray()
terms = np.array(cv.get_feature_names_out())
lo = np.log(((Btr[ytr == 1].sum(0) + 1) / (ytr == 1).sum()) /
            ((Btr[ytr == 0].sum(0) + 1) / (ytr == 0).sum()))
order = np.argsort(-lo)

print("\n" + "=" * 84)
print("1. KEYWORD-LEXICON SWEEP (fitted on train only)")
print("=" * 84)
print(f"{'k':>5}{'macro F1':>12}{'coverage':>11}{'precision(off)':>16}")
best_k, best_f1 = None, -1
for k in (1, 3, 5, 10, 25, 50, 100, 200, 400, 800):
    rule = (Bte[:, order[:k]].sum(1) > 0).astype(int)
    f1 = f1_score(yte, rule, average="macro")
    prec = (yte[rule == 1] == 1).mean() if rule.sum() else float("nan")
    print(f"{k:>5}{f1:>12.4f}{rule.mean():>11.3f}{prec:>16.3f}")
    if f1 > best_f1:
        best_f1, best_k = f1, k
print(f"\nbest k = {best_k} (macro F1 {best_f1:.4f})")
print("top 20 terms:", " ".join(terms[order[:20]]))

lex = (Bte[:, order[:best_k]].sum(1) > 0).astype(int)

# ---------------- trained baselines ----------------
print("\n" + "=" * 84)
print("2. TRAINED BASELINES vs LEXICON")
print("=" * 84)
preds = {f"Lexicon-{best_k}": lex}
for name, vec, clf in [
    ("SVC word", TfidfVectorizer(max_features=5000, token_pattern=DEVA), LinearSVC(C=1, random_state=SEED)),
    ("MNB word", TfidfVectorizer(max_features=5000, token_pattern=DEVA), MultinomialNB()),
    ("SVC char2-5", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5),
                                    max_features=50000),
     LinearSVC(C=1, random_state=SEED)),
]:
    A = vec.fit_transform(Xtr)
    clf.fit(A, ytr)
    preds[name] = clf.predict(vec.transform(Xte))

for s, q in preds.items():
    print(f"  {s:<14} macro F1 {f1_score(yte, q, average='macro'):.4f}   "
          f"weighted {f1_score(yte, q, average='weighted'):.4f}   "
          f"acc {(q == yte).mean():.4f}   CM {confusion_matrix(yte, q).tolist()}")

# ---------------- bootstrap + McNemar ----------------
rng = np.random.default_rng(SEED)
bi = rng.integers(0, len(yte), size=(B, len(yte)))
boot = {s: np.array([f1_score(yte[i], preds[s][i], average="macro",
                              zero_division=0) for i in bi]) for s in preds}
print("\n" + "=" * 84)
print("3. BOOTSTRAP 95% CI")
print("=" * 84)
for s in preds:
    lo_, hi_ = np.percentile(boot[s], [2.5, 97.5])
    print(f"  {s:<14} {f1_score(yte, preds[s], average='macro'):.4f}  "
          f"[{lo_:.4f}, {hi_:.4f}]")


def mcnemar(a, b):
    ao, bo = preds[a] == yte, preds[b] == yte
    n10, n01 = int((ao & ~bo).sum()), int((~ao & bo).sum())
    nd = n10 + n01
    if nd == 0:
        return n10, n01, 1.0
    if nd < 25:
        return n10, n01, float(binomtest(min(n01, n10), nd, 0.5).pvalue)
    return n10, n01, float(chi2.sf((abs(n01 - n10) - 1) ** 2 / nd, 1))


def holm(pv):
    m, o, adj, run = len(pv), np.argsort(pv), np.empty(len(pv)), 0.0
    for r, i in enumerate(o):
        run = max(run, (m - r) * pv[i])
        adj[i] = min(run, 1.0)
    return adj


rows, pvs = [], []
for a, b in itertools.combinations(preds, 2):
    n10, n01, pval = mcnemar(a, b)
    d = boot[a] - boot[b]
    rows.append(dict(a=a, b=b,
                     delta=f1_score(yte, preds[a], average="macro") -
                           f1_score(yte, preds[b], average="macro"),
                     lo=np.percentile(d, 2.5), hi=np.percentile(d, 97.5),
                     n10=n10, n01=n01, p=pval))
    pvs.append(pval)
pw = pd.DataFrame(rows)
pw["p_holm"] = holm(np.array(pvs))
print("\n" + "=" * 84)
print("4. PAIRWISE (Holm-corrected)")
print("=" * 84)
for _, r in pw.iterrows():
    print(f" {'*' if r.p_holm < 0.05 else ' '} {r.a:<14}vs {r.b:<14}"
          f"D={r.delta:+.4f} [{r.lo:+.4f},{r.hi:+.4f}] "
          f"n10={r.n10:<4} n01={r.n01:<4} p_holm={r.p_holm:.4f}")

# ---------------- the decisive split: profane vs non-profane offensive ----
print("\n" + "=" * 84)
print("5. WHERE THE LEXICON BREAKS: recall by offence type")
print("=" * 84)
print(f"{'system':<14}{'profane (L=2)':>16}{'offensive, no profanity (L=1)':>32}")
m2, m1 = (lab_te == 2), (lab_te == 1)
print(f"{'  n in test':<14}{m2.sum():>16}{m1.sum():>32}")
for s in preds:
    print(f"{s:<14}{(preds[s][m2] == 1).mean():>16.3f}"
          f"{(preds[s][m1] == 1).mean():>32.3f}")

print("\nmacro F1 restricted to non-profane offensive + all non-offensive:")
keep = (lab_te != 2)
for s in preds:
    print(f"  {s:<14} {f1_score(yte[keep], preds[s][keep], average='macro'):.4f}")

# ---------------- single-term dominance ----------------
print("\n" + "=" * 84)
print("6. SINGLE-TERM DOMINANCE (the 'dhoti' analogue)")
print("=" * 84)
print(f"{'term':<14}{'train n':>9}{'train %':>9}{'P(off|term)':>13}{'soleF1':>9}")
for t in terms[order[:8]]:
    m_tr = Xtr.str.contains(re.escape(t), regex=True).to_numpy()
    m_te = Xte.str.contains(re.escape(t), regex=True).to_numpy()
    sole = f1_score(yte, m_te.astype(int), average="macro")
    print(f"{t:<14}{m_tr.sum():>9}{m_tr.mean()*100:>8.1f}%"
          f"{ytr[m_tr].mean():>13.3f}{sole:>9.4f}")

# ---------------- near-duplicate leakage ----------------
def sh(t, k=5):
    t = re.sub(r"\s+", " ", t.strip())
    return {t[i:i + k] for i in range(max(1, len(t) - k + 1))} or {t}


trs = [sh(t) for t in Xtr]
inv = defaultdict(set)
for i, s_ in enumerate(trs):
    for g in s_:
        inv[g].add(i)
mj = []
for t in Xte:
    s_ = sh(t)
    cand = defaultdict(int)
    for g in s_:
        for i in inv.get(g, ()):
            cand[i] += 1
    mj.append(max((c / (len(s_) + len(trs[i]) - c) for i, c in cand.items()),
                  default=0.0))
mj = np.array(mj)
print("\n" + "=" * 84)
print("7. NEAR-DUPLICATE LEAKAGE AUDIT")
print("=" * 84)
for thr in (0.7, 0.8, 0.9):
    print(f"  test items with train near-dup at J >= {thr}: "
          f"{(mj >= thr).sum()} ({(mj >= thr).mean()*100:.2f}%)")
cl = mj < 0.8
print(f"\n  clean subset n = {cl.sum()}")
for s in preds:
    full = f1_score(yte, preds[s], average="macro")
    clean = f1_score(yte[cl], preds[s][cl], average="macro")
    print(f"  {s:<14} full {full:.4f}  clean {clean:.4f}  drop {full-clean:+.4f}")
