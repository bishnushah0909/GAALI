#!/usr/bin/env python3
"""Full analysis of GAALI test-set predictions across all six systems + lexicon."""
import os as _os
_D = _os.environ.get("GAALI_DATA_DIR", _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "data"))
import itertools
import re
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.stats import binomtest, chi2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (confusion_matrix, f1_score, precision_score,
                             recall_score)

PRED = _os.path.join(_D, "test_set_model_predictions.csv")
DATA = _os.path.join(_D, "corrected_cleaned_combined_data_kaggle_stratified.csv")
B, SEED, ALPHA = 1000, 42, 0.05

p = pd.read_csv(PRED)
d = pd.read_csv(DATA)
tr = d[d.split == "train"].reset_index(drop=True)
te = d[d.split == "test"].reset_index(drop=True)
y = p.true_label.to_numpy()
txt = p.Comment.astype(str)

NAMES = {"mbert_pred": "mBERT", "muril_pred": "MuRIL",
         "xlm_roberta_pred": "XLM-RoBERTa", "svc_pred": "Linear SVC",
         "indicbert_pred": "IndicBERT", "mnb_pred": "Multinomial NB"}
preds = {NAMES[c]: p[c].to_numpy() for c in NAMES}

# ---- lexicon-25 baseline, fitted on train only -------------------------
cv = TfidfVectorizer(max_features=5000, use_idf=False, norm=None,
                     binary=True).fit(tr.Comment.astype(str))
Btr = (cv.transform(tr.Comment.astype(str)) > 0).toarray()
ytr = tr.Label.to_numpy()
lo = np.log(((Btr[ytr == 1].sum(0) + 1) / (ytr == 1).sum()) /
            ((Btr[ytr == 0].sum(0) + 1) / (ytr == 0).sum()))
top25 = np.argsort(-lo)[:25]
terms = np.array(cv.get_feature_names_out())
Bte = (cv.transform(txt) > 0).toarray()
preds["Lexicon-25"] = (Bte[:, top25].sum(1) > 0).astype(int)
print("Lexicon-25 terms:", ", ".join(terms[top25]))

systems = list(preds)

# ---- 1. full metric table ----------------------------------------------
print("\n" + "=" * 96)
print("1. FULL METRICS (test n = %d)" % len(y))
print("=" * 96)
rows = []
for s in systems:
    q = preds[s]
    rows.append(dict(
        system=s,
        macroP=precision_score(y, q, average="macro", zero_division=0),
        macroR=recall_score(y, q, average="macro", zero_division=0),
        macroF1=f1_score(y, q, average="macro", zero_division=0),
        wP=precision_score(y, q, average="weighted", zero_division=0),
        wR=recall_score(y, q, average="weighted", zero_division=0),
        wF1=f1_score(y, q, average="weighted", zero_division=0),
        acc=(q == y).mean(),
        F1_off=f1_score(y, q, pos_label=1, zero_division=0),
        F1_non=f1_score(y, q, pos_label=0, zero_division=0),
        n_pred_off=int(q.sum()),
    ))
mt = pd.DataFrame(rows).sort_values("macroF1", ascending=False).reset_index(drop=True)
print(mt.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

print("\nConfusion matrices [[TN FP],[FN TP]]:")
for s in mt.system:
    print(f"  {s:<16}{confusion_matrix(y, preds[s]).tolist()}")

# ---- 2. bootstrap CIs + McNemar ----------------------------------------
rng = np.random.default_rng(SEED)
idx = rng.integers(0, len(y), size=(B, len(y)))
boot = {s: np.array([f1_score(y[i], preds[s][i], average="macro",
                              zero_division=0) for i in idx]) for s in systems}

print("\n" + "=" * 96)
print(f"2. MACRO F1 WITH 95 PCT PERCENTILE BOOTSTRAP CI (B = {B})")
print("=" * 96)
ci = {}
for s in mt.system:
    lo_, hi_ = np.percentile(boot[s], [2.5, 97.5])
    ci[s] = (lo_, hi_)
    print(f"  {s:<16} {f1_score(y, preds[s], average='macro'):.4f}  "
          f"[{lo_:.4f}, {hi_:.4f}]   width {hi_-lo_:.4f}")


def mcnemar(a, b):
    ao, bo = preds[a] == y, preds[b] == y
    n10, n01 = int((ao & ~bo).sum()), int((~ao & bo).sum())
    nd = n10 + n01
    if nd == 0:
        return n10, n01, float("nan"), 1.0, "none"
    if nd < 25:
        return n10, n01, float(min(n01, n10)), \
            float(binomtest(min(n01, n10), nd, 0.5).pvalue), "exact"
    st = (abs(n01 - n10) - 1) ** 2 / nd
    return n10, n01, float(st), float(chi2.sf(st, 1)), "chi2"


def holm(pv):
    m, order, adj, run = len(pv), np.argsort(pv), np.empty(len(pv)), 0.0
    for r, i in enumerate(order):
        run = max(run, (m - r) * pv[i])
        adj[i] = min(run, 1.0)
    return adj


pairs, pv = [], []
for a, b in itertools.combinations(list(mt.system), 2):
    n10, n01, st, pval, kind = mcnemar(a, b)
    dif = boot[a] - boot[b]
    dlo, dhi = np.percentile(dif, [2.5, 97.5])
    pairs.append(dict(a=a, b=b,
                      delta=f1_score(y, preds[a], average="macro") -
                            f1_score(y, preds[b], average="macro"),
                      dlo=dlo, dhi=dhi, n10=n10, n01=n01, p=pval, test=kind))
    pv.append(pval)
pw = pd.DataFrame(pairs)
pw["p_holm"] = holm(np.array(pv))
pw["sig"] = pw.p_holm < ALPHA

print("\n" + "=" * 96)
print("3. PAIRWISE McNEMAR + BOOTSTRAP CI ON THE DIFFERENCE (Holm-corrected)")
print("=" * 96)
for _, r in pw.iterrows():
    print(f" {'*' if r.sig else ' '} {r.a:<16}vs {r.b:<16}"
          f"D={r.delta:+.4f} [{r.dlo:+.4f},{r.dhi:+.4f}] "
          f"n10={r.n10:<3} n01={r.n01:<3} p={r.p:.4f} holm={r.p_holm:.4f}")
pw.to_csv("final_pairwise.csv", index=False)
mt.to_csv("final_metrics.csv", index=False)

# ---- 3. leakage-controlled subset ---------------------------------------
def shingles(t, k=5):
    t = re.sub(r"\s+", " ", str(t).strip().lower())
    return {t[i:i + k] for i in range(max(1, len(t) - k + 1))} or {t}


tr_sh = [shingles(t) for t in tr.Comment.astype(str)]
inv = defaultdict(set)
for i, sh in enumerate(tr_sh):
    for g in sh:
        inv[g].add(i)


def maxj(t):
    sh = shingles(t)
    cand = defaultdict(int)
    for g in sh:
        for i in inv.get(g, ()):
            cand[i] += 1
    return max((c / (len(sh) + len(tr_sh[i]) - c) for i, c in cand.items()),
               default=0.0)


mj = np.array([maxj(t) for t in txt])
clean = mj < 0.8
print("\n" + "=" * 96)
print(f"4. LEAKAGE-CONTROLLED SUBSET (Jaccard < 0.8; n = {clean.sum()} of {len(y)})")
print("=" * 96)
for s in mt.system:
    full = f1_score(y, preds[s], average="macro")
    cl = f1_score(y[clean], preds[s][clean], average="macro")
    lk = f1_score(y[~clean], preds[s][~clean], average="macro")
    print(f"  {s:<16} full {full:.4f}   clean {cl:.4f}   leaked {lk:.4f}   "
          f"drop {full-cl:+.4f}")

# ---- 4. dhoti ablation ---------------------------------------------------
dh = txt.str.contains("dhoti", case=False).to_numpy()
print("\n" + "=" * 96)
print(f"5. THE 'dhoti' SUBSET ({dh.sum()} of {len(y)} test comments, "
      f"{y[dh].mean()*100:.1f}% offensive)")
print("=" * 96)
for s in mt.system:
    print(f"  {s:<16} on dhoti {f1_score(y[dh], preds[s][dh], average='macro'):.4f}"
          f"   without dhoti {f1_score(y[~dh], preds[s][~dh], average='macro'):.4f}")

# ---- 5. subgroup recall --------------------------------------------------
groups = {"dhoti": r"\bdhoti", "madhesi": r"\bmadhesi|\bmadesi|\bmadhesh",
          "muslim": r"\bmuslim|\bmusalman", "newar": r"\bnewar",
          "tharu": r"\btharu", "dalit": r"\bdalit"}
print("\n" + "=" * 96)
print("6. RECALL ON OFFENSIVE COMMENTS BY TARGET GROUP")
print("=" * 96)
hdr = "group          n  off%  " + "".join(f"{s[:11]:>13}" for s in mt.system)
print(hdr)
for g, pat in groups.items():
    m = txt.str.contains(pat, case=False, regex=True).to_numpy()
    pos = m & (y == 1)
    if pos.sum() == 0:
        continue
    line = f"{g:<12}{m.sum():>4}{y[m].mean()*100:>6.0f}%"
    for s in mt.system:
        line += f"{(preds[s][pos] == 1).mean():>13.3f}"
    print(line)

# ---- 6. error overlap ----------------------------------------------------
print("\n" + "=" * 96)
print("7. ERROR STRUCTURE")
print("=" * 96)
err = {s: (preds[s] != y) for s in mt.system}
main = [s for s in mt.system if s != "Lexicon-25"]
allwrong = np.ones(len(y), bool)
for s in main:
    allwrong &= err[s]
anywrong = np.zeros(len(y), bool)
for s in main:
    anywrong |= err[s]
print(f"  instances misclassified by ALL six systems : {allwrong.sum()}")
print(f"  instances misclassified by at least one    : {anywrong.sum()}")
print(f"  oracle-ensemble ceiling (any system right) : "
      f"{1 - allwrong.mean():.4f} accuracy")
maj = (np.stack([preds[s] for s in main]).sum(0) >= 3).astype(int)
print(f"  majority vote of six systems, macro F1     : "
      f"{f1_score(y, maj, average='macro'):.4f}")
print("\n  pairwise error-overlap (Jaccard of error sets):")
for a, b in itertools.combinations(main, 2):
    inter = (err[a] & err[b]).sum()
    union = (err[a] | err[b]).sum()
    print(f"    {a:<16}{b:<16}{inter/union:.3f}  ({inter} shared of {union})")

print("\n  hardest instances (missed by all six):")
for t, yt in zip(txt[allwrong][:8], y[allwrong][:8]):
    print(f"    [true={yt}] {t[:96]}")
