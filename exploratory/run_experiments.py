#!/usr/bin/env python3
"""Section 5.2 classical baselines on GAALI, with a leakage-controlled test set."""
import os as _os
_D = _os.environ.get("GAALI_DATA_DIR", _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "data"))
import re
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

CSV = _os.path.join(_D, "corrected_cleaned_combined_data_kaggle_stratified.csv")
SEED = 42

df = pd.read_csv(CSV)
tr = df[df.split == "train"].reset_index(drop=True)
te = df[df.split == "test"].reset_index(drop=True)
Xtr, ytr = tr.Comment.astype(str), tr.Label.to_numpy()
Xte, yte = te.Comment.astype(str), te.Label.to_numpy()


# ---------------------------------------------------------------- leakage mask
def shingles(t, k=5):
    t = re.sub(r"\s+", " ", str(t).strip().lower())
    return {t[i:i + k] for i in range(max(1, len(t) - k + 1))} or {t}


tr_sh = [shingles(t) for t in Xtr]
index = defaultdict(set)
for i, sh in enumerate(tr_sh):
    for g in sh:
        index[g].add(i)


def max_jaccard(t):
    sh = shingles(t)
    cand = defaultdict(int)
    for g in sh:
        for i in index.get(g, ()):
            cand[i] += 1
    best = 0.0
    for i, c in cand.items():
        best = max(best, c / (len(sh) + len(tr_sh[i]) - c))
    return best


te_max_j = np.array([max_jaccard(t) for t in Xte])
for thr in (0.7, 0.8, 0.9):
    print(f"test items with train near-dup at J >= {thr}: "
          f"{(te_max_j >= thr).sum()} ({(te_max_j >= thr).mean()*100:.2f}%)")
clean = te_max_j < 0.8
print(f"\nclean test subset: n = {clean.sum()}  "
      f"(offensive share {yte[clean].mean():.3f})\n")

# ---------------------------------------------------------------- systems
systems = {
    # paper spec: word unigrams, 5,000 features, sklearn defaults
    "SVC_word":  (TfidfVectorizer(max_features=5000), LinearSVC(C=1, random_state=SEED)),
    "MNB_word":  (TfidfVectorizer(max_features=5000), MultinomialNB(alpha=1.0)),
    "LR_word":   (TfidfVectorizer(max_features=5000),
                  LogisticRegression(max_iter=2000, random_state=SEED)),
    # diagnostic: character n-grams, better suited to unstable romanisation
    "SVC_char":  (TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5),
                                  max_features=50000), LinearSVC(C=1, random_state=SEED)),
}

preds_full, rows = {"y_true": yte}, []
for name, (vec, clf) in systems.items():
    Vtr = vec.fit_transform(Xtr)          # fitted on train only
    Vte = vec.transform(Xte)
    clf.fit(Vtr, ytr)
    p = clf.predict(Vte)
    preds_full[name] = p
    rows.append(dict(
        system=name,
        n_features=Vtr.shape[1],
        macro_f1_full=f1_score(yte, p, average="macro"),
        macro_f1_clean=f1_score(yte[clean], p[clean], average="macro"),
        macro_f1_leaked=f1_score(yte[~clean], p[~clean], average="macro"),
        weighted_f1_full=f1_score(yte, p, average="weighted"),
    ))

res = pd.DataFrame(rows)
res["drop"] = res.macro_f1_full - res.macro_f1_clean
print("=" * 84)
print("MACRO F1: full test vs. leakage-controlled subset")
print("=" * 84)
print(res.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

print("\n" + "=" * 84)
print("PER-CLASS REPORT (SVC_word, full test)")
print("=" * 84)
print(classification_report(yte, preds_full["SVC_word"],
                            target_names=["non-offensive", "offensive"], digits=4))
print("confusion matrix [rows=true, cols=pred]:")
print(confusion_matrix(yte, preds_full["SVC_word"]))

pd.DataFrame(preds_full).to_csv("test_predictions.csv", index=False)
clean_df = pd.DataFrame({k: (v[clean] if k != "y_true" else v[clean])
                         for k, v in preds_full.items()})
clean_df.to_csv("test_predictions_clean.csv", index=False)
np.save("clean_mask.npy", clean)
print("\nwrote test_predictions.csv, test_predictions_clean.csv")

# ---------------------------------------------------------------- subgroup errors
print("\n" + "=" * 84)
print("SUBGROUP ERROR ANALYSIS (SVC_word, full test)")
print("=" * 84)
groups = {
    "dhoti (anti-Indian slur)": r"\bdhoti\b|\bdhotii\b",
    "madhesi": r"\bmadhesi|\bmadesi|\bmadhise",
    "muslim": r"\bmuslim|\bmusalman",
    "newar": r"\bnewar|\bnewari",
    "tharu": r"\btharu\b",
    "dalit": r"\bdalit",
}
p = preds_full["SVC_word"]
print(f"{'subgroup':<26}{'n':>5}{'off%':>8}{'recall_off':>12}{'FN':>5}{'FP':>5}")
for g, pat in groups.items():
    m = te.Comment.astype(str).str.contains(pat, case=False, regex=True).to_numpy()
    if m.sum() == 0:
        continue
    pos = m & (yte == 1)
    rec = (p[pos] == 1).mean() if pos.sum() else float("nan")
    fn = int(((yte == 1) & (p == 0) & m).sum())
    fp = int(((yte == 0) & (p == 1) & m).sum())
    print(f"{g:<26}{m.sum():>5}{yte[m].mean()*100:>7.1f}%{rec:>12.4f}{fn:>5}{fp:>5}")
