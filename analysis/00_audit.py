#!/usr/bin/env python3
"""Dataset integrity + train/test leakage audit for GAALI."""
import os as _os
_D = _os.environ.get("GAALI_DATA_DIR", _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "data"))
import re
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

CSV = sys.argv[1] if len(sys.argv) > 1 else \
    _os.path.join(_D, "corrected_cleaned_combined_data_kaggle_stratified.csv")

df = pd.read_csv(CSV)
df.columns = [c.strip() for c in df.columns]
text_col, label_col, split_col = "Comment", "Label", "split"

print("=" * 78)
print("1. BASIC STATISTICS")
print("=" * 78)
n = len(df)
print(f"rows: {n}")
print(f"label counts: {df[label_col].value_counts().to_dict()}")
off = (df[label_col] == 1).mean()
print(f"offensive share: {off:.4f}  ({off*100:.2f}%)")

tokens = df[text_col].astype(str).str.split()
tok_counts = tokens.map(len)
all_tokens = [t for row in tokens for t in row]
vocab = set(all_tokens)
print(f"total tokens: {len(all_tokens):,}")
print(f"vocabulary (types): {len(vocab):,}")
print(f"mean tokens/comment: {tok_counts.mean():.2f}   median: {tok_counts.median():.0f}"
      f"   max: {tok_counts.max()}")
print(f"type-token ratio: {len(vocab)/len(all_tokens):.4f}")
print(f"comments > 128 whitespace tokens: {(tok_counts > 128).sum()} "
      f"({(tok_counts > 128).mean()*100:.2f}%)")

print()
print("=" * 78)
print("2. SPLIT BALANCE")
print("=" * 78)
tab = df.groupby([split_col, label_col]).size().unstack(fill_value=0)
tab["total"] = tab.sum(axis=1)
tab["off_share"] = tab[1] / tab["total"]
print(tab.to_string())
overall = off
print(f"\nmax |split share - overall share| = "
      f"{(tab['off_share'] - overall).abs().max():.5f}")

print()
print("=" * 78)
print("3. EXACT DUPLICATE / LEAKAGE AUDIT")
print("=" * 78)
norm = df[text_col].astype(str).str.strip().str.lower()
dup_total = norm.duplicated().sum()
print(f"exact duplicate strings anywhere in corpus: {dup_total}")

by_split = defaultdict(set)
for s, t in zip(df[split_col], norm):
    by_split[s].add(t)
for a, b in [("train", "test"), ("train", "dev"), ("dev", "test")]:
    inter = by_split[a] & by_split[b]
    print(f"exact overlap {a} <-> {b}: {len(inter)}")
    for x in list(inter)[:5]:
        print(f"    - {x[:90]}")

# duplicates carrying conflicting labels
conf = df.assign(_n=norm).groupby("_n")[label_col].nunique()
print(f"duplicate strings with conflicting labels: {(conf > 1).sum()}")

print()
print("=" * 78)
print("4. NEAR-DUPLICATE LEAKAGE (char 5-gram Jaccard, threshold 0.8)")
print("=" * 78)


def shingles(text, k=5):
    t = re.sub(r"\s+", " ", str(text).strip().lower())
    if len(t) < k:
        return {t}
    return {t[i:i + k] for i in range(len(t) - k + 1)}


train_txt = df.loc[df[split_col] == "train", text_col].astype(str).tolist()
test_txt = df.loc[df[split_col] == "test", text_col].astype(str).tolist()
train_sh = [shingles(t) for t in train_txt]
test_sh = [shingles(t) for t in test_txt]

# inverted index on shingles so we only compare plausible candidates
index = defaultdict(set)
for i, sh in enumerate(train_sh):
    for g in sh:
        index[g].add(i)

hits = []
for j, sh in enumerate(test_sh):
    cand = defaultdict(int)
    for g in sh:
        for i in index.get(g, ()):
            cand[i] += 1
    best_i, best_j = -1, 0.0
    for i, c in cand.items():
        jac = c / (len(sh) + len(train_sh[i]) - c)
        if jac > best_j:
            best_j, best_i = jac, i
    if best_j >= 0.8:
        hits.append((best_j, test_txt[j], train_txt[best_i]))

print(f"test comments with a train near-duplicate at Jaccard >= 0.80: "
      f"{len(hits)} / {len(test_txt)} ({len(hits)/len(test_txt)*100:.2f}%)")
for j, a, b in sorted(hits, reverse=True)[:8]:
    print(f"  J={j:.3f}")
    print(f"    test : {a[:88]}")
    print(f"    train: {b[:88]}")

np.save("nearu_hits.npy", np.array([h[1] for h in hits], dtype=object),
        allow_pickle=True)

print()
print("=" * 78)
print("5. EMOJI-AS-TEXT CHECK")
print("=" * 78)
emoji_words = ["face with tears of joy", "enraged face", "rolling on the floor",
               "loudly crying", "thumbs up", "folded hands", "red heart"]
for w in emoji_words:
    m = df[text_col].astype(str).str.contains(w, case=False, regex=False)
    if m.sum():
        print(f"  '{w}': {m.sum()} comments  (offensive share "
              f"{df.loc[m, label_col].mean():.3f})")
raw_emoji = df[text_col].astype(str).str.contains(
    r"[\U0001F300-\U0001FAFF\u2600-\u27BF]", regex=True)
print(f"  comments still containing raw emoji codepoints: {raw_emoji.sum()}")
