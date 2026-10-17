#!/usr/bin/env python3
"""
Section 5.6 — Statistical reliability for the GAALI benchmark.

Computes, for every system evaluated on the same held-out test set:
  * macro F1 point estimate
  * 95% percentile bootstrap confidence interval          [Eq. (14)]
  * pairwise McNemar tests on discordant pairs            [Eq. (15)]
  * pairwise bootstrap CIs on the DIFFERENCE in macro F1

Input: a CSV with one row per test instance, a column `y_true`, and one
column of predicted labels per system. Labels are 0/1.

Usage:
    python stat_reliability.py test_predictions.csv --out results
"""

import argparse
import itertools
import json

import numpy as np
import pandas as pd
from scipy.stats import binomtest, chi2
from sklearn.metrics import f1_score

B = 1000           # bootstrap resamples, as stated in Eq. (14)
ALPHA = 0.05
SEED = 42
EXACT_CUTOFF = 25  # use exact binomial McNemar when n01 + n10 < 25


# ----------------------------------------------------------------------
# Eq. (14): percentile bootstrap
# ----------------------------------------------------------------------
def bootstrap_indices(n, n_boot=B, seed=SEED):
    """One shared set of resample indices, reused for every system.

    This is what makes the bootstrap PAIRED: system A and system B are
    always scored on the identical resample, so the difference
    distribution reflects only the models, not two independent draws.
    """
    rng = np.random.default_rng(seed)
    return rng.integers(0, n, size=(n_boot, n))


def macro_f1(y_true, y_pred):
    return f1_score(y_true, y_pred, average="macro", zero_division=0)


def bootstrap_scores(y_true, y_pred, idx):
    """Macro F1 on each bootstrap resample. Returns shape (n_boot,)."""
    return np.array([macro_f1(y_true[i], y_pred[i]) for i in idx])


def percentile_ci(samples, alpha=ALPHA):
    lo, hi = np.percentile(samples, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


# ----------------------------------------------------------------------
# Eq. (15): McNemar
# ----------------------------------------------------------------------
def mcnemar(y_true, pred_a, pred_b, exact_cutoff=EXACT_CUTOFF):
    """McNemar's test on the discordant pairs.

    n10 = A right, B wrong;  n01 = A wrong, B right.

    Uses the exact binomial test when the discordant count is small
    (the chi-square approximation is unreliable there) and the
    continuity-corrected chi-square of Eq. (15) otherwise.
    """
    a_ok = pred_a == y_true
    b_ok = pred_b == y_true
    n10 = int(np.sum(a_ok & ~b_ok))
    n01 = int(np.sum(~a_ok & b_ok))
    n_disc = n01 + n10

    if n_disc == 0:
        return dict(n10=n10, n01=n01, stat=float("nan"), p=1.0, test="none")

    if n_disc < exact_cutoff:
        p = binomtest(min(n01, n10), n_disc, 0.5).pvalue
        return dict(n10=n10, n01=n01, stat=float(min(n01, n10)),
                    p=float(p), test="exact binomial")

    stat = (abs(n01 - n10) - 1) ** 2 / n_disc          # Eq. (15)
    p = float(chi2.sf(stat, df=1))
    return dict(n10=n10, n01=n01, stat=float(stat), p=p,
                test="chi2 (continuity-corrected)")


def holm(pvals):
    """Holm-Bonferroni step-down adjustment for the family of pairwise tests."""
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        val = (m - rank) * pvals[i]
        running = max(running, val)
        adj[i] = min(running, 1.0)
    return adj


# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", help="CSV with y_true + one column per system")
    ap.add_argument("--out", default="stat_reliability")
    ap.add_argument("--n-boot", type=int, default=B)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    if "y_true" not in df.columns:
        raise SystemExit("CSV must contain a `y_true` column.")

    y_true = df["y_true"].to_numpy()
    systems = [c for c in df.columns if c != "y_true"]
    preds = {s: df[s].to_numpy() for s in systems}
    n = len(y_true)

    idx = bootstrap_indices(n, args.n_boot, args.seed)
    boot = {s: bootstrap_scores(y_true, preds[s], idx) for s in systems}

    # ---- per-system point estimate + CI ------------------------------
    rows = []
    for s in systems:
        lo, hi = percentile_ci(boot[s])
        point = macro_f1(y_true, preds[s])
        rows.append(dict(system=s, macro_f1=point, ci_low=lo, ci_high=hi,
                         ci_width=hi - lo))
    per_system = pd.DataFrame(rows).sort_values("macro_f1", ascending=False)

    # ---- pairwise -----------------------------------------------------
    pairs, raw_p = [], []
    for a, b in itertools.combinations(systems, 2):
        mc = mcnemar(y_true, preds[a], preds[b])
        d = boot[a] - boot[b]                      # paired difference
        d_lo, d_hi = percentile_ci(d)
        pairs.append(dict(
            system_a=a, system_b=b,
            delta_macro_f1=macro_f1(y_true, preds[a]) - macro_f1(y_true, preds[b]),
            delta_ci_low=d_lo, delta_ci_high=d_hi,
            ci_excludes_zero=bool(d_lo > 0 or d_hi < 0),
            n10=mc["n10"], n01=mc["n01"],
            mcnemar_stat=mc["stat"], mcnemar_p=mc["p"], test=mc["test"],
        ))
        raw_p.append(mc["p"])

    pairwise = pd.DataFrame(pairs)
    pairwise["p_holm"] = holm(np.array(raw_p))
    pairwise["significant_holm"] = pairwise["p_holm"] < ALPHA

    # ---- output -------------------------------------------------------
    per_system.to_csv(f"{args.out}_per_system.csv", index=False)
    pairwise.to_csv(f"{args.out}_pairwise.csv", index=False)

    print(f"\nTest set: n = {n}   bootstrap B = {args.n_boot}   seed = {args.seed}\n")
    print("Macro F1 with 95% percentile bootstrap CI")
    print("-" * 52)
    for _, r in per_system.iterrows():
        print(f"  {r.system:<26} {r.macro_f1:.4f}  "
              f"[{r.ci_low:.4f}, {r.ci_high:.4f}]")

    print("\nPairwise comparisons (Holm-adjusted)")
    print("-" * 76)
    for _, r in pairwise.iterrows():
        mark = "*" if r.significant_holm else " "
        print(f" {mark} {r.system_a:<14} vs {r.system_b:<14} "
              f"Δ={r.delta_macro_f1:+.4f} [{r.delta_ci_low:+.4f}, {r.delta_ci_high:+.4f}]  "
              f"n10={r.n10:<3} n01={r.n01:<3} p={r.mcnemar_p:.4f} "
              f"p_holm={r.p_holm:.4f}")
    print("\n  * = significant at alpha = 0.05 after Holm correction")

    # ---- LaTeX fragment for the results table -------------------------
    with open(f"{args.out}_table.tex", "w") as fh:
        fh.write("% macro F1 with 95% bootstrap CI\n")
        for _, r in per_system.iterrows():
            fh.write(f"{r.system} & {r.macro_f1:.4f} "
                     f"& [{r.ci_low:.4f}, {r.ci_high:.4f}] \\\\\n")

    with open(f"{args.out}_summary.json", "w") as fh:
        json.dump(dict(n=int(n), n_boot=int(args.n_boot), seed=int(args.seed),
                       per_system=per_system.to_dict("records"),
                       pairwise=pairwise.to_dict("records")), fh, indent=2)

    print(f"\nWrote {args.out}_per_system.csv, {args.out}_pairwise.csv, "
          f"{args.out}_table.tex, {args.out}_summary.json")


if __name__ == "__main__":
    main()
