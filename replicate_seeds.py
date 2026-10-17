"""
The Diagnosticity Condition -- multi-seed replication (fully reproducible).

diagnosticity.py and simulate_continuous.py each report ONE pre-specified
random draw (seed = 42). This script repeats the identical data-generating
processes over 2,000 independent seeds (0..1999) so that the single-seed values
can be compared with their sampling distribution and with the exact
Poisson-binomial model values. Seed 42 is included among the 2,000 and is
reported unchanged; no seed was selected or discarded.

AUC is invariant to any monotone transform, so the null-standardization
(ZI = (I - mu)/sd) used in the main scripts does not change any AUC and is
omitted here.

Outputs: replication_results.json, and a printed summary table.
"""
import json
import numpy as np
from scipy.stats import norm, rankdata
from sklearn.linear_model import LogisticRegression

N_SEEDS = 2000
N_GROUP = 250
ALPHA = 0.10
MU = 0.9
Z_CRIT = norm.ppf(1 - ALPHA)
SIGMA8 = np.array([0.9, 1.3, 0.8, 1.6, 1.1, 1.0, 1.4, 0.7])
SIGMA9 = 1.0
BANDWIDTH = 1.0
LAB = np.concatenate([np.ones(N_GROUP), np.zeros(N_GROUP)])


def auc_rank(scores, labels):
    ranks = rankdata(scores, method="average")
    n_pos = labels.sum()
    n_neg = len(labels) - n_pos
    return (ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def section5_draw(seed):
    """Same draw order as diagnosticity.py: structure, then artifact."""
    rng = np.random.default_rng(seed)
    out = []
    for sig8, sig9 in [(MU, MU), (MU, 0.0)]:          # structure, artifact
        n8 = rng.normal(0, 1, size=(N_GROUP, 8))
        n9 = rng.normal(0, 1, size=N_GROUP)
        z8 = (sig8 + n8 * SIGMA8[None, :]) / SIGMA8[None, :]
        z9 = (sig9 + n9 * SIGMA9) / SIGMA9
        out.append((z8, z9))
    return out


def section5_aucs(seed):
    (z8s, z9s), (z8a, z9a) = section5_draw(seed)
    I8s, I8a = (z8s > Z_CRIT).mean(1), (z8a > Z_CRIT).mean(1)
    I9s = ((z8s > Z_CRIT).sum(1) + (z9s > Z_CRIT)) / 9.0
    I9a = ((z8a > Z_CRIT).sum(1) + (z9a > Z_CRIT)) / 9.0
    a_i8 = auc_rank(np.r_[I8s, I8a], LAB)
    a_i9 = auc_rank(np.r_[I9s, I9a], LAB)
    a_best = auc_rank(np.r_[z8s.max(1), z8a.max(1)], LAB)
    a_t9 = auc_rank(np.r_[z9s, z9a], LAB)
    X = np.vstack([np.c_[I8s, z9s], np.c_[I8a, z9a]])
    clf = LogisticRegression().fit(X, LAB)
    a_orc = auc_rank(clf.decision_function(X), LAB)
    return a_i8, a_i9, a_best, a_t9, a_orc


def section4_aucs(seed):
    """Same draw order as simulate_continuous.py: structure, then null."""
    rng = np.random.default_rng(seed)
    z = {}
    for name, eff in [("s", MU), ("n", 0.0)]:
        noise = rng.normal(0, 1, size=(N_GROUP, 8))
        z[name] = (eff + noise * SIGMA8[None, :]) / SIGMA8[None, :]
    def stats(zz):
        return ((zz > Z_CRIT).mean(1),
                (1.0 / (1.0 + np.exp(-(zz - Z_CRIT) / BANDWIDTH))).mean(1),
                zz[:, 0])
    bs, cs, ss = stats(z["s"])
    bn, cn, sn = stats(z["n"])
    return (auc_rank(np.r_[ss, sn], LAB),    # single observation
            auc_rank(np.r_[bs, bn], LAB),    # binary Invariance Score
            auc_rank(np.r_[cs, cn], LAB))    # continuous Invariance Score


# Exact model values (Poisson-binomial, no simulation) -- as in diagnosticity.py
def pb_pmf(probs):
    pmf = np.array([1.0])
    for pr in probs:
        new = np.zeros(len(pmf) + 1)
        new[:-1] += pmf * (1 - pr)
        new[1:] += pmf * pr
        pmf = new
    return pmf


def exact_auc(pmf_pos, pmf_neg):
    gt = eq = 0.0
    for i, pi in enumerate(pmf_pos):
        for j, pj in enumerate(pmf_neg):
            if i > j:
                gt += pi * pj
            elif i == j:
                eq += pi * pj
    return gt + 0.5 * eq


q = 1 - norm.cdf(Z_CRIT - MU / SIGMA8)
p1 = 1 - norm.cdf(Z_CRIT - MU / SIGMA9)
p2 = 1 - norm.cdf(Z_CRIT)
EXACT = {
    "I8": exact_auc(pb_pmf(q), pb_pmf(q)),
    "I9": exact_auc(pb_pmf(np.r_[q, p1]), pb_pmf(np.r_[q, p2])),
    "T9_alone": float(norm.cdf(MU / np.sqrt(2.0))),   # equal-variance normal shift
    # Section 4 single observation = first transformation, standardized shift MU/SIGMA8[0]
    "single_observation": float(norm.cdf((MU / SIGMA8[0]) / np.sqrt(2.0))),
}

if __name__ == "__main__":
    s5 = np.array([section5_aucs(s) for s in range(N_SEEDS)])
    s4 = np.array([section4_aucs(s) for s in range(N_SEEDS)])
    rows = [
        ("Section 5: flat Invariance Score, 8 naive transformations", s5[:, 0], EXACT["I8"]),
        ("Section 5: flat Invariance Score, 8 naive + T9 (unweighted)", s5[:, 1], EXACT["I9"]),
        ("Section 5: single best specification", s5[:, 2], 0.5),
        ("Section 5: T9 alone", s5[:, 3], EXACT["T9_alone"]),
        ("Section 5: I8 + T9 logistic combination (in-sample)", s5[:, 4], None),
        ("Section 4: single observation (structure vs null)", s4[:, 0], EXACT["single_observation"]),
        ("Section 4: binary Invariance Score (structure vs null)", s4[:, 1], None),
        ("Section 4: continuous Invariance Score (structure vs null)", s4[:, 2], None),
    ]
    out = {"n_seeds": N_SEEDS, "n_group": N_GROUP, "rows": []}
    print(f"{'quantity':<62}{'seed42':>8}{'mean':>8}{'sd':>7}{'2.5%':>8}{'97.5%':>8}{'exact':>8}{'pctile42':>9}")
    for name, a, ex in rows:
        s42 = float(a[42])
        pct = float((a < s42).mean() * 100)
        r = {"quantity": name, "seed42": s42, "mean": float(a.mean()),
             "sd": float(a.std(ddof=1)), "p2_5": float(np.percentile(a, 2.5)),
             "p97_5": float(np.percentile(a, 97.5)), "exact": ex,
             "seed42_percentile": pct}
        out["rows"].append(r)
        exs = f"{ex:8.3f}" if ex is not None else "      --"
        print(f"{name:<62}{s42:8.3f}{r['mean']:8.3f}{r['sd']:7.3f}{r['p2_5']:8.3f}{r['p97_5']:8.3f}{exs}{pct:9.1f}")
    with open("replication_results.json", "w") as f:
        json.dump(out, f, indent=2)
