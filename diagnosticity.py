"""
The Diagnosticity Condition -- simulation and exact analytic validation, seed=42, fully reproducible.

NOTE ON TIE HANDLING: an earlier version of auc_rank() used ordinal (stable-sort) ranks.
For a discrete, heavily-tied statistic -- the Invariance Score I is a sum of 8 or 9
binary indicators, so it takes only 9 or 10 distinct values -- ordinal ranking under
a stable sort assigns tied observations ranks in their ORIGINAL array order. Because
the structure group is concatenated before the artifact group, ties are systematically
broken in structure's favor on the low side and artifact's favor on the high side,
producing a small but real downward bias on exactly the coarse, discrete statistics
(I8, I9) this paper's headline numbers depend on. The continuous-score statistics
(T9 alone, the oracle combination) are unaffected, since they have no ties.

FIX: auc_rank now uses average (mid-)ranks for ties, matching the standard
Mann-Whitney U / Wilcoxon rank-sum convention and matching sklearn's roc_auc_score.

This script also adds an EXACT analytic cross-check: because every persistence
indicator here is Bernoulli and the 8 naive channels are independent and identically
distributed between the Structure and Artifact groups by construction, the sampling
distribution of I under each group is a Poisson-binomial distribution computable by
exact convolution (no Monte Carlo, no Gaussian approximation). The exact AUC implied
by this distribution is reported alongside the simulated AUC as an independent,
non-simulated validation of Result 1 and Result 2 -- and is the basis of the
aggregation (dilution) theorem in Section 5.5 of the manuscript.
"""
import numpy as np
import json
from scipy.stats import norm, rankdata
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEED = 42
N_GROUP = 250
N_NULL_SURROGATES = 2000
ALPHA = 0.10
MU_STRUCTURE = 0.9
SIGMA8 = np.array([0.9, 1.3, 0.8, 1.6, 1.1, 1.0, 1.4, 0.7])  # T1..T8
SIGMA9 = 1.0                                                  # T9 (diagnostic)
Z_CRIT = norm.ppf(1 - ALPHA)


def auc_rank(scores, labels):
    """Mann-Whitney-U-equivalent AUC using tie-averaged (mid-)ranks."""
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels)
    ranks = rankdata(scores, method="average")
    n_pos = labels.sum()
    n_neg = len(labels) - n_pos
    return (ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def gen_group(n, kind, rng):
    noise8 = rng.normal(0, 1, size=(n, 8))
    noise9 = rng.normal(0, 1, size=n)
    if kind == "structure":
        signal8, signal9 = MU_STRUCTURE, MU_STRUCTURE
    elif kind == "artifact":
        signal8, signal9 = MU_STRUCTURE, 0.0
    else:
        signal8, signal9 = 0.0, 0.0
    obs8 = signal8 + noise8 * SIGMA8[None, :]
    z8 = obs8 / SIGMA8[None, :]
    obs9 = signal9 + noise9 * SIGMA9
    z9 = obs9 / SIGMA9
    return z8, z9


rng = np.random.default_rng(SEED)
z8_struct, z9_struct = gen_group(N_GROUP, "structure", rng)
z8_artif, z9_artif = gen_group(N_GROUP, "artifact", rng)
z8_null, z9_null = gen_group(N_GROUP, "null", rng)
z8_surr, z9_surr = gen_group(N_NULL_SURROGATES, "null", rng)


def binary_I(z8, z9=None):
    surv8 = (z8 > Z_CRIT).astype(float)
    if z9 is None:
        return surv8.mean(axis=1)
    surv9 = (z9 > Z_CRIT).astype(float)
    allsurv = np.concatenate([surv8, surv9[:, None]], axis=1)
    return allsurv.mean(axis=1)


I8_struct, I8_artif, I8_null, I8_surr = (binary_I(z8_struct), binary_I(z8_artif),
                                          binary_I(z8_null), binary_I(z8_surr))
I9_struct, I9_artif, I9_null, I9_surr = (binary_I(z8_struct, z9_struct), binary_I(z8_artif, z9_artif),
                                          binary_I(z8_null, z9_null), binary_I(z8_surr, z9_surr))

mu8, sd8 = I8_surr.mean(), I8_surr.std(ddof=1)
mu9, sd9 = I9_surr.mean(), I9_surr.std(ddof=1)
ZI8_struct, ZI8_artif, ZI8_null = (I8_struct - mu8) / sd8, (I8_artif - mu8) / sd8, (I8_null - mu8) / sd8
ZI9_struct, ZI9_artif, ZI9_null = (I9_struct - mu9) / sd9, (I9_artif - mu9) / sd9, (I9_null - mu9) / sd9

lab_struct_vs_artif = np.concatenate([np.ones(N_GROUP), np.zeros(N_GROUP)])
lab_struct_vs_null = lab_struct_vs_artif

auc8_struct_vs_artif = auc_rank(np.concatenate([ZI8_struct, ZI8_artif]), lab_struct_vs_artif)
auc9_struct_vs_artif = auc_rank(np.concatenate([ZI9_struct, ZI9_artif]), lab_struct_vs_artif)
auc8_struct_vs_null = auc_rank(np.concatenate([ZI8_struct, ZI8_null]), lab_struct_vs_null)
auc8_artif_vs_null = auc_rank(np.concatenate([ZI8_artif, ZI8_null]), lab_struct_vs_null)
auc9_struct_vs_null = auc_rank(np.concatenate([ZI9_struct, ZI9_null]), lab_struct_vs_null)
auc9_artif_vs_null = auc_rank(np.concatenate([ZI9_artif, ZI9_null]), lab_struct_vs_null)

auc_singlebest_struct_vs_artif = auc_rank(
    np.concatenate([z8_struct.max(axis=1), z8_artif.max(axis=1)]), lab_struct_vs_artif)

auc_z9alone_struct_vs_artif = auc_rank(
    np.concatenate([z9_struct, z9_artif]), lab_struct_vs_artif)

from sklearn.linear_model import LogisticRegression  # noqa: E402
X_train = np.concatenate([
    np.stack([I8_struct, z9_struct], axis=1),
    np.stack([I8_artif, z9_artif], axis=1),
])
y_train = lab_struct_vs_artif
clf = LogisticRegression().fit(X_train, y_train)
oracle_scores = clf.decision_function(X_train)
auc_oracle_struct_vs_artif = auc_rank(oracle_scores, lab_struct_vs_artif)

# ---------------------------------------------------------------------------
# EXACT ANALYTIC CROSS-CHECK (Poisson-binomial convolution, no simulation)
# ---------------------------------------------------------------------------
def pb_pmf(probs):
    """Exact Poisson-binomial pmf of the number of successes, by convolution."""
    pmf = np.array([1.0])
    for pr in probs:
        new = np.zeros(len(pmf) + 1)
        new[:-1] += pmf * (1 - pr)
        new[1:] += pmf * pr
        pmf = new
    return pmf


def exact_auc_from_pmfs(pmf_pos, pmf_neg):
    """Exact rank-based AUC (mid-rank convention) for two independent
    integer-valued distributions, by direct summation over all outcome pairs."""
    k = np.arange(len(pmf_pos))
    P_gt = 0.0
    P_eq = 0.0
    for ks in k:
        if pmf_pos[ks] == 0:
            continue
        for ka in k:
            if pmf_neg[ka] == 0:
                continue
            if ks > ka:
                P_gt += pmf_pos[ks] * pmf_neg[ka]
            elif ks == ka:
                P_eq += pmf_pos[ks] * pmf_neg[ka]
    return P_gt + 0.5 * P_eq


q_analytic = 1 - norm.cdf(Z_CRIT - MU_STRUCTURE / SIGMA8)        # T1-T8, identical both groups
p1_analytic = 1 - norm.cdf(Z_CRIT - MU_STRUCTURE / SIGMA9)       # T9, structure
p2_analytic = 1 - norm.cdf(Z_CRIT)                               # T9, artifact (no signal)

pmf_struct_I9 = pb_pmf(np.concatenate([q_analytic, [p1_analytic]]))
pmf_artif_I9 = pb_pmf(np.concatenate([q_analytic, [p2_analytic]]))
exact_auc_I9 = exact_auc_from_pmfs(pmf_struct_I9, pmf_artif_I9)

pmf_struct_I8 = pb_pmf(q_analytic)
pmf_artif_I8 = pb_pmf(q_analytic)
exact_auc_I8 = exact_auc_from_pmfs(pmf_struct_I8, pmf_artif_I8)

# The dilution ratio implied by the exact model: standardized separation of the
# diagnostic channel alone vs. standardized separation after unweighted 9-way
# averaging, expressed as a variance-weighted dilution factor (Proposition 2).
V_nondiag = float(np.sum(q_analytic * (1 - q_analytic)))        # summed variance, T1-T8
V_diag = float((p1_analytic * (1 - p1_analytic) + p2_analytic * (1 - p2_analytic)) / 2)
dilution_factor = float(1.0 / np.sqrt(1.0 + V_nondiag / V_diag))

results = {
    "seed": SEED, "n_group": N_GROUP, "mu_structure": MU_STRUCTURE,
    "tie_handling": "average (mid-rank); corrects an earlier ordinal-rank implementation",
    "mean_I8_structure": float(I8_struct.mean()),
    "mean_I8_artifact": float(I8_artif.mean()),
    "mean_I8_null": float(I8_null.mean()),
    "mean_I9_structure": float(I9_struct.mean()),
    "mean_I9_artifact": float(I9_artif.mean()),
    "mean_I9_null": float(I9_null.mean()),
    "AUC_I8_structure_vs_null": float(auc8_struct_vs_null),
    "AUC_I8_artifact_vs_null": float(auc8_artif_vs_null),
    "AUC_I8_structure_vs_artifact__NAIVE_8_TRANSFORMATIONS": float(auc8_struct_vs_artif),
    "AUC_I9_structure_vs_null": float(auc9_struct_vs_null),
    "AUC_I9_artifact_vs_null": float(auc9_artif_vs_null),
    "AUC_I9_structure_vs_artifact__WITH_DIAGNOSTIC_T9": float(auc9_struct_vs_artif),
    "AUC_single_best_specification_structure_vs_artifact": float(auc_singlebest_struct_vs_artif),
    "AUC_T9_diagnostic_transformation_ALONE_structure_vs_artifact": float(auc_z9alone_struct_vs_artif),
    "AUC_oracle_weighted_combination_of_I8_and_T9_structure_vs_artifact_INSAMPLE": float(auc_oracle_struct_vs_artif),
    "EXACT_analytic_AUC_I8_structure_vs_artifact__Poisson_binomial_no_simulation": float(exact_auc_I8),
    "EXACT_analytic_AUC_I9_structure_vs_artifact__Poisson_binomial_no_simulation": float(exact_auc_I9),
    "analytic_q_T1_T8": q_analytic.tolist(),
    "analytic_p1_T9_structure": float(p1_analytic),
    "analytic_p2_T9_artifact": float(p2_analytic),
    "variance_weighted_dilution_factor": dilution_factor,
    "naive_1_over_sqrt_N_dilution_factor_for_comparison": float(1 / np.sqrt(9)),
}
with open("diagnosticity_results.json", "w") as f:
    json.dump(results, f, indent=2)
for k, v in results.items():
    print(k, "=", v)

# --- Figure (manuscript Fig. 2): no in-figure title; panel letters only ---
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
    "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "legend.fontsize": 7, "axes.linewidth": 0.6,
})
fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.9))

bins = np.linspace(-0.2, 1.0, 13)
axes[0].hist(I8_null, bins=bins, alpha=0.55, label="Null", color="#999999")
axes[0].hist(I8_artif, bins=bins, alpha=0.55, label="Artifact (confound)", color="#dd8c1e")
axes[0].hist(I8_struct, bins=bins, alpha=0.55, label="Structure (real effect)", color="#2f6ca2")
axes[0].set_xlabel("Invariance Score I (8 naive transformations)")
axes[0].set_ylabel("Number of candidates")
axes[0].legend(loc="upper right", frameon=False)
axes[0].text(-0.16, 1.02, "a", transform=axes[0].transAxes, fontsize=10, fontweight="bold")

zbins = np.linspace(-3.5, 4.5, 17)
axes[1].hist(z9_null, bins=zbins, alpha=0.55, label="Null", color="#999999")
axes[1].hist(z9_artif, bins=zbins, alpha=0.55, label="Artifact (confound)", color="#dd8c1e")
axes[1].hist(z9_struct, bins=zbins, alpha=0.55, label="Structure (real effect)", color="#2f6ca2")
axes[1].set_xlabel("z-score on diagnostic transformation T9")
axes[1].set_ylabel("Number of candidates")
axes[1].legend(loc="upper right", frameon=False)
axes[1].text(-0.16, 1.02, "b", transform=axes[1].transAxes, fontsize=10, fontweight="bold")

plt.tight_layout()
plt.savefig("figure_diagnosticity.png", dpi=600)
plt.savefig("figure_diagnosticity.pdf")
print("Figure saved.")
