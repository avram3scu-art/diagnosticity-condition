"""
Invariance Thesis v0.5 -- continuous persistence simulation (seed=42, fully reproducible)

Extends the v0.4 binary-survival demonstration by replacing the binary
persistence indicator P(Ti) in {0,1} with a continuous relaxation
P(Ti) in [0,1], defined as a smooth (logistic) function of distance
from the same per-transformation detection threshold used in v0.4.
This does NOT use knowledge of the true effect size in its construction
(it is computable from data alone), so it remains a fair, non-circular
comparison against the binary version.

No real-world data is used. This script demonstrates only that the
continuous relaxation of the Invariance Score outperforms the binary
version on synthetic data with a known ground truth, exactly as the
binary version (v0.4, Section 6) outperformed single-observation
detection.
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
TRUE_EFFECT = 0.9
N_TRANSFORM = 8
SIGMA = np.array([0.9, 1.3, 0.8, 1.6, 1.1, 1.0, 1.4, 0.7])
BANDWIDTH = 1.0  # logistic smoothing scale for the continuous indicator

Z_CRIT = norm.ppf(1 - ALPHA)


def simulate_group(n, true_effect, n_transform, sigma, rng):
    noise = rng.normal(0.0, 1.0, size=(n, n_transform))
    obs = true_effect + noise * sigma[np.newaxis, :]
    z = obs / sigma[np.newaxis, :]
    survives_binary = (z > Z_CRIT).astype(float)
    # Continuous relaxation: logistic function of distance from threshold.
    # At z = Z_CRIT, P = 0.5; P -> 1 as z grows past threshold, P -> 0 below it.
    survives_continuous = 1.0 / (1.0 + np.exp(-(z - Z_CRIT) / BANDWIDTH))
    return z, survives_binary, survives_continuous


def auc_rank(scores, labels):
    ranks = rankdata(scores, method="average")
    n_pos = labels.sum()
    n_neg = len(labels) - n_pos
    return (ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def roc_curve_manual(scores, labels):
    thresholds = np.sort(np.unique(scores))[::-1]
    tpr_list, fpr_list = [1.0], [1.0]
    P = labels.sum(); N = len(labels) - P
    for t in thresholds:
        pred = scores >= t
        tp = ((pred == 1) & (labels == 1)).sum()
        fp = ((pred == 1) & (labels == 0)).sum()
        tpr_list.append(tp / P)
        fpr_list.append(fp / N)
    tpr_list.append(0.0); fpr_list.append(0.0)
    return np.array(fpr_list), np.array(tpr_list)


rng = np.random.default_rng(SEED)
z_s, surv_b_s, surv_c_s = simulate_group(N_GROUP, TRUE_EFFECT, N_TRANSFORM, SIGMA, rng)
z_n, surv_b_n, surv_c_n = simulate_group(N_GROUP, 0.0, N_TRANSFORM, SIGMA, rng)
_, surv_b_surr, surv_c_surr = simulate_group(N_NULL_SURROGATES, 0.0, N_TRANSFORM, SIGMA, rng)

# Binary I and ZI (reproducing v0.4 primary configuration, same seed/path)
I_b_s = surv_b_s.mean(1); I_b_n = surv_b_n.mean(1)
mu_b = surv_b_surr.mean(1).mean(); sd_b = surv_b_surr.mean(1).std(ddof=1)
ZI_b_s = (I_b_s - mu_b) / sd_b; ZI_b_n = (I_b_n - mu_b) / sd_b

# Continuous I and ZI
I_c_s = surv_c_s.mean(1); I_c_n = surv_c_n.mean(1)
mu_c = surv_c_surr.mean(1).mean(); sd_c = surv_c_surr.mean(1).std(ddof=1)
ZI_c_s = (I_c_s - mu_c) / sd_c; ZI_c_n = (I_c_n - mu_c) / sd_c

labels = np.concatenate([np.ones(N_GROUP), np.zeros(N_GROUP)])
baseline_scores = np.concatenate([z_s[:, 0], z_n[:, 0]])
inv_b_scores = np.concatenate([ZI_b_s, ZI_b_n])
inv_c_scores = np.concatenate([ZI_c_s, ZI_c_n])

auc_baseline = auc_rank(baseline_scores, labels)
auc_binary = auc_rank(inv_b_scores, labels)
auc_continuous = auc_rank(inv_c_scores, labels)

fpr_base, tpr_base = roc_curve_manual(baseline_scores, labels)
fpr_bin, tpr_bin = roc_curve_manual(inv_b_scores, labels)
fpr_con, tpr_con = roc_curve_manual(inv_c_scores, labels)

results = {
    "seed": SEED, "n_group": N_GROUP, "n_transform": N_TRANSFORM,
    "true_effect": TRUE_EFFECT, "bandwidth": BANDWIDTH,
    "auc_single_observation": float(auc_baseline),
    "auc_binary_invariance_score": float(auc_binary),
    "auc_continuous_invariance_score": float(auc_continuous),
    "gap_binary_vs_baseline": float(auc_binary - auc_baseline),
    "gap_continuous_vs_binary": float(auc_continuous - auc_binary),
    "gap_continuous_vs_baseline": float(auc_continuous - auc_baseline),
}
with open("results_continuous.json", "w") as f:
    json.dump(results, f, indent=2)
for k, v in results.items():
    print(k, "=", v)

# Robustness sweep, binary vs continuous, same grid as v0.4
sweep_rows = []
for eff in [0.6, 0.8, 0.9, 1.0, 1.1]:
    for n_t in [5, 8]:
        rng2 = np.random.default_rng(SEED)
        sig = SIGMA[:n_t]
        zs, sbs, scs = simulate_group(N_GROUP, eff, n_t, sig, rng2)
        zn, sbn, scn = simulate_group(N_GROUP, 0.0, n_t, sig, rng2)
        _, sbsurr, scsurr = simulate_group(N_NULL_SURROGATES, 0.0, n_t, sig, rng2)
        Ibs, Ibn = sbs.mean(1), sbn.mean(1)
        mub, sdb = sbsurr.mean(1).mean(), sbsurr.mean(1).std(ddof=1)
        ZIbs, ZIbn = (Ibs - mub) / sdb, (Ibn - mub) / sdb
        Ics, Icn = scs.mean(1), scn.mean(1)
        muc, sdc = scsurr.mean(1).mean(), scsurr.mean(1).std(ddof=1)
        ZIcs, ZIcn = (Ics - muc) / sdc, (Icn - muc) / sdc
        lab2 = np.concatenate([np.ones(N_GROUP), np.zeros(N_GROUP)])
        ab = auc_rank(np.concatenate([zs[:, 0], zn[:, 0]]), lab2)
        abin = auc_rank(np.concatenate([ZIbs, ZIbn]), lab2)
        acon = auc_rank(np.concatenate([ZIcs, ZIcn]), lab2)
        sweep_rows.append({
            "true_effect_size": eff, "n_transformations": n_t,
            "auc_single_observation": float(ab),
            "auc_binary": float(abin),
            "auc_continuous": float(acon),
            "gap_continuous_minus_binary": float(acon - abin),
        })
with open("sweep_continuous.json", "w") as f:
    json.dump(sweep_rows, f, indent=2)
print("\nSweep (continuous vs binary):")
for r in sweep_rows:
    print(r)

# --- Figure: ROC comparison, three curves ---
fig, ax = plt.subplots(figsize=(5.6, 4.6))
ax.plot(fpr_base, tpr_base, label=f"Single observation (AUC={auc_baseline:.3f})", color="#dd8c1e", linewidth=2)
ax.plot(fpr_bin, tpr_bin, label=f"Binary Invariance Score (AUC={auc_binary:.3f})", color="#8fa8bf", linewidth=2)
ax.plot(fpr_con, tpr_con, label=f"Continuous Invariance Score (AUC={auc_continuous:.3f})", color="#2f6ca2", linewidth=2.5)
ax.plot([0, 1], [0, 1], "--", color="#999999", linewidth=1)
ax.set_xlabel("False positive rate")
ax.set_ylabel("True positive rate")
ax.set_title("Continuous persistence further improves detection")
ax.legend(fontsize=8, loc="lower right")
plt.tight_layout()
plt.savefig("figure_continuous.png", dpi=200)
print("Figure saved: figure_continuous.png")
