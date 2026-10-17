# The Diagnosticity Condition
[

![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23432737.svg)

](https://doi.org/10.5281/zenodo.23432737)
Simulation and analytic-validation code for:

> Avramescu, A. N. *The Diagnosticity Condition: A Dilution Theorem for Persistence-Based Evidence.* Manuscript under review, Foundations of Science.
>
> Code: Avramescu, A. N. (2026). *diagnosticity-condition* (v1.0.0). Zenodo. https://doi.org/10.5281/zenodo.23432737

This repository contains the two reproducible synthetic experiments
behind the paper's results. No real-world data is used anywhere; every
number in the paper's tables and figures is generated from the seeded
scripts below and can be regenerated exactly.

## What's here

| File | Produces | Paper section |
|---|---|---|
| `simulate_continuous.py` | Figure 1 (ROC curves), `results_continuous.json`, `sweep_continuous.json` | Section 4 — baseline demonstration (persistence vs. pure noise) |
| `diagnosticity.py` | Figure 2, Table 1, `diagnosticity_results.json` | Section 5 — the diagnosticity condition, the worked example, and the exact Poisson-binomial dilution theorem (Section 5.5) |

## Reproducing the results

```bash
pip install -r requirements.txt
python simulate_continuous.py
python diagnosticity.py
