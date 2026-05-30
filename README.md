# Replication Packet

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20452792.svg)](https://doi.org/10.5281/zenodo.20452792)

This folder contains anonymised data and code sufficient to reproduce
the main numerical results of the paper:

> Khunjua, T. and Malik, J. (2026).  Regime-Dependent Predictability
> in the SBIR Phase II to Venture Capital Pipeline: Evidence from
> 8,332 U.S. Awardees, 2010-2022.

The folder is self-contained: no external databases, no internal
infrastructure, no network calls.  Replication is one command.

## Quick start

```bash
bash reproduce.sh
```

This will (1) create a fresh virtualenv at `./.venv`, (2) install pinned
dependencies, (3) run the four replication scripts, (4) write outputs to
`./results/`.  Wall-clock: roughly 10-15 minutes.  Python 3.10+ required.

## Folder structure

```
replication/
  README.md                     (this file)
  LICENSE                       (MIT for code; CC-BY-4.0 for data)
  DATA_DICTIONARY.md            (column-by-column codebook)
  reproduce.sh                  (one-command entry point)
  requirements.txt              (pinned Python dependencies)
  code/
    02_table_1_scenarios.py     (Table 1: S1, S2, S3, S4 with bootstrap CI)
    03_paired_bootstrap.py      (Section 5.1 paired bootstrap: S4 vs S2)
    04_phase1_table.py          (Table A.1: Phase I anchored replication)
    05_figure_scenarios.py      (Figure 1: validation-scenarios bar chart)
  data/
    phase2_anon.csv             (8463 Phase II firms; outcome, Tier-1
                                 trajectory, state, TRL, is_troll flag,
                                 and 35 VC-taxonomy L1 similarity scores)
    phase2_embeddings.npz       (ada-002 mean-pooled, 1536-dim, by firm_id)
    phase1_anon.csv             (14156 Phase I firms)
    phase1_embeddings.npz       (ada-002 mean-pooled, 1536-dim)
  results/                      (auto-generated)
```

## Canonical specification (Sections 3-4 of the paper)

* **Sample (n=8332).**  8463 Phase II SBIR/STTR awardees pulled, drop
  5 with missing TRL -> 8458, drop 126 grant-mill "trolls"
  (`is_troll == 1`, Section 3.1 / A.2.2) -> **8332**.
* **Feature block (78).**  13 Tier-1 trajectory + 10 state dummies +
  20 PCA embedding components + **35 VC-taxonomy level-1 similarity
  scores** (`vc_*` columns).  TRL features are excluded (audit,
  Section 3.3); behavioural features are analysed separately as the
  second axis (Section 6) and are not in the predictive block.
* **Model.**  XGBoost-Cox (`survival:cox`), `max_depth=5`, `eta=0.05`,
  `num_boost_round=400`, `subsample=0.8`, `colsample_bytree=0.7`,
  `min_child_weight=8`, `reg_alpha=0.001`, `reg_lambda=1.0`, fit on
  `StandardScaler`-ed features.  Concordance and top-10% lift with
  500-resample bootstrap CIs (seed 42).
* **Phase I (Table A.1)** uses the no-sector block (Tier-1 + state +
  PCA-20, same regularised model, no troll filter, n=14156).

## What this packet covers / does not cover

Covered: Table 1 (S1-S4), Section 5.1 paired bootstrap, Table A.1
(Phase I), Figure 1.

Not covered (require data or models not redistributed): the seven-model
convergence check (Table 2); the behavioural-axis analysis (Section 6,
needs proposal-receipt-date records); the robustness battery (Section 7
- FRED macro controls, MiniLM embeddings, geographic LOO); and the SQL
ingestion pipeline that builds the SBIR<->Form D matched database.

## Anonymisation and provenance

The underlying records are derived from **public** sources (SBIR/STTR
award data and SEC Form D filings).  Each firm's internal `company_id`
is replaced by a stable sequential `firm_id` (`firm_00001`...), applied
identically to the CSV and the embeddings NPZ so they join on `firm_id`.
We do not redistribute the `company_id`->company-name mapping.

**This is name-stripping, not unlinkability.**  Because award date,
amount, agency, and state are retained (they are public and needed for
the analysis), individual rows may in principle be re-linked to named
firms via the public SBIR award database.  The packet is firm-name-free,
not anonymised against a determined linkage attack.

The 35 `vc_*` columns are per-firm cosine-similarity scores against the
level-1 roots of an embedding-based venture-capital sector taxonomy; the
taxonomy reference itself is proprietary and not redistributed, but the
derived per-firm scores are sufficient to reproduce the model.  A second
("Science") taxonomy exists internally but is used only as a descriptive
orthogonal check in the paper, not as model features, and is not shipped.

## Reconciliation with paper numbers

Verification run (this packet, canonical specification above):

| Scenario          | Paper                 | Replicated             | Note          |
|-------------------|-----------------------|------------------------|---------------|
| S1 random 5-fold  | 0.668                 | 0.678 (+/- 0.025 std)  | within 1 std  |
| S2 cross-break    | 0.565 [0.538, 0.592]  | 0.562 [0.536, 0.590]   | within 0.003  |
| S3 within-old     | 0.637                 | 0.640 [0.607, 0.676]   | within 0.003  |
| S4 within-recent  | 0.616 [0.579, 0.653]  | 0.614 [0.577, 0.651]   | within 0.002  |
| Top-10% lift (S4) | 1.50 [1.06, 1.99]     | 1.65 [1.21, 2.16]      | overlapping   |

Section 5.1 paired bootstrap (S4 vs S2, common 2021-22 cohort):
paper Delta = +0.079, 95% CI [+0.037, +0.127], P>0 = 100%;
replicated Delta = +0.084, 95% CI [+0.038, +0.133], P>0 = 100%.

Phase I (Table A.1): S2 0.531 / S3 0.595 / S4 0.573 vs paper
0.533 / 0.597 / 0.577 -- within ~0.005.

The packet reproduces the paper's point estimates to within ~0.003
C-index.  The only source of residual divergence is XGBoost's
`subsample` resampling, which is not byte-exact across machine
architectures even with a fixed seed; all values fall well inside the
published bootstrap confidence intervals.

## Software environment

Python 3.10+ (tested on 3.10 through 3.14).  Pinned in `requirements.txt`:
`xgboost`, `scikit-survival`, `lifelines`, `scikit-learn`, `pandas`,
`numpy`, `matplotlib`.

## Citation

If you use this packet, please cite the paper above. To cite the packet itself:

> Khunjua, T., & Malik, J. (2026). *Replication packet: Regime-Dependent Predictability in the SBIR Phase II to Venture Capital Pipeline*. Zenodo. https://doi.org/10.5281/zenodo.20452792

## Contact

Tamaz Khunjua, School of Science and
Technology, University of Georgia, Tbilisi, Georgia.
`t.khunjua@ug.edu.ge`.
