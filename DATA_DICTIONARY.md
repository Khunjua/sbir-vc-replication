# Data Dictionary

Two anonymised firm-level datasets. Each row is one SBIR/STTR awardee,
keyed by `firm_id` (a stable `firm_00001`-style id that joins the CSV to
the matching `*_embeddings.npz`). Derived from public SBIR/STTR award
records and SEC Form D filings; see README for anonymisation caveats.

## Survival-outcome encoding (both files)

The outcome is time-to-event for a venture-capital follow-on, modelled
with Cox/XGBoost-Cox:

- `event_1m` — 1 if the firm filed an SEC Form D with total amount sold
  >= $1M at any point after its index date; else 0 (right-censored).
- `days_to_event_1m` — days from index date to the event (if
  `event_1m=1`) or to the end of observation (if censored).
- XGBoost-Cox label convention used by the scripts:
  `y = days_to_event_1m` if event else `-days_to_event_1m`.
- Concordance is computed with
  `sksurv.metrics.concordance_index_censored(event_bool, time, risk)`.

## phase2_anon.csv  (8463 rows x 75 cols; canonical analytic sample = 8332)

**Identifiers / split**
| column | meaning |
|---|---|
| `firm_id` | anonymised stable id (joins to phase2_embeddings.npz) |
| `index_date`, `index_year` | date / year of the firm's first Phase II award (the index event) |
| `split` | `train` (index <=2018) / `validate` (>=2019); the cross-break partition |

**Outcomes** (primary + robustness)
| column | meaning |
|---|---|
| `event_1m`, `days_to_event_1m` | **primary**: Form D >= $1M (see encoding above) |
| `event_5m`, `days_to_event_5m` | robustness: Form D >= $5M |
| `event_10m`, `days_to_event_10m` | robustness: Form D >= $10M |
| `censoring_horizon_days` | length of the observation window for the firm |

**Tier-1 trajectory features** (in the canonical model)
| column | meaning |
|---|---|
| `n_pre_grants` | # SBIR/STTR grants before the index Phase II |
| `total_pre_amount` | summed pre-index award $ (model uses `log1p`) |
| `max_single_award` | largest single pre-index award $ (model uses `log1p`) |
| `years_first_to_index_days` | days from first-ever grant to index |
| `n_grantors_pre` | # distinct funding agencies pre-index |
| `n_active_years_pre` | # distinct years with a grant pre-index |
| `n_phase1_pre` | # Phase I grants pre-index |
| `phase1_to_phase2_days` | days from first Phase I to index Phase II |
| `has_dod/has_nih/has_nsf/has_doe/has_nasa/has_usda` | agency indicator flags |
| `state_at_index` | US state (top-10 -> dummies, rest -> OTHER) |

**TRL block** (NOT in the canonical model; excluded per audit, Section 3.3.
`trl_at_index` is used only to drop 5 rows with missing TRL.)
| column | meaning |
|---|---|
| `n_trl_observations`, `trl_mean_pre`, `trl_volatility`, `trl_max_pre`, `trl_min_pre`, `trl_first_grant`, `trl_at_index`, `trl_delta`, `trl_velocity_per_year`, `trl_span_days` | LLM-derived Technology-Readiness-Level summaries over pre-index grants |

**Flags / buckets**
| column | meaning |
|---|---|
| `has_embedding` | 1 if an abstract embedding exists for the firm |
| `amt_bucket`, `trl_delta_bin` | categorical buckets (descriptive only) |
| `is_troll` | **1 = grant-mill firm excluded from the canonical sample** (126 firms). Formula: `n_pre_grants>30` OR (`years_active>15` AND `total_grants_ever_by_2022>20`). Scripts drop `is_troll==1` after the TRL filter -> n=8332. |

**Sector features** (in the canonical model)
| column | meaning |
|---|---|
| `vc_*` (35 columns) | per-firm cosine similarity to each level-1 root of the VC-Sectors taxonomy (e.g. `vc_SpaceTech`, `vc_BioTech`), 0 if the root is absent from the firm's first-grant top-10 tags. |

## phase1_anon.csv  (14156 rows x 20 cols)

Phase I anchored sample (Appendix A.4). No sector or troll columns — the
Phase I canonical model is the no-sector specification.

| column | meaning |
|---|---|
| `firm_id` | anonymised stable id (joins to phase1_embeddings.npz) |
| `index_date`, `index_year` | date / year of the firm's first Phase I award |
| `n_pre_grants`, `total_pre_amount`, `max_single_award`, `years_first_to_index_days`, `n_grantors_pre`, `n_active_years_pre`, `n_phase1_pre`, `n_phase2_pre` | Tier-1 trajectory (82% of Phase I firms have the Phase I as their first grant, so most pre-index counts are 0) |
| `has_dod/has_nih/has_nsf/has_doe/has_nasa/has_usda` | agency flags |
| `state_at_index` | US state |
| `event_1m`, `days_to_event_1m` | primary outcome (same encoding as Phase II) |

## Embeddings (`*_embeddings.npz`)

`firm_ids` (str array) and `embeddings` (float32, n x 1536) — OpenAI
`text-embedding-ada-002` vectors, mean-pooled over the firm's pre-index
grant abstracts. Join to the CSV on `firm_id`. The replication scripts
fit PCA-20 on the training fold and project the validation fold.
