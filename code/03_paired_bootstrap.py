#!/usr/bin/env python3
"""
03_paired_bootstrap.py -- paired-cohort bootstrap difference test
for S4 (within-recent) vs S2 (cross-cohort).  Reproduces the test
reported in Section 5.1 of the paper.

Both the S4-trained model (trained on 2018-20) and the S2-trained
model (trained on 2010-18) are evaluated on the same 2021-22
validation cohort (a strict subset of both validation cohorts).
The validation cohort is bootstrap-resampled 500 times; we report
the bootstrap distribution of Delta = C(S4) - C(S2), the 95%
percentile CI, and the share of resamples with Delta > 0.

Canonical 78-feature block (Tier-1 + state + PCA-20 + 35 VC-taxonomy
L1 similarity), regularised XGBoost-Cox on standardised features,
126-firm troll exclusion (n=8332).

Paper claim (Section 5.1): Delta has bootstrap mean +0.079 with
95% CI [+0.037, +0.127], P(Delta > 0) = 100%.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from sksurv.metrics import concordance_index_censored

DATA = Path(__file__).resolve().parent.parent / "data"
SEED = 42
N_BOOT = 500

XGB_PARAMS = dict(
    objective="survival:cox",
    tree_method="hist",
    max_depth=5,
    eta=0.05,
    subsample=0.8,
    colsample_bytree=0.7,
    min_child_weight=8,
    reg_alpha=0.001,
    reg_lambda=1.0,
    verbosity=0,
)
NUM_BOOST_ROUND = 400

TIER1 = [
    "log_total_pre", "log_max_award", "n_pre_grants", "n_grantors_pre",
    "n_phase1_pre", "years_first_to_index_days", "n_active_years_pre",
    "has_dod", "has_nih", "has_nsf", "has_doe", "has_nasa", "has_usda",
]


def load_data():
    ml = pd.read_csv(DATA / "phase2_anon.csv")
    emb_file = np.load(DATA / "phase2_embeddings.npz", allow_pickle=False)
    emb_ids = np.array(emb_file["firm_ids"])
    emb_vec = np.array(emb_file["embeddings"])

    ml = ml.dropna(subset=["trl_at_index"]).reset_index(drop=True)
    ml = ml[ml["is_troll"] == 0].reset_index(drop=True)
    ml["log_total_pre"] = np.log1p(ml["total_pre_amount"])
    ml["log_max_award"] = np.log1p(ml["max_single_award"])

    top10 = ml["state_at_index"].value_counts().head(10).index.tolist()
    ml["state_grp"] = ml["state_at_index"].where(ml["state_at_index"].isin(top10), "OTHER")
    state_d = pd.get_dummies(ml["state_grp"], prefix="state", drop_first=True).astype(int)
    state_cols = list(state_d.columns)
    ml = pd.concat([ml, state_d], axis=1)

    vc_cols = [c for c in ml.columns if c.startswith("vc_")]
    for c in vc_cols:
        ml[c] = ml[c].fillna(0.0)

    emb_idx = {fid: i for i, fid in enumerate(emb_ids)}
    emb_aligned = emb_vec[ml["firm_id"].map(emb_idx).values]
    return ml, emb_aligned, state_cols, vc_cols


def fit_xgb_cox(X_tr, y_event, y_time):
    y_xgb = np.where(y_event, y_time, -y_time)
    dtr = xgb.DMatrix(X_tr, label=y_xgb)
    return xgb.train(XGB_PARAMS, dtr, num_boost_round=NUM_BOOST_ROUND)


def build_features(ml, emb_aligned, train_mask, test_mask, state_cols, vc_cols):
    pca = PCA(n_components=20, random_state=SEED)
    pca_tr = pca.fit_transform(emb_aligned[train_mask])
    pca_te = pca.transform(emb_aligned[test_mask])
    base = TIER1 + state_cols + vc_cols
    X_tr = np.concatenate([ml.loc[train_mask, base].values, pca_tr], axis=1)
    X_te = np.concatenate([ml.loc[test_mask, base].values, pca_te], axis=1)
    scaler = StandardScaler()
    return scaler.fit_transform(X_tr), scaler.transform(X_te)


def main():
    print("=" * 70)
    print(" Paired bootstrap: S4 (within-recent) vs S2 (cross-cohort)")
    print(" Common validation cohort: 2021-22 firms")
    print("=" * 70)

    ml, emb_aligned, state_cols, vc_cols = load_data()
    print(f"Sample: {len(ml)} firms; feature block "
          f"{len(TIER1)}+{len(state_cols)}+20+{len(vc_cols)} = "
          f"{len(TIER1)+len(state_cols)+20+len(vc_cols)}.")

    s2_train = ml["index_year"].between(2010, 2018).values
    s4_train = ml["index_year"].between(2018, 2020).values
    common_val = ml["index_year"].between(2021, 2022).values
    print(f"S2 train: {s2_train.sum()};  S4 train: {s4_train.sum()};  "
          f"common 2021-22 val cohort: {common_val.sum()}")

    X_s2_tr, X_s2_val = build_features(ml, emb_aligned, s2_train, common_val, state_cols, vc_cols)
    X_s4_tr, X_s4_val = build_features(ml, emb_aligned, s4_train, common_val, state_cols, vc_cols)

    y_s2_tr_e = ml.loc[s2_train, "event_1m"].values.astype(bool)
    y_s4_tr_e = ml.loc[s4_train, "event_1m"].values.astype(bool)
    y_s2_tr_t = ml.loc[s2_train, "days_to_event_1m"].values
    y_s4_tr_t = ml.loc[s4_train, "days_to_event_1m"].values

    y_val_e = ml.loc[common_val, "event_1m"].values.astype(bool)
    y_val_t = ml.loc[common_val, "days_to_event_1m"].values

    print("\nFitting S2 model (train 2010-18)...")
    bst_s2 = fit_xgb_cox(X_s2_tr, y_s2_tr_e, y_s2_tr_t)
    print("Fitting S4 model (train 2018-20)...")
    bst_s4 = fit_xgb_cox(X_s4_tr, y_s4_tr_e, y_s4_tr_t)

    risk_s2 = bst_s2.predict(xgb.DMatrix(X_s2_val))
    risk_s4 = bst_s4.predict(xgb.DMatrix(X_s4_val))

    c_s2_pt, *_ = concordance_index_censored(y_val_e, y_val_t, risk_s2)
    c_s4_pt, *_ = concordance_index_censored(y_val_e, y_val_t, risk_s4)
    print(f"\nPoint estimates on common 2021-22 cohort:")
    print(f"  C(S2 model)   = {c_s2_pt:.4f}")
    print(f"  C(S4 model)   = {c_s4_pt:.4f}")
    print(f"  Delta         = {c_s4_pt - c_s2_pt:+.4f}")

    rng = np.random.default_rng(SEED)
    deltas, c_s2_boots, c_s4_boots = [], [], []
    n = len(y_val_e)
    for _ in range(N_BOOT):
        idx = rng.integers(0, n, n)
        if y_val_e[idx].sum() < 5:
            continue
        try:
            c2, *_ = concordance_index_censored(y_val_e[idx], y_val_t[idx], risk_s2[idx])
            c4, *_ = concordance_index_censored(y_val_e[idx], y_val_t[idx], risk_s4[idx])
        except Exception:
            continue
        c_s2_boots.append(c2)
        c_s4_boots.append(c4)
        deltas.append(c4 - c2)

    deltas = np.array(deltas)
    c_s2_boots = np.array(c_s2_boots)
    c_s4_boots = np.array(c_s4_boots)

    print(f"\nBootstrap (B = {len(deltas)} valid resamples, seed 42):")
    print(f"  C(S2) bootstrap mean = {c_s2_boots.mean():.4f}, "
          f"95% CI [{np.percentile(c_s2_boots,2.5):.4f}, {np.percentile(c_s2_boots,97.5):.4f}]")
    print(f"  C(S4) bootstrap mean = {c_s4_boots.mean():.4f}, "
          f"95% CI [{np.percentile(c_s4_boots,2.5):.4f}, {np.percentile(c_s4_boots,97.5):.4f}]")
    print(f"\n  Delta = C(S4) - C(S2) on common 2021-22 cohort:")
    print(f"    mean   = {deltas.mean():+.4f}")
    print(f"    95% CI = [{np.percentile(deltas,2.5):+.4f}, {np.percentile(deltas,97.5):+.4f}]")
    print(f"    P(Delta > 0) = {(deltas > 0).mean():.3f}")
    print(f"    P(Delta > 0.02) = {(deltas > 0.02).mean():.3f}")
    print()
    print("Paper Section 5.1 claim: Delta = +0.079, 95% CI [+0.037, +0.127], P > 0 = 100%.")


if __name__ == "__main__":
    main()
