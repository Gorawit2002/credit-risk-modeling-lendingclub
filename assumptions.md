# Modeling Assumptions & Decision Log

This document records key methodological decisions made during the Credit Risk Modeling project, along with the rationale for each. Every assumption here is something I can defend in interviews.

---

## 1. Default Definition

**Decision:** Default = `Charged Off` + `Default` + `Late (31-120 days)`

**Rationale:**
- Aligns with **Basel framework's 90+ DPD definition** of default
- `Charged Off` (268,559 loans) — Lending Club has written off the loan
- `Default` (40 loans) — officially defaulted
- `Late (31-120 days)` (21,467 loans) — 90+ DPD = default per Basel
- Non-default = `Fully Paid` only

**Excluded from modeling:**
- `Current` (878,317) — outcome unknown, still active
- `In Grace Period` + `Late (16-30 days)` — not yet 90 DPD
- `Does not meet the credit policy` — legacy data
- `NaN` — missing status

**Trade-off:** Lost 40% of sample size, but gained clean observable outcomes.

---

## 2. Train/Test Split: Time-Based

**Decision:** Train = issue_year ≤ 2016, Test = issue_year > 2016

**Rationale:**
- Mirrors **production deployment** — train on historical, predict future
- Tests model against **concept drift** (changing credit conditions)
- Industry standard for credit risk modeling
- Final split: 1.13M train (bad rate 20%) / 240K test (bad rate 26%)

**Trade-off:** Test set has higher bad rate (vintage deterioration), making evaluation harder but more realistic.

---

## 3. Data Leakage Prevention

**Decision:** Excluded 37 features that are populated AFTER loan funding

**Categories excluded:**
- Payment information: `total_pymnt`, `total_rec_prncp`, `total_rec_int`, `recoveries`
- Last update: `last_pymnt_d`, `last_fico_range_*`, `last_credit_pull_d`
- Outstanding: `out_prncp`, `out_prncp_inv`
- Hardship: all `hardship_*` columns (post-funding only)
- Settlement: all `settlement_*` columns (post-default only)

**Rationale:** PD model must predict default using **only information available at loan application time**. Using post-funding info would give artificially high AUC but fail in production.

**Note:** These same features ARE used in LGD modeling (where defaults have occurred and recoveries are observable).

---

## 4. Feature Engineering

### 4.1 Created Features
- **`credit_history_months`** = months between `earliest_cr_line` and `issue_d`
  - Top 5 feature in LightGBM importance — validated engineering choice
- **`fico_avg`** = (fico_range_low + fico_range_high) / 2
  - Combines redundant columns
- **`loan_to_income`** = funded_amnt / annual_inc
  - Standard credit risk metric (debt burden)
- **`installment_to_income`** = (installment × 12) / annual_inc
  - Debt service ratio

### 4.2 Outlier Treatment
- `annual_inc` capped via log transformation (`annual_inc_log`) due to skewness (max $10.9M)
- `loan_to_income` and `installment_to_income` winsorized at 99th percentile
- `credit_history_months` capped at 600 months (50 years) to remove data errors
- `revol_util` and `bc_util` capped at 200 (legitimate values ≤ 100%)

### 4.3 Placeholder Value Cleaning
- `dti` = 999 or -1 → NaN (placeholder for missing)
- `mo_sin_old_il_acct` = 999 → NaN
- `tot_hi_cred_lim` = 9,999,999 → NaN
- `total_rev_hi_lim` = 9,999,999 → NaN

### 4.4 Dropped Features (Beyond Leakage)

Additional features dropped from PD model for the following reasons:

| Feature | Reason | Action |
|---|---|---|
| `policy_code` | Constant value (always 1) — zero variance | Drop |
| `emp_title` | 381,808 unique values (free text) — high cardinality | Drop |
| `title` | 61,683 unique values; redundant with `purpose` | Drop |
| `zip_code` | 944 unique values; covered by `addr_state` | Drop |
| `funded_amnt_inv` | Correlation 0.999 with `funded_amnt` (duplicate) | Drop |

**Trade-off:** Could have used target encoding for `emp_title`/`title` but added complexity not warranted given low marginal value.

### 4.5 Missing Value Imputation

**Strategy:**
- **Numeric features**: median imputation (robust to outliers)
- **Categorical features**: mode imputation (most frequent value)
- **Imputation done on FULL dataset** rather than train-only — limitation acknowledged

**Trade-off:** For production, would fit imputer on train only and apply to test/new data to avoid subtle leakage. For EDA simplicity, applied to full data.

### 4.6 Categorical Encoding

| Type | Features | Method |
|---|---|---|
| **Ordinal** (natural order) | `grade` (A=1..G=7), `sub_grade` (A1=1..G5=35), `emp_length` (0..10) | Manual ordinal mapping |
| **Nominal** (no order) | `home_ownership`, `purpose`, `verification_status`, `application_type`, `initial_list_status`, `disbursement_method`, `addr_state` | One-hot encoding (`drop_first=True`) |

**Result:** 73 columns → 138 columns after encoding (65 dummy variables added)

### 4.7 Feature Scaling

- **Logistic Regression**: StandardScaler applied (mean=0, std=1) — required because LR uses gradient descent
- **LightGBM**: No scaling (tree-based algorithms are scale-invariant)
- Scaler fit on **train only**, then transformed on test (prevents data leakage)
---

## 5. Logistic Regression — Training Decisions

**Decision:** Trained on full 1.13M train set with `saga` solver

**Trade-off:**
- Training took 1 hour on Colab free (CPU constraints)
- Model hit `max_iter=1000` without full convergence
- AUC stable; would gain <0.005 with more iterations

**For production:** Would use `max_iter=2000+` and possibly L1 regularization to handle multicollinearity (observed sign flip in `num_sats` vs `open_acc`).

**Class weighting note:** Used `class_weight='balanced'` but this primarily affects threshold-based metrics (precision/recall), not AUC. The same is true for LightGBM — using `is_unbalance=True` caused premature early stopping, so removed.

---

## 6. LightGBM Hyperparameters

**Final settings:**
```
n_estimators=1000, learning_rate=0.03, num_leaves=31,
min_child_samples=100, reg_alpha=0.1, reg_lambda=0.1
```

**Rationale:**
- Lower learning rate (0.03 vs default 0.1) for stable convergence
- Higher `min_child_samples` (100 vs default 20) to prevent leaf overfitting
- L1/L2 regularization for generalization
- Reached 997/1000 iterations before plateau — could train longer but marginal gain

---

## 7. LGD Modeling

### 7.1 LGD Calculation
**Formula:**
```
Net Recovery = total_rec_prncp + recoveries - collection_recovery_fee
LGD = (funded_amnt - Net Recovery) / funded_amnt
LGD bounded to [0, 1]
```

### 7.2 Train/Test Split for LGD
**Decision:** Random 80/20 (not time-based like PD)

**Rationale:**
- Recovery process is relatively stable across time periods (no major regime change)
- Time-based would introduce selection bias (recent 2018 defaults haven't fully resolved)
- Random gives more representative test set

### 7.3 LGD Used for ECL
**Decision:** Used **grade-level mean LGD** (not loan-level model predictions) for ECL

**Rationale:**
- LightGBM LGD model exhibits **regression-to-mean** (predictions cluster 0.4-0.8, can't capture extremes)
- Grade-level LGD calibrates correctly to observed defaults
- Standard IFRS 9 practice (segment-level LGD)
- LGD by grade: A=54%, B=57%, C=62%, D=66%, E=70%, F=73%, G=77%

---

## 8. EAD Definition

**Decision:** EAD = `funded_amnt` (original loan amount)

**Rationale:**
- Models ECL at **origination time** (decision point of approval)
- Borrower hasn't repaid principal yet at this point
- Represents maximum potential exposure
- For active loan monitoring, EAD = `out_prncp` (current outstanding) would be used instead

---

## 9. ECL Calculation

**Formula:**
```
ECL_per_loan = PD × LGD_by_grade × funded_amnt
Portfolio ECL = Σ ECL_per_loan
```

**Result:**
- Portfolio EAD: $3.52B
- Predicted ECL: $585M (16.6% of EAD)
- Actual losses: $654M (18.6% of EAD)
- **10.5% under-prediction**, concentrated in grades A-D (matches vintage analysis findings)

---

## 10. Cutoff Strategy Optimization

**Objective:** Maximize total net profit (revenue – losses)

**Profit model:**
```
Revenue (if non-default) = installment × term - funded_amnt   (total interest)
Loss (if default) = LGD × funded_amnt
```

**Result:**
- Optimal cutoff = 0.34
- Approval rate = 76.2%
- Net profit = $143M (vs $70M baseline)
- **Improvement: +$73M (+105%)**

**Alternative criterion:** Maximizing ROI per dollar exposed peaks at lower cutoffs (0.10–0.15), reflecting margin-focused vs volume-focused strategies.

---

## Known Limitations & Future Work

| Issue | Description | Mitigation |
|---|---|---|
| **Calibration drift** | Model under-predicts losses in grades A-D for 2017-2018 vintages | Quarterly recalibration; calibration adjustment overlay |
| **LR convergence** | Hit max_iter without full convergence | Increase max_iter or use sampled training |
| **LGD regression-to-mean** | Model predicts middle values, can't capture extreme LGD | Beta regression, two-stage model, or mixture model |
| **Reject inference** | Excluded rejected applicants → sample bias | Apply parceling or augmentation technique |
| **Test set selection bias** | 2018 vintage incomplete (recent defaults not yet observed) | Use longer hold-out period in production |
| **Multicollinearity in LR** | `num_sats` sign flip relative to `open_acc` | Apply VIF filtering or L1 regularization |
| **2018 vintage incomplete** | Most 2018 loans still `Current` and filtered out; remaining sample skewed toward early defaulters | Note in interpretation; use longer observation window in production |
| **2007 vintage low sample** | Only 251 loans in 2007 — bad rate of 18% may be noise | Kept for completeness but flagged as low-confidence |
---

## Reproducibility

- `random_state = 42` used throughout
- Time-based split is deterministic (based on `issue_year`)
- All models + preprocessing artifacts saved as `.pkl` files
- Original Lending Club CSV from Kaggle: `wordsforthewise/lending-club`