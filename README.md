# Credit Risk Modeling — PD, LGD, EAD & ECL

End-to-end credit risk modeling pipeline using the Lending Club dataset (2.26M loans, 2007–2018), implementing the IFRS 9 / Basel framework for Expected Credit Loss estimation.

## Key Results

| Metric | Value |
|---|---|
| **PD Test AUC** | 0.72 (LightGBM), 0.70 (Logistic Regression) |
| **PD Test KS** | 0.32 (LightGBM) — passes 0.30 industry threshold |
| **LGD MAE** | 0.147 (17% better than mean baseline) |
| **Portfolio ECL** | $585M predicted vs $654M actual (within 10%) |
| **Optimal cutoff strategy** | $143M profit vs $70M baseline (**+105%**) |

## Project Structure

```
credit-risk-modeling-lendingclub/
├── notebooks/
│   ├── 01_eda_and_prep.ipynb        # EDA + Data Preparation (Phases 1-3)
│   ├── 02_pd_model.ipynb            # PD Model (Phase 4)
│   └── 03_lgd_ead_ecl.ipynb         # LGD + EAD + ECL + Cutoff (Phases 5-8)
├── assumptions.md                   # Key modeling decisions documented
└── README.md
```

## Approach

### 1. Data Preparation
- Loaded 2.26M Lending Club loans
- Defined default per Basel 90+ DPD standard (Charged Off + Late 31-120 days)
- Filtered to 1.37M loans with observable outcomes
- Time-based train/test split (≤2016 / >2016) to simulate production deployment

### 2. PD Modeling
- **Logistic Regression** as interpretable baseline (industry-standard for scorecards)
- **LightGBM** as challenger model
- Excluded 37 post-funding features to prevent data leakage
- Engineered features: `credit_history_months`, `loan_to_income`, `fico_avg`

### 3. LGD Modeling
- LightGBM regression on 290K defaulted loans
- 17% improvement in MAE over historical-average baseline
- Used segment-level (by grade) LGD for ECL to avoid regression-to-mean

### 4. EAD & ECL
- EAD = funded_amnt (origination-time exposure)
- ECL = PD × LGD × EAD per loan, aggregated to portfolio

### 5. Cutoff Strategy
- Simulated 90 cutoff thresholds (0.05–0.95)
- Optimized for net profit (revenue – losses)
- Optimal cutoff = 0.34 → 76% approval rate, $143M profit (105% above baseline)

## Key Insights

1. **Vintage deterioration**: Bad rate increased from 13% (2010) to 27% (2017), matching Lending Club's known credit policy loosening
2. **LightGBM beats Logistic Regression** by 2.1pp AUC, capturing non-linear patterns
3. **Calibration drift**: Model under-predicts losses for grades A–D (2017-2018 vintage worse than historical pattern)
4. **Profit vs ROI optimize differently**: Total profit peaks at cutoff 0.34; ROI per dollar peaks at 0.10–0.15

## Tech Stack

`Python` `pandas` `scikit-learn` `LightGBM` `matplotlib` `seaborn` `Google Colab`

## How to Reproduce

1. Download Lending Club dataset from [Kaggle](https://www.kaggle.com/datasets/wordsforthewise/lending-club)
2. Place `accepted_2007_to_2018Q4.csv.gz` in Google Drive at `MyDrive/Colab Notebooks/Data/credit-risk-modeling/`
3. Open notebooks in Colab in sequential order (01 → 02 → 03)
4. Run all cells

## Limitations

- Logistic Regression hit max_iter on full data — could improve with more iterations or sample-then-extrapolate
- LGD model exhibits regression-to-mean (predicts middle values); could improve with beta regression
- Reject inference not addressed (rejected applicants excluded)
- Production deployment would need quarterly recalibration

## Author

Gorawit Khovintasets · [GitHub](https://github.com/Gorawit2002) · [Portfolio](https://gorawit-portfolio.vercel.app/)