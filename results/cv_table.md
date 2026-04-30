# Comparison Table (5-fold CV)

SBERT=all-MiniLM-L6-v2, seed=42


## Test Metrics (mean ± std across folds)

| Model | RMSE | MAE | R² |
|---|---|---|---|
| predict_mean | 1.4380 ± 0.0561 | 1.1749 ± 0.0566 | -0.0001 ± 0.0001 |
| ols_metadata | 1.1302 ± 0.0465 | 0.8829 ± 0.0417 | 0.3821 ± 0.0167 |
| ols_structural | 1.0763 ± 0.0395 | 0.8328 ± 0.0313 | 0.4396 ± 0.0068 |
| tfidf_xgboost | 1.0829 ± 0.0341 | 0.8192 ± 0.0220 | 0.4325 ± 0.0140 |
| sbert_xgboost | 0.9999 ± 0.0273 | 0.7678 ± 0.0243 | 0.5156 ± 0.0256 |
| sbert_xgboost_noweight | 0.9583 ± 0.0327 | 0.7197 ± 0.0268 | 0.5556 ± 0.0094 |

## Pooled Paired Comparison vs SBERT+XGBoost

| Baseline | ΔMAE (baseline − SBERT) | 95% CI | Wilcoxon p |
|---|---|---|---|
| predict_mean | +0.4070 | [+0.3835, +0.4293] | 9.34e-212 |
| ols_metadata | +0.1150 | [+0.1004, +0.1303] | 6.74e-37 |
| ols_structural | +0.0649 | [+0.0526, +0.0783] | 9.83e-16 |
| tfidf_xgboost | +0.0514 | [+0.0344, +0.0674] | 2.61e-05 |
| sbert_xgboost_noweight | -0.0482 | [-0.0572, -0.0391] | 2.50e-26 |

_Positive ΔMAE means SBERT has lower error (SBERT wins)._
