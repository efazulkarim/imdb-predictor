# Comparison Table (single split)

Test set n=780, random_state=42, SBERT=all-MiniLM-L6-v2


## Test Metrics (point estimate [95% bootstrap CI])

| Model | RMSE | MAE | R² |
|---|---|---|---|
| predict_mean | 1.5184 [1.4561, 1.5794] | 1.2472 [1.1864, 1.3048] | -0.0007 [-0.0083, -0.0000] |
| ols_metadata | 1.1844 [1.1283, 1.2388] | 0.9459 [0.8939, 0.9937] | 0.3911 [0.3534, 0.4282] |
| ols_structural | 1.1292 [1.0682, 1.1863] | 0.8814 [0.8314, 0.9270] | 0.4466 [0.4070, 0.4892] |
| tfidf_xgboost | 1.1288 [1.0599, 1.1947] | 0.8539 [0.8009, 0.9033] | 0.4470 [0.3887, 0.5007] |
| sbert_xgboost | 1.0321 [0.9739, 1.0939] | 0.7889 [0.7438, 0.8372] | 0.5376 [0.4825, 0.5867] |
| sbert_xgboost_noweight | 0.9974 [0.9417, 1.0553] | 0.7487 [0.7014, 0.7932] | 0.5682 [0.5260, 0.6083] |

## Paired Comparison vs SBERT+XGBoost

| Baseline | ΔMAE (baseline − SBERT) | 95% CI | Wilcoxon p |
|---|---|---|---|
| predict_mean | +0.4583 | [+0.3936, +0.5219] | 4.25e-34 |
| ols_metadata | +0.1570 | [+0.1120, +0.2000] | 9.09e-11 |
| ols_structural | +0.0925 | [+0.0535, +0.1322] | 2.69e-05 |
| tfidf_xgboost | +0.0649 | [+0.0153, +0.1081] | 1.41e-02 |
| sbert_xgboost_noweight | -0.0402 | [-0.0655, -0.0144] | 1.20e-02 |

_Positive ΔMAE means SBERT has lower error (SBERT wins)._
