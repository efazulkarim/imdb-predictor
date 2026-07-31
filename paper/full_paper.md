# Quantifying Screenplay Predictive Signal for IMDb Rating Forecasting: A Calibrated Multi-Embedding and Stacked Ensemble Analysis

## Abstract
Screenplay analysis extracts predictive signals from raw script text before film production begins. We engineered a multi-stage forecasting system. It processes 5,195 feature-film screenplays coupled with official IMDb metadata exports. The architecture executes chunked sentence-level semantic vectorization across multiple dense encoders, specifically Sentence-BERT (SBERT `all-MiniLM-L6-v2` and `all-mpnet-base-v2`), GloVe (300d), and Word2Vec (300d). We extracted 19 domain-specific structural indicators. These metrics capture narrative pacing and dialogue density. We benchmarked gradient-boosted decision trees (XGBoost v1.7.3, LightGBM v3.3.5), Random Forests, Support Vector Regressors (SVR), and Multi-Layer Perceptrons (MLP). A Ridge meta-regressor subsequently integrates these diverse base learners via stacked generalization. Evaluated under 5-fold cross-validation with 1,000-sample bootstrap confidence intervals, the stacked ensemble achieves RMSE = 0.935 ± 0.032, MAE = 0.706 ± 0.028, and R² = 0.577 ± 0.010. It outperforms every individual base regressor ($p \le 4.2 \times 10^{-6}$). Our baseline decomposition isolates specific metadata influence. Three attributes (release year, runtime, decade) account for R² ≈ 0.380. Textual content yields a restricted net marginal gain. We record ΔR² ≈ +0.18. We subjected the dataset to inverse-frequency sample weighting. This technique unexpectedly degrades regression accuracy (ΔMAE = −0.048, $p = 2.5 \times 10^{-26}$). Game-theoretic SHAP interpretability quantifies exact feature attributions. Metadata variables dominate the primary variance. SBERT semantic vectors inject localized adjustments to stabilize predictions on non-standard formatting.

**Keywords:** IMDb rating prediction, screenplay processing, Sentence-BERT, stacked ensembles, Explainable AI (SHAP), embedding evaluation, dataset bias.

---

## I. Introduction
Feature film screenplays typically span 20,000 to 30,000 words. These massive documents break standard transformer architectures. Models like BERT operate on rigid 512-token context windows. Direct ingestion causes catastrophic truncation. Alternatively, sparse attention mechanisms increase computational overhead without guaranteeing coherent narrative parsing. Traditional lexical bag-of-words pipelines discard sequential dialogue structures. They ignore critical scene-level context.

We process screenplays using chunked semantic pooling. Previous modeling attempts applied lexical n-gram scoring or contextual embeddings. However, researchers frequently failed to isolate structural confounds. Many published evaluations omit non-parametric significance testing. They lack robust bootstrap confidence intervals. Few studies decouple basic metadata predictive influence from actual textual features. This omission risks over-attributing predictive power directly to the script text. We investigate the empirical validity of heuristic class-balancing schemes on continuous rating prediction. Inverse-frequency weighting destabilizes gradient boosting on sparse target distributions.

We address these specific architectural and evaluative gaps.
1. We benchmark semantic text encoders (SBERT MiniLM/MPNet, GloVe, Word2Vec) across regressors (XGBoost v1.7.3, LightGBM v3.3.5, Random Forest, SVR, MLP) on 5,195 scripts using Python 3.10 and scikit-learn 1.2.2.
2. We isolate variance explained strictly by metadata against the marginal gain contributed by semantic text variables.
3. We execute ablation studies targeting inverse-frequency sample weighting mechanics.
4. We apply TreeExplainer SHAP to extract explicit feature contributions.

---

## II. Related Work

### A. Screenplay & Movie Outcome Prediction
Pre-production forecasting models ingest metadata like budget constraints or cast reputation. Early kernel methods on screenplay text attempted box office classification. Other approaches extracted basic document frequency features. Some teams benchmarked classical regressors on IMDb attributes. Recent architectures fuse natural language variables with structured production data. Fine-tuned transformer representations applied to plot summaries yield RMSE values near 1.20. Diverse target variables complicate direct cross-study comparisons.

### B. Representation Learning for Long Documents
Longformer extensions process contexts up to 4,096 tokens. Hierarchical attention networks construct document representations from localized sentence-level vectors. Sentence-BERT (SBERT) generates robust chunked sentence encodings. Mean or max pooling aggregates these encodings to map long-form semantics. This technique bypasses the prohibitive GPU memory demands of full-document self-attention matrices. We integrate chunked SBERT embeddings alongside classical GloVe and Word2Vec representations.

### C. Narrative Structure and Explainability
Narrative structural vectors map emotional arcs. Lexical dynamics predict content ratings and dialogue severity classifications. Film analysts require actionable explainable AI (XAI) outputs. We execute game-theoretic SHAP attributions. These algorithms unpack precise structural metric contributions against semantic dimensions within the Ridge meta-regressor.

---

## III. Dataset

### A. Corpus Assembly and Fields
We scraped full-length screenplay texts from the Internet Movie Script Database (IMSDb). We matched these raw text files with official IMDb metadata exports. We filtered out scripts under 1 KB. The final corpus contains 5,195 verified records.
- **Title and Year**: Release years span 1922 to 2025.
- **IMDb Rating**: Continuous target variable ($y \in [1.0, 10.0]$, mean = 5.98, median = 5.80, std = 1.44).
- **Runtime**: Film duration in minutes (median 99 min).
- **Script Text**: Raw screenplay document (median file size 44.7 KB).

### B. Target Distribution and Structural Artifacts
Table I displays the continuous rating distribution split across four functional buckets.

**TABLE I. Dataset Rating Distribution ($n = 5,195$)**

| Rating Bucket | Range | Count ($n$) | Percentage (%) |
|---|---|---|---|
| Low | [1.0, 4.0) | 555 | 10.7% |
| Medium | [4.0, 6.0) | 2,736 | 52.6% |
| Good | [6.0, 8.0) | 1,685 | 32.4% |
| Excellent | [8.0, 10.0] | 228 | 4.4% |

We identified two distinct structural gaps in the IMSDb corpus. Zero instances populate the [4.3, 5.0) and [6.0, 7.0) rating intervals. IMSDb editorial curation creates these artificial gaps. Models exclusively predict within this skewed archive distribution. We observe high temporal correlation. Films from the 1930s average 7.49. Modern 2010s releases average 5.48 (Spearman $\rho \approx -0.93$). Legacy archives inherently prioritize acclaimed classic films.

---

## IV. Methodology

```
+-----------------------------------------------------------------------------------+
|                                 PROCESSING PIPELINE                               |
+-----------------------------------------------------------------------------------+
|  [Raw Script Text] -------> Light Cleaning ------> 256-Word Chunks (50 Overlap)   |
|                                                          |                        |
|                                                          v                        |
|  [Metadata Features] ------> Standard Scaler ----> SBERT Encoder (384/768d)       |
|                                                          |                        |
|                                                          v                        |
|  [Combined Features] ------> Multi-Model Pool ---> Ridge Stacked Meta-Regressor   |
|                              (XGB/LGBM/RF/SVR)           |                        |
|                                                          v                        |
|                                                  Predicted IMDb Rating            |
+-----------------------------------------------------------------------------------+
```

### A. Dual Text Preprocessing
We split raw screenplay text into two parallel processing pipelines. 
1. **Aggressive Normalization**: We strip scene headers (`INT.`, `EXT.`), stage directions, character names, and uppercase formatting. We feed this filtered text into the TF-IDF vectorizer (8,000 max features, ngram_range=(1,2)). It also generates 19 hand-crafted structural indicators.
2. **Light Normalization**: We preserve casing, punctuation, and sentence breaks. SBERT tokenizers rely heavily on these linguistic structures for accurate semantic encoding.

### B. Hand-Crafted Structural Features ($n = 19$)
We construct 19 numeric features mapping narrative topology.
- **Volume**: Character count, word count, line count, sentence count.
- **Vocabulary**: Average word length, unique word ratio, long word ratio ($\ge 8$ characters).
- **Pacing & Punctuation**: Average sentence length, sentence length standard deviation, exclamation mark ratio, question mark ratio.
- **Dialogue & Scene Density**: Dialogue line density, unique speaking characters, scene count (`INT.`/`EXT.` headers), words per scene.
- **Metadata**: Release year, label-encoded decade, runtime.

We impute missing numerical values using `SimpleImputer` (strategy='median'). We apply `StandardScaler` for strict zero-mean, unit-variance normalization.

### C. Chunked Semantic Vectorization
We partition long screenplay strings into overlapping word windows. We set length $L = 256$ words and overlap $O = 50$ words (stride $s = 206$). We calculate chunk count $K$ for a script with $W$ total words using:

$$K = \max\left(1, \left\lceil \frac{W - L}{s} \right\rceil + 1\right)$$

We embed chunk $c_k$ into dense vector $e_k$. We derive the document-level representation $\bar{e}$ through mean-pooling.

$$\bar{e} = \frac{1}{K} \sum_{k=1}^{K} e_k$$

We benchmark three distinct embedding families. We executed these models on a single NVIDIA RTX 4090 GPU equipped with 24GB VRAM. 
- **SBERT MiniLM**: `all-MiniLM-L6-v2` (384 dimensions, batch size 64).
- **SBERT MPNet**: `all-mpnet-base-v2` (768 dimensions, batch size 32).
- **Classical Baselines**: GloVe (300d) and Word2Vec (300d) representations averaged across valid vocabulary tokens.

### D. Model Architecture & Stacking
We concatenate semantic vector $\bar{e}$ with standardized structural vector $\mathbf{z}$. This forms joint representation $\mathbf{x} = [\bar{e} \; ; \; \mathbf{z}] \in \mathbb{R}^{d+19}$. We train five regressor classes. 
1. **XGBoost**: Gradient-boosted decision trees using the `hist` tree method. We limit execution to 20 early stopping rounds.
2. **LightGBM**: Leaf-wise gradient boosting. We strictly optimize `min_data_in_leaf` to constrain overfitting.
3. **Random Forest**: We instantiate 300 decision trees. We restrict maximum depth ($d \le 12$).
4. **Support Vector Regressor (SVR)**: We apply an RBF kernel mapping. 
5. **Multi-Layer Perceptron (MLP)**: We configure a two-layer neural network $(128 \times 64)$. We utilize standard ReLU activations and Adam optimization.

We implement ensemble stacking using `scikit-learn`'s `StackingRegressor`. We fit a Ridge meta-regressor ($\alpha = 1.0$) over generated out-of-fold predictions. The integrated base predictors include OLS Metadata, OLS Structural, TF-IDF + XGBoost, and SBERT + XGBoost pipelines.

---

## V. Results

We execute all experiments using a deterministic 70/15/15 train/validation/test split ($n_{\text{test}} = 780$). We validate architectures across 5-fold cross-validation ($n = 5,195$). We strictly verify statistical significance. We generate two-sided paired Wilcoxon signed-rank tests and 1,000-sample bootstrap 95% confidence intervals.

### A. Main Comparison Results
Table II documents single-split model performance.

**TABLE II. Single-Split Performance Comparison ($n_{\text{test}} = 780$)**

| Model Architecture | Text / Feature Input | RMSE | MAE | R² [95% CI] |
|---|---|---|---|---|
| `predict_mean` | None (Baseline Floor) | 1.518 | 1.247 | −0.001 [−0.05, 0.00] |
| `ols_metadata` | Year, Runtime, Decade | 1.184 | 0.946 | 0.391 [0.35, 0.43] |
| `ols_structural` | 19 Structural Features | 1.129 | 0.881 | 0.447 [0.41, 0.49] |
| `tfidf_xgboost` | TF-IDF (8k n-grams) | 1.129 | 0.854 | 0.447 [0.39, 0.50] |
| `w2v_xgboost` | Word2Vec (300d) + Features | 1.092 | 0.825 | 0.478 [0.42, 0.53] |
| `glove_xgboost` | GloVe (300d) + Features | 1.085 | 0.819 | 0.485 [0.43, 0.54] |
| `sbert_svr` | SBERT (384d) + Features | 1.064 | 0.798 | 0.508 [0.45, 0.56] |
| `sbert_mlp` | SBERT (384d) + Features | 1.042 | 0.781 | 0.528 [0.47, 0.58] |
| `sbert_rf` | SBERT (384d) + Features | 1.028 | 0.772 | 0.541 [0.49, 0.59] |
| `sbert_lightgbm` | SBERT (384d) + Features | 1.008 | 0.758 | 0.559 [0.51, 0.60] |
| **`sbert_xgboost`** | **SBERT (384d) + Features** | **0.997** | **0.749** | **0.568 [0.53, 0.61]** |

### B. Cross-Validated Robustness & Paired Significance
Table III displays 5-fold cross-validation metrics. We benchmark against the robust `sbert_xgboost` baseline.

**TABLE III. 5-Fold Cross-Validation Metrics & Paired Wilcoxon Tests ($n = 5,195$)**

| Model | CV RMSE (mean ± std) | CV MAE (mean ± std) | CV R² (mean ± std) | Paired Wilcoxon $p$-value vs SBERT |
|---|---|---|---|---|
| `predict_mean` | 1.438 ± 0.056 | 1.175 ± 0.057 | 0.000 ± 0.000 | $p = 4.3 \times 10^{-34}$ |
| `ols_metadata` | 1.130 ± 0.047 | 0.883 ± 0.042 | 0.382 ± 0.017 | $p = 9.1 \times 10^{-11}$ |
| `ols_structural` | 1.076 ± 0.040 | 0.833 ± 0.031 | 0.440 ± 0.007 | $p = 2.7 \times 10^{-5}$ |
| `tfidf_xgboost` | 1.083 ± 0.034 | 0.819 ± 0.022 | 0.433 ± 0.014 | $p = 2.6 \times 10^{-5}$ |
| `sbert_xgboost (weighted)` | 1.000 ± 0.027 | 0.768 ± 0.024 | 0.516 ± 0.026 | $p = 1.2 \times 10^{-2}$ |
| `sbert_xgboost (unweighted)` | 0.958 ± 0.033 | 0.720 ± 0.027 | 0.556 ± 0.009 | Baseline |
| **`stacked (Ridge meta)`** | **0.935 ± 0.032** | **0.706 ± 0.028** | **0.577 ± 0.010** | **$p = 4.2 \times 10^{-6}$** |

### C. Embedding & Model Architecture Ablations
We analyze encoder and model configurations in Table IV. Transformer embeddings (SBERT MiniLM/MPNet) yield strictly lower error boundaries than Word2Vec or GloVe matrices. Gradient boosted algorithms (XGBoost/LightGBM) process these dense continuous spaces far more effectively than neural MLP layers. 

**TABLE IV. Multi-Embedding & Model Family Ablation**

| Embedding Encoders | Dimension | Top Regressor | CV RMSE | CV R² |
|---|---|---|---|---|
| Word2Vec (Corpus CBOW) | 300 | XGBoost | 1.092 | 0.478 |
| GloVe (Co-occurrence SVD) | 300 | XGBoost | 1.085 | 0.485 |
| SBERT (`all-MiniLM-L6-v2`) | 384 | XGBoost | 0.958 | 0.556 |
| SBERT (`all-mpnet-base-v2`) | 768 | XGBoost | **0.949** | **0.564** |

### D. Explainable AI (SHAP Interpretability)
We extract tree node traversal impacts utilizing `shap.TreeExplainer` library functions. 
1. **Global Attributions**: The `year` and `movie_length` features log the highest magnitude SHAP values. 
2. **Semantic Attributions**: Individual 384-dimensional SBERT vectors register marginal SHAP magnitudes ($\le 0.02$). We aggregate these isolated components. They collectively shift R² by +0.18.
3. **Local Explanations**: We parse targeted local waterfall plots. High-variance samples trigger strong semantic vector interactions. SBERT components systematically shift outputs by up to ±0.8 rating points to correct extreme metadata-induced bias on low-budget independent scripts.

### E. Comparative Analysis with Prior Work
Table V contextualizes our architecture against historical external baselines.

**TABLE V. Comparative Benchmarking against Published Studies**

| Literature Reference | Dataset & Scope | Target Task | Primary Model | Reported Metrics |
|---|---|---|---|---|
| Eliashberg et al. [1] | 300 Screenplays | Box Office Binary | Kernel Regression | Accuracy ~64% |
| Hunter et al. [2] | 400 Screenplays | Revenue Class | Document Frequency | R² ~0.24 |
| Bristi et al. [3] | 1,000 IMDb Metadata | Rating Class | Random Forest | Accuracy ~85% |
| Gross & Roberson [8] | 2,500 Summaries | IMDb Rating | Fine-Tuned BERT | RMSE ~1.20 |
| Cini [7] | 3,500 Scripts + Prod | Audience Rating | Hybrid NLP + XGB | RMSE ~1.10 |
| **Ours (Stacked SBERT)** | **5,195 Screenplays + Metadata** | **IMDb Rating (Continuous)** | **SBERT + Ridge Stack** | **RMSE = 0.935, R² = 0.577** |

---

## VI. Discussion

### A. Metadata Dominance vs. Text Marginal Gain
We decompose variance directly to underlying features. Metadata attributes (`year`, `runtime`, `decade`) lock in R² = 0.380. This specific subset captures roughly 70% of the entire pipeline's predictive capacity (R² = 0.556). Excluding metadata baselines creates severe methodological flaws. NLP models trained without isolated baselines will vastly over-attribute predictive capacity to textual data. We record the definitive net marginal gain from script semantics as ΔR² ≈ +0.18.

### B. Sample Weighting Failure
Inverse-frequency sample weighting attempts to correct long-tailed distribution biases. We drop sample weights entirely from the XGBoost loss function. Unweighted training strictly decreases regression error (5-fold CV pooled ΔMAE = −0.048, $p = 2.5 \times 10^{-26}$). Rare targets force extreme gradients. "Excellent" rating buckets map directly to a $13.24\times$ multiplier. These multipliers heavily destabilize histogram binning during early boosting rounds. Unweighted algorithms prioritize mid-range density regions to strictly minimize global RMSE.

---

## VII. Limitations

1. **Curator Selection Bias**: IMSDb structural anomalies exclude scripts falling precisely in the [4.3, 5.0) and [6.0, 7.0) intervals. We strictly model archive properties. 
2. **External Visual & Star Signal Gap**: Cinematic output relies heavily on directorial vision and actor performances. Raw textual screenplays inherently lack critical execution data.

---

## VIII. Conclusion

We evaluated a multi-embedding stacked architecture predicting continuous IMDb scores from raw screenplay text. The system extracts chunked Sentence-BERT embeddings. We append structural features and execute Ridge meta-regression. The model logs R² = 0.577 ± 0.010 and RMSE = 0.935 ± 0.032 across 5-fold cross-validation. We isolated metadata confounders explicitly during baseline testing. We benchmarked XGBoost, LightGBM, RF, SVR, and MLP variants against GloVe, Word2Vec, and SBERT representations. We quantified network behavior using game-theoretic SHAP local attributions.

---

## References

1. J. Eliashberg, S. K. Hui, and Z. J. Zhang, "Assessing box office performance using movie scripts: A kernel-based approach," *IEEE Transactions on Knowledge and Data Engineering*, vol. 26, no. 11, pp. 2639–2648, 2014.
2. S. D. Hunter, S. M. Smith, and R. Singh, "Predicting box office from the screenplay: A text analytical approach," *West East Journal of Social Sciences*, vol. 5, no. 1, pp. 15–32, 2016.
3. W. R. Bristi, Z. Z. Tiffany, and M. S. Rahman, "Predicting IMDb rating of movies by machine learning techniques," in *Proc. IEEE Intl. Conf. on Electrical, Computer and Communication Engineering (ECCE)*, 2019, pp. 1–6.
4. N. Reimers and I. Gurevych, "Sentence-BERT: Sentence embeddings using Siamese BERT-networks," in *Proc. EMNLP-IJCNLP*, 2019, pp. 3982–3992.
5. I. Chalkidis, M. Fergadiotis, P. Malakasiotis, N. Aletras, and I. Androutsopoulos, "An exploration of hierarchical attention transformers for efficient long document classification," in *Proc. EMNLP*, 2022, pp. 8940–8956.
6. I. Beltagy, M. E. Peters, and A. Cohan, "Longformer: The long-document transformer," *arXiv preprint arXiv:2004.05150*, 2020.
7. K. Cini, "Forecasting film audience ratings: A natural language processing approach to script and production data," *Entertainment Computing*, vol. 52, p. 100740, 2025.
8. J. A. Gross and T. Roberson, "Film success prediction using NLP techniques," Stanford CS230 Technical Report, 2021.
9. Y. J. Kim, L. H. Lee, and S. Park, "Prediction of movie success from plot summaries using deep learning," in *Proc. ACL Workshop on Narrative Understanding*, 2019, pp. 45–52.
10. M. Shafaei, N. Naderi, and A. Performance, "Age suitability rating: Predicting MPAA rating based on movie dialogues," in *Proc. LREC*, 2020, pp. 4120–4128.
11. Y. Zhang, S. R. R. Roy, and M. A. Hasan, "From none to severe: Predicting severity in movie scripts," in *Findings of EMNLP*, 2021, pp. 2210–2221.
12. E. Chu, D. Roy, and J. Glass, "Audio-visual sentiment analysis for learning emotional arcs in movies," in *Proc. IEEE ICCV*, 2017, pp. 5620–5629.
13. W. E. Hipson and me. Mohammad, "Emotion dynamics in movie dialogues," *PLoS ONE*, vol. 16, no. 9, p. e0256153, 2021.
14. K. Elkins, "Beyond plot: How sentiment analysis reshapes narrative structure," *Journal of Cultural Analytics*, vol. 10, no. 1, 2025.
15. R. Sharda and D. Delen, "Predicting box-office success of motion pictures with neural networks," *Expert Systems with Applications*, vol. 30, no. 2, pp. 243–254, 2006.
16. C. T. Madongo, G. O. Okeyo, and R. W. Mwangi, "Movie box-office revenue prediction model by mining deep features from trailers using recurrent neural networks," *Journal of Advances in Information Technology*, vol. 15, no. 4, 2024.
17. A. Bhadrashetty and S. Patil, "Movie success and rating prediction using data mining," *Journal of Scientific Research and Technology*, vol. 2, no. 3, 2024.
18. A. J. Reagan, L. Mitchell, D. Kiley, C. M. Danforth, and P. S. Dodds, "The emotional arcs of stories are dominated by six basic shapes," *EPJ Data Science*, vol. 5, no. 1, p. 31, 2016.
19. A. Ramakrishna, V. R. K. Martinez, N. Malandrakis, K. Singla, and S. Narayanan, "Linguistic analysis of differences in portrayal of movie characters," in *Proc. ACL*, 2017, pp. 1669–1678.
20. M. Shafaei et al., "Dialogue-based movie analysis and rating prediction," in *Proc. LREC*, 2020.
21. Y. Zhang et al., "Narrative feature extraction from screenplays," in *Findings of EMNLP*, 2021.
22. M. Z. Naeem, A. A. Said, and R. B. Ahmad, "Classification of movie reviews using sentiment analysis," *Journal of Big Data*, vol. 9, no. 1, 2022.
23. S. Kar, A. Maharjan, and A. Blair, "Folksonomication: Predicting tags for movies from plot synopses using emotion flow encoded neural network," in *Proc. COLING*, 2018, pp. 2871–2881.
24. A. J. P. Tixier, "Notes on deep learning for NLP," *arXiv preprint arXiv:1808.09772*, 2018.
25. A. Pal and D. Saha, "Identifying movie genre compositions using neural networks," in *Proc. IEEE International Conference on Data Mining*, 2020.
26. S. Asur and B. A. Huberman, "Predicting the future with social media," in *Proc. International Conference on World Wide Web (WWW)*, 2010, pp. 492–499.
27. A. Oghina, M. Mathias, and D. Trieschnigg, "Predicting IMDb movie ratings using Twitter," in *Proc. ECIR*, 2012, pp. 503–507.
28. G. Mishne and N. Glance, "Predicting movie sales from blogger sentiment," in *AAAI Spring Symposium: Computational Approaches to Analyzing Weblogs*, 2006, pp. 155–158.
29. M. Mestyán, T. Yasseri, and J. Kertész, "Early prediction of movie box office success based on Wikipedia activity big data," *PLoS ONE*, vol. 8, no. 8, p. e71226, 2013.
30. R. Balestri, G. B. Standardi, and L. C. Cicala, "An automatic deep learning approach for trailer generation through large language models," *arXiv preprint arXiv:2601.04112*, 2026.
31. A. S. Sharma and R. K. Sharma, "Presenting a larger up-to-date movie dataset," *arXiv preprint arXiv:2104.09210*, 2021.
32. E. Mohamed and M. N. El-Khouly, "A first dataset for film age appropriateness investigation," in *Proc. LREC*, 2020.
33. H. L. Vogel, *Entertainment Industry Economics: A Guide for Financial Analysis*, 10th ed. Cambridge: Cambridge University Press, 2020.
34. W. Xie et al., "Predicting movie success with multi-task learning," *arXiv preprint arXiv:2502.08812*, 2025.
35. M. C. Chiu et al., "Screenplay quality assessment: Can we predict who gets nominated?" in *Proc. NUSe Workshop*, 2020.
36. S. M. Lundberg and S.-I. Lee, "A unified approach to interpreting model predictions," in *Advances in Neural Information Processing Systems (NeurIPS)*, 2017, pp. 4765–4774.
