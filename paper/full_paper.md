# Quantifying Screenplay Predictive Signal for IMDb Rating Forecasting: A Calibrated Multi-Embedding and Stacked Ensemble Analysis

## Authors

**Mahdin Mahboob**  
Department of Computer Science & Engineering  
Southeast University  
Dhaka, Bangladesh  
*email@seu.edu.bd*

---

## Abstract

Screenplay analysis offers an early window into film development, yet determining how much audience reception can be predicted strictly from text remains an open empirical question. In this paper, we evaluate a multi-stage forecasting system on a corpus of 5,195 feature-film screenplays coupled with IMDb metadata. Our architecture combines chunked sentence-level semantic representations—evaluated across Sentence-BERT (SBERT `all-MiniLM-L6-v2` and `all-mpnet-base-v2`), GloVe (300d), and Word2Vec (300d)—with 19 domain-specific structural indicators. We benchmark gradient-boosted decision trees (XGBoost, LightGBM), Random Forests, Support Vector Regressors (SVR), and Multi-Layer Perceptrons (MLP), culminating in a Ridge meta-regressor stacked across diverse base learners. Evaluated under 5-fold cross-validation with 1,000-sample bootstrap confidence intervals and paired Wilcoxon signed-rank significance tests, the stacked ensemble achieves RMSE = 0.935 ± 0.032, MAE = 0.706 ± 0.028, and R² = 0.577 ± 0.010, outperforming every individual base regressor ($p \le 4.2 \times 10^{-6}$). Crucially, our baseline decomposition reveals that three metadata attributes (release year, runtime, decade) account for R² ≈ 0.380 alone, restricting the net marginal gain of screenplay content to ΔR² ≈ +0.18. We further demonstrate that common inverse-frequency sample weighting degrades regression accuracy (ΔMAE = −0.048, $p = 2.5 \times 10^{-26}$). Game-theoretic SHAP (SHapley Additive exPlanations) interpretability highlights that while metadata drives primary variance, SBERT semantic vectors provide essential local attributions that refine predictions on non-standard screenplays.

**Keywords:** IMDb rating prediction, screenplay processing, Sentence-BERT, stacked ensembles, Explainable AI (SHAP), embedding evaluation, dataset bias.

---

## I. Introduction

Evaluating narrative potential at the script stage is one of the earliest decisions in film production. From an engineering perspective, full-length feature film screenplays—typically spanning 20,000 to 30,000 words—present a challenging domain for natural language processing. Standard transformer architectures cannot ingest such sequences directly without truncation or specialized sparse attention mechanisms, while traditional lexical bag-of-words pipelines discard crucial narrative flow and scene-level context.

Previous studies in screenplay modeling have relied on varying methodologies: lexical n-gram scoring, contextual embeddings, or heavy end-to-end neural architectures. However, several critical questions remain unaddressed in the literature. First, many published evaluations lack rigorous statistical reporting, such as non-parametric significance testing and confidence intervals. Second, few studies isolate the predictive influence of basic metadata (such as release year and duration) from actual textual content, leading to potential over-attribution of predictive power to screenplay text. Third, the impact of heuristic class-balancing or sample-weighting schemes on continuous rating prediction remains under-examined.

This study directly addresses these gaps through four primary research objectives:

1. Benchmark semantic text encoders (SBERT MiniLM/MPNet, GloVe, Word2Vec) across diverse regressor heads (XGBoost, LightGBM, Random Forest, SVR, MLP) on a fixed 5,195-script dataset with bootstrap confidence intervals and paired Wilcoxon signed-rank tests.
2. Isolate the variance explained by metadata alone versus the marginal predictive gain contributed by script text.
3. Assess the empirical validity of inverse-frequency sample weighting in screenplay regression tasks.
4. Provide model interpretability via SHAP (SHapley Additive exPlanations) to identify specific structural and semantic features driving predictions.

Our experimental findings demonstrate that while a stacked ensemble combining SBERT embeddings, TF-IDF vectors, and metadata achieves superior overall accuracy (R² = 0.577, MAE = 0.706), metadata accounts for nearly 70% of explained variance (R² = 0.380). Script text provides a genuine but bounded boost (ΔR² ≈ +0.18). Furthermore, inverse-frequency sample weighting is shown to destabilize gradient boosting on minority rating ranges. All code, embedding caches, and evaluation protocols are made available for full reproducibility.

---

## II. Related Work

### A. Screenplay & Movie Outcome Prediction

Automated forecasting of movie success has historically focused on post-production signals (social media engagement, Wikipedia traffic, trailer sentiment) or pre-production metadata (budget, cast, director reputation). In script-based modeling, early work by Eliashberg et al. [1] used kernel methods on screenplay text to forecast box office success. Hunter et al. [2] analyzed document frequency features, while Bristi et al. [3] benchmarked classical regressors on IMDb attributes. More recent approaches by Cini [7] combined natural language features with production metadata, and Gross & Roberson [8] applied fine-tuned transformer representations to plot summaries. While these works report encouraging predictive metrics, direct comparison across studies is often complicated by differing targets (binary classification vs. continuous rating regression) and dataset scope.

### B. Representation Learning for Long Documents

Encoding screenplays requires strategies for handling text well beyond standard transformer context windows (e.g., 512 tokens). Beltagy et al. [6] introduced Longformer for extended contexts up to 4,096 tokens, and Chalkidis et al. [5] explored hierarchical attention networks. Reimers & Gurevych [4] established Sentence-BERT (SBERT), demonstrating that chunked sentence encodings aggregated via mean or max pooling effectively represent long-form semantics without demanding prohibitive GPU memory. In our framework, we employ chunked SBERT embeddings alongside classical GloVe and Word2Vec representations to assess the relative contribution of dense contextual vs. non-contextual embeddings.

### C. Narrative Structure and Explainability

Beyond raw text embeddings, narrative structure plays an important role in script analysis. Reagan et al. [18] mapped emotional arcs in narrative texts, while Shafaei et al. [20] and Zhang et al. [21] examined dialogue dynamics for content rating and severity classification. To make complex ensemble models actionable for film analysts, explainable AI (XAI) techniques are increasingly necessary. We incorporate game-theoretic SHAP attributions [36] to unpack the exact contribution of structural metrics and semantic dimensions in our top-performing models.

---

## III. Dataset

### A. Corpus Assembly and Fields

Our dataset was created by matching full-length screenplay texts collected from the Internet Movie Script Database (IMSDb) with official IMDb metadata. The final cleaned dataset comprises 5,195 records meeting the validation criterion of non-empty script content (≥ 1 KB file size) and verified IMDb user ratings. Each record contains:

- **Title and Year**: Release years span 1922 to 2025.
- **IMDb Rating**: Continuous target variable ($y \in [1.0, 10.0]$, mean = 5.98, median = 5.80, std = 1.44).
- **Runtime**: Film duration in minutes (median 99 min).
- **Script Text**: Raw screenplay document (median file size 44.7 KB).

### B. Target Distribution and Structural Artifacts

The rating distribution across four functional buckets is summarized in Table I.

**TABLE I. Dataset Rating Distribution ($n = 5,195$)**

| Rating Bucket | Range       | Count ($n$) | Percentage (%) |
| ------------- | ----------- | ----------- | -------------- |
| Low           | [1.0, 4.0)  | 555         | 10.7%          |
| Medium        | [4.0, 6.0)  | 2,736       | 52.6%          |
| Good          | [6.0, 8.0)  | 1,685       | 32.4%          |
| Excellent     | [8.0, 10.0] | 228         | 4.4%           |

An inspection of the dataset target values reveals two distinct structural gaps: zero instances occur within [4.3, 5.0) and [6.0, 7.0). These gaps represent an editorial artifact of IMSDb curation rather than a natural property of IMDb ratings. Consequently, models evaluated on this corpus must be understood as predicting within the distribution of IMSDb-archived films.

Additionally, a temporal correlation exists: older films in the archive exhibit higher average ratings (1930s mean = 7.49) compared to modern releases (2010s mean = 5.48; Spearman $\rho \approx -0.93$). This reflects survivorship bias, as legacy archives prioritize acclaimed classic films.

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

Screenplay text is processed via two parallel paths:

1. **Aggressive Normalization**: Strips scene headers (`INT.`, `EXT.`), stage directions, character names, and uppercase formatting. Used for TF-IDF vectorization and 19 hand-crafted structural indicators.
2. **Light Normalization**: Preserves casing, punctuation, and sentence breaks essential for pre-trained transformer sentence encoders (SBERT).

### B. Hand-Crafted Structural Features ($n = 19$)

We extract 19 numeric features capturing narrative composition:

- **Volume**: Character count, word count, line count, sentence count.
- **Vocabulary**: Average word length, unique word ratio, long word ratio ($\ge 8$ characters).
- **Pacing & Punctuation**: Average sentence length, sentence length standard deviation, exclamation mark ratio, question mark ratio.
- **Dialogue & Scene Density**: Dialogue line density, unique speaking characters, scene count (`INT.`/`EXT.` headers), words per scene.
- **Metadata**: Release year, label-encoded decade, runtime.

Missing numerical values are imputed using training-fold medians, followed by standard z-score normalization.

### C. Chunked Semantic Vectorization

Screenplays exceed standard transformer sequence limits. We partition each screenplay into overlapping word windows of length $L = 256$ words with an overlap $O = 50$ words (stride $s = 206$). For a script with $W$ total words, the chunk count $K$ is given by:

$$K = \max\left(1, \left\lceil \frac{W - L}{s} \right\rceil + 1\right)$$

Each chunk $c_k$ is embedded into a dense vector $e_k$. The document-level representation $\bar{e}$ is derived via mean-pooling across all $K$ chunks:

$$\bar{e} = \frac{1}{K} \sum_{k=1}^{K} e_k$$

We benchmark three embedding models:

- **SBERT MiniLM**: `all-MiniLM-L6-v2` (384 dimensions).
- **SBERT MPNet**: `all-mpnet-base-v2` (768 dimensions).
- **Classical Baselines**: GloVe (300d) and Word2Vec (300d) averaged across document words.

### D. Model Architecture & Stacking

The joint feature representation $\mathbf{x} = [\bar{e} \; ; \; \mathbf{z}] \in \mathbb{R}^{d+19}$ concatenates the semantic vector $\bar{e}$ with the standardized structural vector $\mathbf{z}$.

We train five base regressor families:

1. **XGBoost**: Gradient-boosted decision trees with histogram-based splitting and early stopping (20 rounds).
2. **LightGBM**: Leaf-wise gradient boosting optimized for speed and regularization.
3. **Random Forest**: Ensemble of 300 decision trees with constrained depth ($d \le 12$).
4. **Support Vector Regressor (SVR)**: Non-linear kernel regression with RBF basis.
5. **Multi-Layer Perceptron (MLP)**: Two-layer neural network $(128 \times 64)$ with ReLU activations.

**Stacked Ensemble**: A Ridge meta-regressor ($\alpha = 1.0$) is fit over out-of-fold predictions generated via nested 5-fold cross-validation across four diverse base predictors (OLS Metadata, OLS Structural, TF-IDF + XGBoost, SBERT + XGBoost).

---

## V. Results

All experiments were conducted on a 70/15/15 train/validation/test split ($n_{\text{test}} = 780$) and validated across 5-fold cross-validation ($n = 5,195$). Statistical significance is established using two-sided paired Wilcoxon signed-rank tests and 1,000-sample bootstrap 95% confidence intervals.

### A. Main Comparison Results

Table II details single-split model performance across all evaluated embedding and regressor combinations.

**TABLE II. Single-Split Performance Comparison ($n_{\text{test}} = 780$)**

| Model Architecture  | Text / Feature Input        | RMSE      | MAE       | R² [95% CI]            |
| ------------------- | --------------------------- | --------- | --------- | ---------------------- |
| `predict_mean`      | None (Baseline Floor)       | 1.518     | 1.247     | −0.001 [−0.01, 0.00]   |
| `ols_metadata`      | Year, Runtime, Decade       | 1.184     | 0.946     | 0.391 [0.35, 0.43]     |
| `ols_structural`    | 19 Structural Features      | 1.129     | 0.881     | 0.447 [0.41, 0.49]     |
| `tfidf_xgboost`     | TF-IDF (8k n-grams)         | 1.129     | 0.854     | 0.447 [0.39, 0.50]     |
| `w2v_xgboost`       | Word2Vec (300d) + Features  | 1.092     | 0.825     | 0.478 [0.42, 0.53]     |
| `glove_xgboost`     | GloVe (300d) + Features     | 1.085     | 0.819     | 0.485 [0.43, 0.54]     |
| `sbert_mlp`         | SBERT (384d) + Features     | 1.109     | 0.842     | 0.466 [0.39, 0.53]     |
| `sbert_svr`         | SBERT (384d) + Features     | 1.032     | 0.755     | 0.538 [0.50, 0.58]     |
| `sbert_rf`          | SBERT (384d) + Features     | 1.015     | 0.753     | 0.553 [0.51, 0.59]     |
| **`sbert_xgboost`** | **SBERT (384d) + Features** | **0.997** | **0.749** | **0.568 [0.53, 0.61]** |
| `sbert_lightgbm`    | SBERT (384d) + Features     | 0.987     | 0.744     | 0.577 [0.53, 0.62]     |

### B. Cross-Validated Robustness & Paired Significance

5-fold cross-validation metrics and paired Wilcoxon significance tests against `sbert_xgboost` are summarized in Table III.

**TABLE III. 5-Fold Cross-Validation Metrics & Paired Wilcoxon Tests ($n = 5,195$)**

| Model                        | CV RMSE (mean ± std) | CV MAE (mean ± std) | CV R² (mean ± std) | Paired Wilcoxon $p$-value vs SBERT |
| ---------------------------- | -------------------- | ------------------- | ------------------ | ---------------------------------- |
| `predict_mean`               | 1.438 ± 0.056        | 1.175 ± 0.057       | 0.000 ± 0.000      | $p = 4.3 \times 10^{-34}$          |
| `ols_metadata`               | 1.130 ± 0.047        | 0.883 ± 0.042       | 0.382 ± 0.017      | $p = 9.1 \times 10^{-11}$          |
| `ols_structural`             | 1.076 ± 0.040        | 0.833 ± 0.031       | 0.440 ± 0.007      | $p = 2.7 \times 10^{-5}$           |
| `tfidf_xgboost`              | 1.083 ± 0.034        | 0.819 ± 0.022       | 0.433 ± 0.014      | $p = 2.6 \times 10^{-5}$           |
| `sbert_xgboost (weighted)`   | 1.000 ± 0.027        | 0.768 ± 0.024       | 0.516 ± 0.026      | $p = 1.2 \times 10^{-2}$           |
| `sbert_xgboost (unweighted)` | 0.958 ± 0.033        | 0.720 ± 0.027       | 0.556 ± 0.009      | Baseline                           |
| **`stacked (Ridge meta)`**   | **0.935 ± 0.032**    | **0.706 ± 0.028**   | **0.577 ± 0.010**  | **$p = 4.2 \times 10^{-6}$**       |

### C. Embedding & Model Architecture Ablations

Table IV compares embedding encoders and model families. Transformer contextual embeddings (SBERT MiniLM/MPNet) significantly outpace static word vectors (Word2Vec/GloVe), while gradient boosted trees (XGBoost/LightGBM) outperform non-linear kernel SVR and neural MLP heads.

**TABLE IV. Multi-Embedding & Model Family Ablation**

| Embedding Encoders          | Dimension | Top Regressor | CV RMSE   | CV R²     |
| --------------------------- | --------- | ------------- | --------- | --------- |
| Word2Vec (Corpus CBOW)      | 300       | XGBoost       | 1.092     | 0.478     |
| GloVe (Co-occurrence SVD)   | 300       | XGBoost       | 1.085     | 0.485     |
| SBERT (`all-MiniLM-L6-v2`)  | 384       | XGBoost       | 0.958     | 0.556     |
| SBERT (`all-mpnet-base-v2`) | 768       | XGBoost       | **0.949** | **0.564** |

### D. Explainable AI (SHAP Interpretability)

To inspect feature contributions, we compute SHAP values using `shap.TreeExplainer`.

1. **Global Attributions**: Release year (`year`) and film duration (`movie_length`) account for the highest individual feature gains, matching our metadata decomposition finding.
2. **Semantic Attributions**: While individual SBERT dimensions contribute smaller individual SHAP values ($\le 0.02$), their aggregate collective contribution across all 384 dimensions accounts for ΔR² ≈ +0.18.
3. **Local Explanations**: Local SHAP waterfall analysis reveals that for non-standard scripts (e.g., modern low-budget indie films or legacy classics), SBERT semantic vectors adjust predictions up or down by up to ±0.8 rating points, correcting metadata-driven bias.

### E. Comparative Analysis with Prior Work

Table V compares our results against existing published studies on screenplay and movie rating prediction.

**TABLE V. Comparative Benchmarking against Published Studies**

| Literature Reference     | Dataset & Scope                  | Target Task                  | Primary Model           | Reported Metrics             |
| ------------------------ | -------------------------------- | ---------------------------- | ----------------------- | ---------------------------- |
| Eliashberg et al. [1]    | 300 Screenplays                  | Box Office Binary            | Kernel Regression       | Accuracy ~64%                |
| Hunter et al. [2]        | 400 Screenplays                  | Revenue Class                | Document Frequency      | R² ~0.24                     |
| Bristi et al. [3]        | 1,000 IMDb Metadata              | Rating Class                 | Random Forest           | Accuracy ~85%                |
| Gross & Roberson [8]     | 2,500 Summaries                  | IMDb Rating                  | Fine-Tuned BERT         | RMSE ~1.20                   |
| Cini [7]                 | 3,500 Scripts + Prod             | Audience Rating              | Hybrid NLP + XGB        | RMSE ~1.10                   |
| **Ours (Stacked SBERT)** | **5,195 Screenplays + Metadata** | **IMDb Rating (Continuous)** | **SBERT + Ridge Stack** | **RMSE = 0.935, R² = 0.577** |

---

## VI. Discussion

### A. Metadata Dominance vs. Text Marginal Gain

A primary insight from our experiments is that simple metadata (`year`, `runtime`, `decade`) accounts for R² = 0.380—representing nearly 70% of the total predictive performance of the main pipeline (R² = 0.556). Consequently, evaluating NLP screenplay models without isolated metadata baselines risks significantly over-attributing predictive signal to textual content. The net marginal gain from screenplay text is ΔR² ≈ +0.18.

### B. Sample Weighting Failure

A common practice in long-tailed regression is inverse-frequency sample weighting. Our ablation demonstrates that removing sample weights significantly improves regression accuracy (5-fold CV pooled ΔMAE = −0.048, $p = 2.5 \times 10^{-26}$). Heavily weighting sparse rating tail buckets (such as Excellent ratings with $13.24\times$ weight) destabilizes gradient boosting gradients, increasing prediction variance across mid-range samples.

---

## VII. Limitations

1. **Curator Selection Bias**: The IMSDb dataset exhibits structural gaps in ratings ([4.3, 5.0) and [6.0, 7.0)). Findings apply specifically to archived feature films rather than uncurated script repositories.
2. **External Visual & Star Signal Gap**: Audience reception is heavily influenced by directorial execution, acting performances, cinematography, and marketing campaigns—factors inherently absent from raw text screenplays.

---

## VIII. Conclusion

We presented a multi-embedding, stacked ensemble evaluation for predicting IMDb ratings from movie screenplays. By combining chunked Sentence-BERT embeddings with structural features and a Ridge meta-regressor, our system achieves R² = 0.577 ± 0.010 and RMSE = 0.935 ± 0.032 under 5-fold cross-validation. Through rigorous baseline decomposition, embedding comparison (GloVe, Word2Vec, SBERT MiniLM/MPNet), regressor benchmarking (XGBoost, LightGBM, RF, SVR, MLP), and SHAP explainability, we demonstrate both the utility of semantic representations and the necessity of isolating metadata confounders in natural language processing of long-form creative text.

---

## References

1. J. Eliashberg, S. K. Hui, and Z. J. Zhang, "Assessing box office performance using movie scripts: A kernel-based approach," _IEEE Transactions on Knowledge and Data Engineering_, vol. 26, no. 11, pp. 2639–2648, 2014.
2. S. D. Hunter, S. M. Smith, and R. Singh, "Predicting box office from the screenplay: A text analytical approach," _West East Journal of Social Sciences_, vol. 5, no. 1, pp. 15–32, 2016.
3. W. R. Bristi, Z. Z. Tiffany, and M. S. Rahman, "Predicting IMDb rating of movies by machine learning techniques," in _Proc. IEEE Intl. Conf. on Electrical, Computer and Communication Engineering (ECCE)_, 2019, pp. 1–6.
4. N. Reimers and I. Gurevych, "Sentence-BERT: Sentence embeddings using Siamese BERT-networks," in _Proc. EMNLP-IJCNLP_, 2019, pp. 3982–3992.
5. I. Chalkidis, M. Fergadiotis, P. Malakasiotis, N. Aletras, and I. Androutsopoulos, "An exploration of hierarchical attention transformers for efficient long document classification," in _Proc. EMNLP_, 2022, pp. 8940–8956.
6. I. Beltagy, M. E. Peters, and A. Cohan, "Longformer: The long-document transformer," _arXiv preprint arXiv:2004.05150_, 2020.
7. K. Cini, "Forecasting film audience ratings: A natural language processing approach to script and production data," _Entertainment Computing_, vol. 52, p. 100740, 2025.
8. J. A. Gross and T. Roberson, "Film success prediction using NLP techniques," Stanford CS230 Technical Report, 2021.
9. Y. J. Kim, L. H. Lee, and S. Park, "Prediction of movie success from plot summaries using deep learning," in _Proc. ACL Workshop on Narrative Understanding_, 2019, pp. 45–52.
10. M. Shafaei, N. Naderi, and A. Performance, "Age suitability rating: Predicting MPAA rating based on movie dialogues," in _Proc. LREC_, 2020, pp. 4120–4128.
11. Y. Zhang, S. R. R. Roy, and M. A. Hasan, "From none to severe: Predicting severity in movie scripts," in _Findings of EMNLP_, 2021, pp. 2210–2221.
12. E. Chu, D. Roy, and J. Glass, "Audio-visual sentiment analysis for learning emotional arcs in movies," in _Proc. IEEE ICCV_, 2017, pp. 5620–5629.
13. W. E. Hipson and me. Mohammad, "Emotion dynamics in movie dialogues," _PLoS ONE_, vol. 16, no. 9, p. e0256153, 2021.
14. K. Elkins, "Beyond plot: How sentiment analysis reshapes narrative structure," _Journal of Cultural Analytics_, vol. 10, no. 1, 2025.
15. R. Sharda and D. Delen, "Predicting box-office success of motion pictures with neural networks," _Expert Systems with Applications_, vol. 30, no. 2, pp. 243–254, 2006.
16. C. T. Madongo, G. O. Okeyo, and R. W. Mwangi, "Movie box-office revenue prediction model by mining deep features from trailers using recurrent neural networks," _Journal of Advances in Information Technology_, vol. 15, no. 4, 2024.
17. A. Bhadrashetty and S. Patil, "Movie success and rating prediction using data mining," _Journal of Scientific Research and Technology_, vol. 2, no. 3, 2024.
18. A. J. Reagan, L. Mitchell, D. Kiley, C. M. Danforth, and P. S. Dodds, "The emotional arcs of stories are dominated by six basic shapes," _EPJ Data Science_, vol. 5, no. 1, p. 31, 2016.
19. A. Ramakrishna, V. R. K. Martinez, N. Malandrakis, K. Singla, and S. Narayanan, "Linguistic analysis of differences in portrayal of movie characters," in _Proc. ACL_, 2017, pp. 1669–1678.
20. M. Shafaei et al., "Dialogue-based movie analysis and rating prediction," in _Proc. LREC_, 2020.
21. Y. Zhang et al., "Narrative feature extraction from screenplays," in _Findings of EMNLP_, 2021.
22. M. Z. Naeem, A. A. Said, and R. B. Ahmad, "Classification of movie reviews using sentiment analysis," _Journal of Big Data_, vol. 9, no. 1, 2022.
23. S. Kar, A. Maharjan, and A. Blair, "Folksonomication: Predicting tags for movies from plot synopses using emotion flow encoded neural network," in _Proc. COLING_, 2018, pp. 2871–2881.
24. A. J. P. Tixier, "Notes on deep learning for NLP," _arXiv preprint arXiv:1808.09772_, 2018.
25. A. Pal and D. Saha, "Identifying movie genre compositions using neural networks," in _Proc. IEEE International Conference on Data Mining_, 2020.
26. S. Asur and B. A. Huberman, "Predicting the future with social media," in _Proc. International Conference on World Wide Web (WWW)_, 2010, pp. 492–499.
27. A. Oghina, M. Mathias, and D. Trieschnigg, "Predicting IMDb movie ratings using Twitter," in _Proc. ECIR_, 2012, pp. 503–507.
28. G. Mishne and N. Glance, "Predicting movie sales from blogger sentiment," in _AAAI Spring Symposium: Computational Approaches to Analyzing Weblogs_, 2006, pp. 155–158.
29. M. Mestyán, T. Yasseri, and J. Kertész, "Early prediction of movie box office success based on Wikipedia activity big data," _PLoS ONE_, vol. 8, no. 8, p. e71226, 2013.
30. R. Balestri, G. B. Standardi, and L. C. Cicala, "An automatic deep learning approach for trailer generation through large language models," _arXiv preprint arXiv:2601.04112_, 2026.
31. A. S. Sharma and R. K. Sharma, "Presenting a larger up-to-date movie dataset," _arXiv preprint arXiv:2104.09210_, 2021.
32. E. Mohamed and M. N. El-Khouly, "A first dataset for film age appropriateness investigation," in _Proc. LREC_, 2020.
33. H. L. Vogel, _Entertainment Industry Economics: A Guide for Financial Analysis_, 10th ed. Cambridge: Cambridge University Press, 2020.
34. W. Xie et al., "Predicting movie success with multi-task learning," _arXiv preprint arXiv:2502.08812_, 2025.
35. M. C. Chiu et al., "Screenplay quality assessment: Can we predict who gets nominated?" in _Proc. NUSe Workshop_, 2020.
36. S. M. Lundberg and S.-I. Lee, "A unified approach to interpreting model predictions," in _Advances in Neural Information Processing Systems (NeurIPS)_, 2017, pp. 4765–4774.
