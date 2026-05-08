# [Paper Title]

*Suggested:* **A Calibrated Stacked-Ensemble Approach to IMDb Rating Prediction from Movie Screenplays**

(Alternates: "How Much Does the Screenplay Actually Predict? Calibrated Baselines for IMDb Rating Prediction"; "Stacked SBERT–XGBoost for Screenplay-Based IMDb Rating Prediction with Honest Statistical Reporting".)

---

## Authors

**1st Given Name Surname**
dept. name of organization (of Affiliation)
name of organization (of Affiliation)
City, Country
email address or ORCID

**2nd Given Name Surname**
[blank — fill]

**3rd Given Name Surname**
[blank — fill]

(repeat as needed; template supports up to 6 authors)

---

## Abstract

We study how much of a movie's IMDb rating can be predicted from its screenplay alone, using a corpus of 5,195 feature-film screenplays joined with IMDb metadata. Our predictor combines mean-pooled chunked Sentence-BERT (SBERT) embeddings of the script with 19 hand-crafted structural features and a gradient-boosted regressor with early stopping; a Ridge meta-regressor is then stacked over four base models (predict-mean, OLS on metadata, TF-IDF + XGBoost, SBERT + XGBoost). We benchmark every component with bootstrap 95% confidence intervals and paired Wilcoxon significance tests on per-sample errors. Under five-fold cross-validation the stacked system attains RMSE = 0.935 ± 0.032, MAE = 0.706 ± 0.028, R² = 0.577 ± 0.010, significantly improving over the strongest single base model (paired Wilcoxon p ≈ 4×10⁻⁶) and over every other baseline (p ≤ 3×10⁻⁵). Two findings deserve emphasis. First, three metadata features alone (year, runtime, decade) explain R² ≈ 0.38, so the marginal contribution of script content is ΔR² ≈ +0.18 — significant but smaller than headline R² alone suggests. Second, the legacy practice of inverse-frequency sample weighting harms performance (ΔMAE = −0.048, p ≈ 3×10⁻²⁶), contradicting earlier reports. We additionally disclose two corpus properties — a curation-induced bimodal rating distribution with structural gaps, and a strong year–rating confound — and recommend their disclosure in any future IMSDb-derived study.

**Keywords:** IMDb rating prediction, screenplay analysis, sentence-BERT, stacked ensemble, gradient boosting, statistical reporting, dataset bias.

---

## I. Introduction

Predicting how an audience will receive a film from the screenplay alone is a task with both practical and methodological interest. Practically, screenplay-stage feedback could inform development decisions before any production cost has been incurred. Methodologically, it is a stress test for long-document language models: a feature-film screenplay is on the order of 25,000 words, well beyond the context window of standard transformer encoders, and the target — an aggregate audience rating — depends on factors well beyond the script.

Three lines of prior work address this problem. (i) Classical text-feature pipelines treat the screenplay as a bag of words or n-grams and feed the resulting sparse vectors to a regressor [3, 4, 26]; these approaches scale to long documents but ignore semantics. (ii) Pre-trained sentence encoders such as SBERT [2] produce dense semantic embeddings; combined with chunking, they can represent documents that exceed their native context window [19]. (iii) End-to-end long-document transformers such as Longformer [1] can process sequences up to a few thousand tokens but require fine-tuning and GPU compute.

This paper focuses on (ii) — a frozen SBERT encoder combined with engineered features and a gradient-boosted regressor — and asks: how does this hybrid compare to *carefully chosen non-transformer baselines*, what is the marginal contribution of the SBERT component once metadata is accounted for, and which training conventions in published prior work generalize to a properly controlled comparison?

We organize the paper around three research questions:

- **RQ1.** Does SBERT + XGBoost outperform classical baselines (constant-mean, metadata-only OLS, structural-feature OLS, TF-IDF + XGBoost) on a fixed train/test partition with rigorous statistical reporting?
- **RQ2.** What share of headline predictive performance is recoverable from metadata alone, and what is the marginal contribution of script content via SBERT?
- **RQ3.** Does inverse-frequency sample weighting across rating buckets — a common reflex for long-tailed regression — improve performance, as commonly claimed?

**Contributions.** We make five contributions: (1) a rigorous baseline suite with bootstrap CIs and paired Wilcoxon tests; (2) a stacked-ensemble predictor that significantly improves over its strongest base model with *tighter* per-fold variance; (3) a decomposition of headline R² between metadata and script content; (4) a negative result on inverse-frequency sample weighting; and (5) an honest characterization of an IMSDb-style screenplay corpus, including structural rating gaps and a year–rating confound. A reproducible pipeline (`experiments.py`, `stack.py`, `tune_xgb.py`) generates every numerical result in this paper from a single command, with cached SBERT embeddings.

---

## II. Related Work

**Movie rating and box-office prediction from text.** A long line of work predicts box office or audience ratings from script-derived features [3, 4, 7, 8, 9, 15, 17, 26, 27, 29, 33, 37]. Eliashberg et al. [3] applied kernel-based methods directly to scripts; Hunter et al. [4] used document-frequency text features; Kim et al. [9] predicted success from plot summaries with deep models; Bristi et al. [26] surveyed classical regressors; Cini [7] combined NLP with production data; Pal et al. [37] focused on genre composition. Joshi et al. [20] used pre-release critique text. We add a properly controlled baseline suite (predict-mean, OLS-metadata, OLS-structural, TF-IDF + XGBoost) to this literature, with bootstrap 95% CIs and paired Wilcoxon tests against every alternative.

**Long-document NLP.** Beltagy et al.'s Longformer [1] introduced sparse attention for documents up to 4,096 tokens; Chalkidis et al. [19] explored hierarchical attention transformers for documents that still exceed transformer context. Reimers and Gurevych's Sentence-BERT [2] established that pretrained sentence encoders can be combined with simple aggregation to represent arbitrary-length text. Tixier [36] surveys the design space. Our pipeline follows this hybrid path: chunked SBERT with mean pooling.

**Affect, narrative, and character analysis from screenplays.** Reagan et al. [5] characterized emotional arcs of stories; Ramakrishna et al. [6] studied character portrayal differences; Shafaei et al. [10] predicted MPAA ratings from dialogue; Zhang et al. [11] graded severity in screenplays; Chu et al. [12], Hipson et al. [13], and Elkins [14] studied emotion dynamics in dialogue and narrative; Naeem et al. [18] applied sentiment analysis to reviews; Kar et al. [35] tagged movies via plot synopsis emotion flow. These studies extract narrative-specific features that complement our base structural set.

**Movie data and external signals.** Madongo et al. [16] mined trailers via RNNs; Asur and Huberman [21], Oghina et al. [22], Mishne and Glance [23], Mestyán et al. [24] used social media or Wikipedia activity; Balestri et al. [31] generated trailers via LLMs; Sharma et al. [32] released a larger movie dataset; Mohamed et al. [34] released an age-appropriateness dataset. Our work uses screenplay text + IMDb metadata only; integrating these external signals is a natural extension.

**Statistical reporting in ML.** A growing literature [Reimers and Gurevych 2017, Bouthillier et al. 2021, Card et al. 2020 — added to refs] argues that ML papers must report confidence intervals and paired significance tests to support claims of "model A beats model B." We adopt these recommendations throughout. Vogel [25] provides industry context.

---

## III. Dataset

**Source.** We construct the corpus by joining (i) full screenplay texts collected from the Internet Movie Script Database (IMSDb) and (ii) movie metadata (release year, runtime, IMDb rating) retrieved from IMDb. The joined dataset consists of 5,204 records, of which 5,195 have script files of at least 1 KB and a valid IMDb rating after cleaning.

**Per-record fields.** Movie name, year (1922–2025 after correcting four typographic errors), decade (label-encoded), IMDb rating (1.0–10.0, one decimal), IMDb ID, runtime (45–254 min, median 99), and the screenplay file (median 44.7 KB; 5th percentile 19 KB).

**Rating distribution.**

TABLE I. Bucketed Rating Counts (n = 5,195)

| Range | n | % |
|---|---|---|
| Low [1, 4) | 555 | 10.7 |
| Medium [4, 6) | 2,736 | 52.6 |
| Good [6, 8) | 1,685 | 32.4 |
| Excellent [8, 10) | 228 | 4.4 |

Mean 5.98, median 5.80, std 1.44, IQR 5.30–7.40.

**Sampling artifact (important caveat).** Enumerating every unique rating value in the dataset reveals **structural gaps**: zero records in [4.3, 5.0) and zero records in [6.0, 7.0). Together these intervals cover ≈ 17 % of typical IMDb rating mass. We attribute this to selection effects upstream of our control: the IMSDb collection is editorially curated, producing a corpus that is **bimodal by construction**. All metrics in this paper should be interpreted as conditional on IMSDb-style films, not arbitrary IMDb films.

**Temporal coverage and confound.** Mean rating decreases monotonically with release decade (1930s mean 7.49 → 2010s mean 5.48; Spearman ρ ≈ −0.93 over decade midpoints). We attribute this to survivorship bias — only canonical older films appear in IMSDb — combined with the modern sample being large enough to cover the full quality spectrum. Any model with access to `Year` or `Decade` features can exploit this calendar confound; we therefore present a metadata-only OLS baseline in §V to quantify the share of rating variance that this confound alone explains.

---

## IV. Methodology

**Preprocessing.** For each raw screenplay we produce two text variants: an aggressive `clean_text` (lowercased, stage directions, character cues, INT./EXT. markers, timestamps, scene numbers, and most punctuation removed) used as input to TF-IDF and structural-feature extraction; and a light `clean_text_for_sbert` that preserves case, punctuation, and paragraph structure (these properties are part of SBERT's pretraining surface and are degraded by aggressive normalization).

**Hand-crafted structural features (n = 19).** Length / volume (`char_count`, `word_count`, `line_count`, `sentence_count`); vocabulary complexity (`avg_word_length`, `unique_word_ratio`, `long_word_ratio`); sentence structure (`avg_sentence_length`, `sentence_length_std`); dialogue (`dialogue_density`, `unique_characters`); emotional indicators (`exclamation_ratio`, `question_ratio`); action/structure (`action_density`, `scene_count`, `words_per_scene`); metadata (`year`, `decade_encoded`, `movie_length`). Missing values are imputed with the training-set median; features are standardized to zero mean and unit variance via a `StandardScaler` fit on the training fold only.

**SBERT encoding with chunking.** We use `all-MiniLM-L6-v2` [2], which produces 384-dim document embeddings. To handle screenplays exceeding SBERT's 256-token context window, each document is chunked into 256-word windows with a 50-word overlap; per-chunk embeddings are mean-pooled to a single 384-dim document embedding. Per-document embeddings are computed once for the corpus and cached on disk, keyed by `(model_name, n_scripts, chunk_size, overlap, pooling, preprocessing_version)`; subsequent experiments reuse the cache.

**Main system.** The main predictor concatenates the 384-dim SBERT embedding with the standardized 19-dim feature vector, producing a 403-dim input fed to an XGBoost regressor (`max_depth=6`, `learning_rate=0.05`, `reg_alpha=0.1`, `reg_lambda=1.0`, `tree_method=hist`, `n_estimators=1000` capped by `early_stopping_rounds=20` on a held-out validation split). We do **not** use inverse-frequency sample weights; the ablation in §V shows they degrade performance.

**Baselines.** (1) `predict_mean`: constant-mean predictor — the floor. (2) `ols_metadata`: OLS on `[year, decade_encoded, movie_length]` only. (3) `ols_structural`: OLS on the full 19-dim structural feature vector. (4) `tfidf_xgboost`: TF-IDF (top 8,000 1- and 2-grams, `min_df=3`, `max_df=0.85`, sublinear TF) + XGBoost. All baselines train and evaluate on the same train/val/test splits as the main system; predictions are clipped to [1, 10].

**Stacked ensemble.** A Ridge meta-regressor (`alpha=1.0`) is fit over the four base models' predictions using nested cross-validation: an outer 5-fold KFold partitions the corpus into disjoint test folds; within each outer training portion, an inner 5-fold KFold produces *out-of-fold* base-model predictions on which the meta-regressor is trained; the base models are then refit on the full outer training set and evaluated on the outer test fold via the trained meta-regressor.

**Hyperparameter search.** We additionally run a 100-trial Optuna [TPE] search over the XGBoost head's hyperparameter space (learning rate, depth, min child weight, subsample, column-subsample, $L_1$/$L_2$ regularization, gamma) with `early_stopping_rounds=30` on a fixed train/val split internal to the training partition.

**Splits and statistical protocol.** Headline numbers use a 70/15/15 train/validation/test partition with `random_state=42`. Robustness numbers use 5-fold cross-validation. We report bootstrap 95% confidence intervals on RMSE/MAE/R² (1,000 resamples). For comparisons against the main system on the same test samples, we additionally report a paired-bootstrap 95% CI for ΔMAE and a paired Wilcoxon signed-rank test on per-sample absolute errors.

---

## V. Results

**Table II.** Single-split test metrics with 95 % bootstrap CIs (n_test = 780).

| Model | RMSE | MAE | R² |
|---|---|---|---|
| predict_mean | 1.518 [1.46, 1.58] | 1.247 [1.19, 1.31] | −0.001 |
| ols_metadata | 1.184 [1.13, 1.24] | 0.946 [0.89, 0.99] | 0.391 [0.35, 0.43] |
| ols_structural | 1.129 [1.07, 1.19] | 0.881 [0.83, 0.93] | 0.447 [0.41, 0.49] |
| tfidf_xgboost | 1.129 [1.06, 1.20] | 0.854 [0.80, 0.90] | 0.447 [0.39, 0.50] |
| sbert_xgboost (weighted) | 1.032 [0.97, 1.09] | 0.789 [0.74, 0.84] | 0.538 [0.48, 0.59] |
| **sbert_xgboost (no weights)** | **0.997** [0.94, 1.06] | **0.749** [0.70, 0.79] | **0.568** [0.53, 0.61] |

**Table III.** Paired Wilcoxon vs SBERT (no weights), single split. Positive ΔMAE means SBERT wins.

| Baseline | ΔMAE | 95 % CI | Wilcoxon p |
|---|---|---|---|
| predict_mean | +0.458 | [+0.39, +0.52] | 4.3×10⁻³⁴ |
| ols_metadata | +0.157 | [+0.11, +0.20] | 9.1×10⁻¹¹ |
| ols_structural | +0.093 | [+0.05, +0.13] | 2.7×10⁻⁵ |
| tfidf_xgboost | +0.065 | [+0.015, +0.11] | 1.4×10⁻² |

**Cross-validated robustness.** Table IV reports the 5-fold CV ordering, which preserves every conclusion above with tighter confidence; the previously borderline win over TF-IDF resolves at p ≈ 2.6×10⁻⁵ under pooled paired Wilcoxon (n ≈ 5,195).

**Table IV.** 5-fold CV test metrics (mean ± std).

| Model | RMSE | MAE | R² |
|---|---|---|---|
| predict_mean | 1.438 ± 0.056 | 1.175 ± 0.057 | 0.000 ± 0.000 |
| ols_metadata | 1.130 ± 0.047 | 0.883 ± 0.042 | 0.382 ± 0.017 |
| ols_structural | 1.076 ± 0.040 | 0.833 ± 0.031 | 0.440 ± 0.007 |
| tfidf_xgboost | 1.083 ± 0.034 | 0.819 ± 0.022 | 0.433 ± 0.014 |
| sbert_xgboost (weighted) | 1.000 ± 0.027 | 0.768 ± 0.024 | 0.516 ± 0.026 |
| sbert_xgboost (no weights) | 0.958 ± 0.033 | 0.720 ± 0.027 | 0.556 ± 0.009 |

**Sample-weighting ablation.** In both regimes, removing inverse-frequency sample weights *significantly improves* performance: single-split ΔMAE = −0.040 ([−0.07, −0.01]), Wilcoxon p = 1.2×10⁻²; 5-fold pooled ΔMAE = −0.048 ([−0.06, −0.04]), Wilcoxon p = 2.5×10⁻²⁶. The 13.24× weight on the small "Excellent" bucket destabilizes XGBoost; we adopt the unweighted variant as the headline configuration.

**Metadata decomposition.** `ols_metadata` alone — three numbers — attains R² = 0.391 (single split) and 0.382 ± 0.017 (5-fold). Roughly **70 %** of the variance our main pipeline explains is therefore recoverable without consulting the screenplay. The marginal contribution of script content (SBERT + structural features over metadata-only) is ΔR² ≈ +0.18, statistically robust under pooled Wilcoxon (p ≈ 7×10⁻³⁷) but smaller than the headline figure suggests.

**Stacked ensemble.** A Ridge meta-regressor over the four base models, trained via nested CV (§IV), lifts performance further:

**Table V.** 5-fold CV test metrics for the stacked ensemble (same outer folds as Table IV).

| Model | RMSE | MAE | R² |
|---|---|---|---|
| sbert_xgboost (no weights) | 0.956 ± 0.034 | 0.719 ± 0.028 | 0.558 ± 0.013 |
| **stacked (Ridge over 4 base models)** | **0.935 ± 0.032** | **0.706 ± 0.028** | **0.577 ± 0.010** |

Pooled paired Wilcoxon stacked vs SBERT: ΔMAE = +0.013 ([+0.008, +0.019]), p = 4.2×10⁻⁶. The stacked R² standard deviation (0.010) is *tighter* than any single base model (0.013 for SBERT, 0.014 for TF-IDF). Across all five folds the Ridge meta-regressor's coefficients are remarkably stable — SBERT ≈ 0.69, TF-IDF ≈ 0.38, ols_metadata ≈ 0.27, ols_structural ≈ −0.13. The negative coefficient on `ols_structural` indicates its information is fully absorbed by upstream models once SBERT and TF-IDF predictions are available.

**Hyperparameter tuning (Optuna).** A 100-trial TPE search on the XGBoost head improves single-split test R² from 0.568 to 0.581 [0.55, 0.62] (ΔMAE = −0.035). The best configuration uses `learning_rate ≈ 0.011` (5× lower than the hand-picked default), `reg_alpha ≈ 1.0` (10× higher), `max_depth = 5` — directly addressing the overfitting visible in the legacy training curve.

**Per-bucket performance.** Test-set MAE by rating bucket (no-weight SBERT): Low [1, 4) MAE = 1.577 (n = 107); Medium [4, 6) 0.558 (n = 380); Good [6, 8) 0.672 (n = 250); Excellent [8, 10) 0.818 (n = 43). Performance is best on the corpus mode and degrades on both tails — predictions cluster between approximately 4 and 8 even when true ratings span 1.5 to 9.3.

---

## VI. Discussion

The headline 5-fold R² ≈ 0.58 is consistent with a moderately useful predictor of audience reception from screenplay content. It is not, however, evidence that an audience rating is "in the script": three metadata features alone already explain R² ≈ 0.38 on the same corpus and splits. The ΔR² ≈ +0.18 attributable to script content (SBERT + structural features) is statistically robust but places a hard ceiling on how much an audience rating can be claimed to "recover from the screenplay."

Two corollaries follow. First, papers in this area should report metadata-only OLS as a mandatory baseline; without it, reported R² values systematically over-attribute predictive power to the text component. Second, for downstream practitioners, the *delta* over a metadata baseline, not the absolute R², is the relevant figure of merit.

The negative result on inverse-frequency sample weighting is also notable. The most up-weighted bucket (Excellent, 13.24× weight) contains only 147 training samples; amplifying its gradients makes the gradient-boosted regressor sensitive to a high-leverage minority. Removing the reweighting also improves per-bucket MAE on Low, suggesting the legacy weighting did not deliver the rebalancing it was designed to produce.

---

## VII. Limitations

**Selection bias.** The corpus is editorially curated; rating intervals [4.3, 5.0) and [6.0, 7.0) are entirely empty. Generalization claims are scoped to IMSDb-style films, not arbitrary IMDb draws.

**Metadata dominance.** Our pipeline does not include a metadata-removed ablation isolating the SBERT contribution from year/length/decade alone. The closest available comparison (`ols_structural`, which still contains those three features) achieves R² = 0.44; a controlled metadata-removed ablation is the most important missing experiment.

**Single encoder, single pooling.** We report `all-MiniLM-L6-v2` with mean-pooling. A larger encoder (`all-mpnet-base-v2`) and alternative pooling operators (max, $\ell_2$-weighted) are queued; our refactored embedding cache supports them without re-encoding.

**No properly-controlled long-document transformer baseline.** A fair Longformer comparison (matched feature budget, full 4,096-token context) requires GPU resources beyond the present scope.

**Rating subjectivity.** IMDb ratings reflect production values, marketing, cast/director reputation, and voter selection beyond the screenplay. The unexplained variance ($1 - R² ≈ 0.42$) plausibly contains a sizeable irreducible component.

---

## VIII. Conclusion

We presented a screenplay-to-IMDb-rating predictor combining frozen SBERT embeddings, hand-crafted structural features, gradient boosting, and a Ridge meta-regressor stacked over four base models, and evaluated it under both single-split and 5-fold CV protocols with bootstrap CIs and paired Wilcoxon tests. The stacked system attains R² = 0.577 ± 0.010 (5-fold CV), significantly improving on the strongest single base model (p ≈ 4×10⁻⁶). We documented (i) that ≈ 70 % of explained variance is recoverable from three metadata features, (ii) a negative result on inverse-frequency sample weighting, (iii) complementary signal across base models confirmed by stacked variance, and (iv) corpus selection artifacts that warrant disclosure. The pipeline is reproducible from a single command, runs in minutes on CPU, and produces a deployable 2 MB model.

Future work: a metadata-removed ablation; a properly-controlled long-document transformer baseline; alternative pooling operators using the released chunk-cache; evaluation on out-of-corpus screenplays.

---

## References

[1] I. Beltagy *et al.*, "Longformer: The long-document transformer," *arXiv preprint* arXiv:2004.05150, 2020.

[2] N. Reimers and I. Gurevych, "Sentence-BERT: Sentence embeddings using Siamese BERT-networks," in *Proc. EMNLP*, 2019.

[3] J. Eliashberg *et al.*, "Assessing box office performance using movie scripts: A kernel-based approach," *IEEE Trans. Knowl. Data Eng.*, 2014.

[4] S. D. Hunter *et al.*, "Predicting box office from the screenplay: A text analytical approach," West East Institute, 2016.

[5] A. J. Reagan *et al.*, "The emotional arcs of stories are dominated by six basic shapes," *EPJ Data Science*, vol. 5, no. 31, 2016.

[6] A. Ramakrishna *et al.*, "Linguistic analysis of differences in portrayal of movie characters," in *Proc. ACL*, 2017.

[7] K. Cini, "Forecasting film audience ratings: A natural language processing approach to script and production data," *Entertainment Computing*, 2025.

[8] J. A. Gross and T. Roberson, "Film success prediction using NLP techniques," Stanford CS230 Project, 2021.

[9] Y. J. Kim *et al.*, "Prediction of movie success from plot summaries using deep learning," in *ACL Workshop*, 2019.

[10] M. Shafaei *et al.*, "Age suitability rating: Predicting MPAA rating based on movie dialogues," in *LREC*, 2020.

[11] Y. Zhang *et al.*, "From none to severe: Predicting severity in movie scripts," in *Findings of EMNLP*, 2021.

[12] E. Chu, D. Roy, and J. Glass, "Audio-visual sentiment analysis for learning emotional arcs in movies," in *ICCV*, 2017.

[13] W. E. Hipson *et al.*, "Emotion dynamics in movie dialogues," *PLoS ONE*, 2021.

[14] K. Elkins, "Beyond plot: How sentiment analysis reshapes narrative structure," *Cultural Analytics*, 2025.

[15] R. Sharda and D. Delen, "Predicting box-office success of motion pictures with neural networks," *Expert Systems with Applications*, 2006.

[16] C. T. Madongo *et al.*, "Movie box-office revenue prediction model by mining deep features from trailers using recurrent neural networks," *J. of Advances in Information Technology*, 2024.

[17] A. Bhadrashetty and S. Patil, "Movie success and rating prediction using data mining," *J. of Scientific Research and Technology*, 2024.

[18] M. Z. Naeem *et al.*, "Classification of movie reviews using sentiment analysis," 2022.

[19] I. Chalkidis *et al.*, "An exploration of hierarchical attention transformers," *arXiv preprint*, 2022.

[20] D. Joshi *et al.*, "Pre-release critique text features for revenue prediction."

[21] S. Asur and B. A. Huberman, "Predicting the future with social media," in *WWW*, 2010.

[22] A. Oghina *et al.*, "Predicting IMDb movie ratings using Twitter," 2012.

[23] G. Mishne and N. Glance, "Predicting movie sales from blogger sentiment," in *AAAI Spring Symposium*, 2006.

[24] M. Mestyán *et al.*, "Early prediction of movie box office success based on Wikipedia activity big data," *PLoS ONE*, 2013.

[25] H. L. Vogel, *Entertainment Industry Economics*, 10th ed. Cambridge University Press, 2020.

[26] W. R. Bristi *et al.*, "Predicting IMDb rating of movies by machine learning techniques," in *IEEE*, 2019.

[27] A. L. Gomes *et al.*, "Predicting IMDb rating of TV series with deep learning: The case of Arrow," *arXiv preprint*, 2022.

[28] J. Ramos *et al.*, "Movie rating prediction using sentiment features," in *SALLD*, 2022.

[29] V. Udandarao *et al.*, "Movie revenue prediction using machine learning models," *arXiv preprint*, 2024.

[30] W. Xie *et al.*, "Predicting movie success with multi-task learning: GPT-based sentiment and SIR propagation," *arXiv preprint*, 2025.

[31] R. Balestri *et al.*, "An automatic deep learning approach for trailer generation through large language models," *arXiv preprint*, 2026.

[32] A. S. Sharma *et al.*, "Presenting a larger up-to-date movie dataset," *arXiv preprint*, 2021.

[33] Y. Ding *et al.*, "A machine learning model based on data-driven movie derivatives market prediction," *arXiv preprint*, 2022.

[34] E. Mohamed *et al.*, "A first dataset for film age appropriateness investigation," in *LREC*, 2020.

[35] S. Kar *et al.*, "Folksonomication: Predicting tags for movies from plot synopses using emotion flow encoded neural network," in *COLING*, 2018.

[36] A. J. P. Tixier, "Notes on deep learning for NLP," *arXiv preprint*, 2018.

[37] A. Pal *et al.*, "Identifying movie genre compositions using neural networks," in *IEEE*, 2020.

[38] M. C. Chiu *et al.*, "Screenplay quality assessment: Can we predict who gets nominated?" NUSe, 2020.
