# What Survives a Strict Split? An Empirical Audit of Long-Document Screenplay Modelling for IMDb Rating Prediction

**Anonymous authors**

***Abstract**—Screenplay-to-rating regression is usually reported on a random split of an archival corpus. We argue that this number is not the number a reader wants, and we measure the gap. Working from a corpus of 5,204 IMSDb screenplays joined to IMDb metadata, we first run a record-identity audit — content SHA-256 digests, IMDb identifiers, title×year keys, and a normalised title-family key — and publish the resulting exclusion ledger, retaining 5,164 records after removing nine sub-1 KB fragments, four corrupt release years, and twenty-seven records whose positional identity in the archived embedding matrix cannot be resolved uniquely. We then quantify a defect invisible at the API surface: the pipeline slides a 256-word window over each script, but the encoder admits only 254 WordPiece tokens, so 98.45% of windows are cut and 31.60% of all tokens never reach the model. Finally, we evaluate identical representations and learners under three partitions: random $k$-fold, title-family GroupKFold, and a chronological 2020–2025 holdout. Skill is strongly protocol-dependent: the best system reports $R^2=0.549$ $[0.530, 0.567]$ under random folds, $0.536$ $[0.516, 0.554]$ once franchises are held together, and $0.309$ $[0.199, 0.404]$ when the test set lies in the future. Block-permutation analysis on the temporal holdout explains why. Release year and decade, which jointly supply 36.6% of in-sample split gain, contribute exactly $\Delta\text{MAE}=0.000$ once test years fall outside the training range, while runtime alone (+0.199) matches the entire 384-dimensional sentence-embedding block (+0.195). Semantic embeddings survive the strict split; the calendar confound does not.*

***Index Terms**—data leakage, evaluation protocol, sentence embeddings, long-document NLP, screenplay analysis, temporal generalisation, grouped cross-validation.*

---

## I. Introduction

A feature-film screenplay runs to roughly 25,000 words. Sentence encoders read 256 tokens. That mismatch is the central engineering problem in screenplay modelling, and the usual answer — slide a window, embed each window, average the results — is cheap, reproducible, and widely used. What is far less often examined is whether the evaluation wrapped around that pipeline measures anything a practitioner could rely on, because an archival screenplay corpus assembled from a community repository carries two structural hazards that a shuffled train/test split will silently convert into apparent skill.

The first hazard is duplication. Community archives accumulate the same work under several file names, franchises share vocabulary and structure across installments, and remakes reuse plot and dialogue wholesale. Under a random partition, a near-copy in the training set makes its twin in the test set trivially predictable. The effect is not exotic; it is the dominant failure mode catalogued in surveys of leakage-driven irreproducibility across applied machine learning [32], and has been documented specifically for train–test overlap in NLP benchmarks [30], [31].

The second hazard is time. Ratings in an archival corpus are collected once, at scrape time, but the films span a century, and both the selection process and the rating behaviour of the audience drift. A model given a release-year feature can recover a large share of target variance from the calendar alone, without reading a single line of dialogue. That shortcut evaporates the moment the test films are drawn from years the model never saw, which is the only setting in which a screenplay-stage predictor would actually be deployed.

Layered underneath both is a quieter defect specific to chunked long-document pipelines. Chunking is almost always specified in words; truncation always happens in subword tokens. Nobody is told when the budget is exceeded — the encoder simply drops the tail — so a configuration that reads as "256-word windows with 50-word overlap" can in practice discard a third of the corpus without emitting a single warning.

This paper is an audit, not a leaderboard entry. We take an existing SBERT-plus-gradient-boosting pipeline over 5,204 IMSDb screenplays and ask what its reported performance is actually made of. Three contributions follow:

- **C1 — Record-identity protocol and exclusion ledger.** We define film identity through four keys — content SHA-256 digest, IMDb identifier, title×year, and a normalised title-family stem — audit all 5,204 joined records against each, and report every exclusion (Section III). The audited corpus contains no duplicate identifiers and no duplicate title×year pairs, but 80 title collisions and 164 multi-member title families that a random split would straddle.
- **C2 — Measurement of silent truncation.** Running the encoder's own tokeniser over the pipeline's own windows, we find that 98.45% of 256-word windows exceed the 254-token content budget and that 31.60% of WordPiece tokens are discarded before the encoder sees them, with a median window of 371 tokens against a budget of 254 (Section IV-B).
- **C3 — Three-protocol evaluation with paired inference.** Holding representation and learner fixed, we vary only the partition: random $k$-fold, title-family GroupKFold, and a chronological 2020–2025 holdout. We report bootstrap intervals, paired Wilcoxon signed-rank tests with matched-pairs effect sizes, and a block-permutation attribution that isolates which feature families survive temporal extrapolation (Sections V and VI).

We do not claim a new state of the art. We claim that the difference between $R^2=0.549$ and $R^2=0.309$ on the same corpus, the same features, and the same learner is the finding.

---

## II. Related Work

**Script-driven forecasting and narrative analysis.** Early work extracted features from scripts to forecast commercial outcomes: Eliashberg et al. [3] used spec-script descriptors in kernel regression to predict ROI, while Hunter et al. [4] targeted box-office revenue. Parallel research maps narrative structures, such as emotional arc typologies [5], character dialogue linguistics [6], and multimodal channels [9]. Downstream script tasks include MPAA age-suitability ratings [7], [25], severity classification [8], and tag prediction [26]. Screenplay quality prediction in awards settings [27] shares our fundamental obstacle: ratings represent social aggregations only partially dictated by textual content.

**Metadata and multimodal forecasting.** Alternative forecasting leverages non-textual channels: production budgets and cast metadata [10], social media volume [15], [17], Twitter chatter [16], Wikipedia page views [18], trailers [11], and review sentiment [13], [22]. Tabular IMDb rating regression is widely studied [12], [20], [23], including deep models for serials [21] and joint sentiment-diffusion networks [24]. Vogel [19] notes the natural ceiling: box-office and audience reception are heavily governed by release timing, competition, and distribution budgets unobservable in a script.

**Long-document representations.** Sentence-BERT [2] yields dense sentence representations from Siamese objectives. Applying fixed-length encoders to book- or script-length documents requires hierarchical attention [14], sparse attention transformers [1], or chunk-and-pool averaging. Chunk-and-pool is the prevailing default because it avoids expensive fine-tuning, runs on CPU, and scales linearly. Classical $n$-gram TF-IDF remains a strong lexical baseline on long technical text.

**Data leakage and evaluation protocol design.** Gorman and Bedrick [28] proved that POS tagger rankings permute under alternative splits, prompting Søgaard et al. [29] to urge split designs aligned with claimed generalization. Lewis et al. [30] and Elangovan et al. [31] formalized train–test leakage and overlap measurement in NLP benchmarks. Kapoor and Narayanan [32] catalogued leakage failure modes across applied ML, arguing for pre-registered partitions. Temporal out-of-distribution evaluation [33] and corpus provenance documentation [34], [35] further highlight the necessity of strict split designs.

---

## III. Corpus Integrity and Exclusion Audit

### A. Source and Join
The corpus joins full screenplay texts scraped from the Internet Movie Script Database (IMSDb) to IMDb metadata. Each of the 5,204 joined records contains title, release year, decade bin, IMDb identifier (`tt` ID), runtime in minutes, IMDb rating (1.0–10.0 scale, one-decimal resolution), and script text path. Ratings exhibit mean 5.98, median 5.80, standard deviation 1.44; runtimes span 45–254 minutes (median 99).

### B. Four Identity Keys
Deduplication requires explicit identity criteria. We audit four keys of decreasing strictness:
1. *Content digest:* $\text{SHA-256}(b)$ of raw bytes $b$. Collisions resolve by preserving the earliest record in load order.
2. *IMDb identifier:* Canonical `tt`-identifier catches identical films scraped under varied titles.
3. *Title×year:* Fallback for missing or malformed IDs.
4. *Title family:* Normalizes title $t$ by lowercasing, stripping non-alphanumeric tokens and leading articles, and iteratively stripping trailing sequel markers (Arabic/Roman numerals, "Part $N$"). Families group installments together (e.g., *Toy Story 1–3* map to `toy story`). Algorithm 1 details the reduction.

```
Algorithm 1: Title-Family Normalisation phi(t)
Input: Raw title string t
Output: Family stem string u
1: u <- Lowercase(Trim(t))
2: u <- Replace(u, [^a-z0-9 ], " ")
3: u <- CollapseSpaces(u)
4: u <- StripLeadingArticles(u)
5: repeat
6:    u_prev <- u
7:    u <- StripTrailingSequelMarkers(u)
8: until u == u_prev
9: return u if u != "" else Lowercase(Trim(t))
```

### C. Exclusion Flow and Audit Results
Let $R_0$ be the initial set ($|R_0|=5,204$). Four nested filters apply sequentially:
$$\mathcal{R}_k = \{r \in \mathcal{R}_{k-1} : \pi_k(r)\}$$
where $\pi_{\text{len}}$ requires $|text| \ge 1,000$ chars; $\pi_{\text{hash}}$ checks byte uniqueness; $\pi_{\text{year}}$ enforces release year $\in [1900, 2025]$; and $\pi_{\text{id}}$ guarantees unique positional mapping in the archived embedding matrix:
$$5,204 \xrightarrow{\pi_{\text{len}}} 5,195 \xrightarrow{\pi_{\text{hash}}} 5,195 \xrightarrow{\pi_{\text{year}}} 5,191 \xrightarrow{\pi_{\text{id}}} 5,164$$

Table I details the exclusion ledger.

**TABLE I.** Corpus Integrity Audit and Exclusion Ledger ($N=5,204$).

| Check / Exclusion Category | Count | % of Corpus |
|:---|:---:|:---:|
| Joined records (IMSDb $\bowtie$ IMDb) | 5,204 | 100.00% |
| *Identity Keys* | | |
| Distinct script file names | 5,204 | 100.00% |
| Distinct IMDb identifiers | 5,204 | 100.00% |
| Duplicate title×year pairs | 0 | 0.00% |
| Title collisions (same title, different year) | 80 groups / 165 rec. | 3.17% |
| SHA-256 duplicates (audited subset, $n=356$) | 0 | — |
| *Exclusions* | | |
| Script text $<1,000$ characters | 9 | 0.17% |
| Corrupt release year outside $[1900, 2025]$ | 4 | 0.08% |
| Unresolvable matrix positional identity | 27 | 0.52% |
| **Final Analysis Corpus** | **5,164** | **99.23%** |
| Title families (grouping units) | 4,979 | — |
| Multi-member families | 164 | — |

Three findings emerge from Table I:
1. *Clean identifier join:* Zero duplicate IMDb IDs or duplicate title×year pairs exist. Naive duplicate leakage is absent.
2. *Legitimate title collisions:* 80 titles span 165 records (remakes and same-title films). Grouping them into families prevents dialogue and plot leakage across train and test.
3. *Corrupted metadata:* Four release years carry typographical errors (years 1080 and 1088 for *Breaker Morant*, *Mississippi Burning*, *Torch Song Trilogy*, *The Blues Brothers*). The original pipeline label-encoded 14 decade strings, injecting corrupt ordinal values (`1080s`, `1900s`, `1996s`). We excise these records rather than silently patching them.
4. *Positional ambiguity:* 27 rows had ambiguous indices in the cached embedding matrix (a consequence of caching without explicit primary keys); these are pruned to guarantee integrity.

---

## IV. Methodology and Representations

### A. Screenplay Feature Extraction
Nineteen numeric predictors are extracted: 16 structural statistics from raw scripts plus 3 metadata fields (runtime, release year, decade). The 16 script features encompass:
- *Size & volume:* `char_count`, `word_count`, `line_count`, `sentence_count`, `scene_count` (case-insensitive `INT.`/`EXT.` count), and inverse tempo `words_per_scene`.
- *Lexical sophistication:* `avg_word_length`, `unique_word_ratio` (type-token ratio), and `long_word_ratio` ($\ge 8$ chars).
- *Sentence geometry:* `avg_sentence_length` and `sentence_length_std`.
- *Dialogue & affect:* `dialogue_density` (share of lines matching ALL-CAPS speaker cue), `unique_characters`, `exclamation_ratio`, `question_ratio`, and `action_density` (bracketed stage directions).

*Degeneracy audit:* Mean `dialogue_density` is 0.0087 ($\pm 0.0361$) and mean `unique_characters` is 3.41. The extraction regex assumed a strict `NAME:` convention missing from most IMSDb formats, rendering both features transcription artifacts rather than narrative metrics.

### B. Token-Aligned Chunking and Silent Truncation
Given script word tokens $D = (w_1, \ldots, w_N)$, windows of length $L=256$ words with overlap $o=50$ yield stride $s = L - o = 206$. The $j$-th window is $C_j = (w_{(j-1)s+1}, \ldots, w_{\min((j-1)s+L, N)})$, giving $M = \lceil (N-L)/s \rceil + 1$ windows. Each window embeds via encoder $f: C \to \mathbb{R}^d$ ($d=384$, `all-MiniLM-L6-v2`), and document representation is the arithmetic mean:
$$\mathbf{e}(D) = \frac{1}{M}\sum_{j=1}^M f(C_j)$$

Because $f(C_j)$ is $L_2$-normalised, the squared norm of the pooled vector is:
$$\|\mathbf{e}(D)\|_2^2 = \frac{1}{M^2}\left(M + 2\sum_{1 \le i < j \le M} \langle f(C_i), f(C_j) \rangle\right)$$
which tends to $1/M$ as window embeddings decorrelate. With median $M=36$ windows per script, localized scene signal is aggressively diluted.

**Silent token truncation:** The transformer accepts at most $T_{\max}=256$ positions, with 2 reserved for `[CLS]` and `[SEP]`, leaving a content budget $B = 254$ WordPiece tokens. Letting $\tau(\cdot)$ be the tokenizer, the encoder observes $\tau(C_j)$ truncated to $B$. The discarded token fraction for document $D$ is:
$$\rho(D) = 1 - \frac{\sum_{j=1}^M \min(|\tau(C_j)|, B)}{\sum_{j=1}^M |\tau(C_j)|}$$

Evaluating $\tau$ across 14,558 windows from 356 scripts reveals that 256 English words average $369.3$ WordPiece tokens (median 371, 99th percentile 453, maximum 613). Consequently, **98.45% of windows exceed the token limit**, and **31.60% of all generated tokens are silently dropped** before encoding.

### C. Models and Controls
- **Lexical Control:** TF-IDF on unigrams/bigrams (8,000 features, English stop words removed, min document frequency 3, max 0.85, sublinear term frequency $1 + \log(\text{tf})$) trained on normalized text and fed to gradient-boosted trees.
- **Learners:** Gradient-boosted decision trees (500 estimators, learning rate 0.05, max depth 6, histogram splitting, early stopping after 20 rounds, predictions clipped to $[1.0, 10.0]$).
- **Probes:** Ordinary Least Squares (OLS) on the 3 metadata columns; Ridge regression ($\alpha=1.5$) on standardized embeddings; and a constant-mean baseline.

### D. Partition Protocols
We evaluate across three split designs on the $n=5,164$ corpus:
- **P1 (Random):** Standard shuffled 5-fold cross-validation. Average franchise test overlap rate $\lambda = 5.50\%$.
- **P2 (Grouped):** GroupKFold grouped by title family ($|G|=4,979$), guaranteeing $\lambda = 0\%$. Realized fold sizes are 1,032–1,033, identical to P1.
- **P3 (Chronological):** Train on films released $\le 2015$ ($n=3,918$); validate on 2016–2019 ($n=731$); test on 2020–2025 ($n=515$, $\lambda = 6.41\%$). Model selection and early stopping use validation films strictly preceding test films.

### E. Statistical Inference
Point estimates are paired with 95% percentile bootstrap intervals ($B=1,000$). System differences are tested via the two-sided paired Wilcoxon signed-rank test on absolute errors $e_i^A - e_i^B$, reporting non-tied pair counts, test statistic $W$, exact $p$-value, and matched-pairs rank-biserial correlation:
$$r = \frac{R^+ - R^-}{R^+ + R^-} \in [-1, 1]$$

---

## V. Experimental Results

### A. The Impact of Split Protocol
Tables II and III present performance across the three protocols.

**TABLE II.** 5-Fold Cross-Validation Performance: Random (P1) vs. Grouped by Title Family (P2) ($N=5,164$).

| System | P1 RMSE | P1 MAE | P1 $R^2$ | P2 RMSE | P2 MAE | P2 $R^2$ | $\Delta R^2$ (P2 $-$ P1) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Constant mean | 1.4373 | 1.1761 | $-0.0000$ | 1.4375 | 1.1765 | $-0.0004$ | $-0.0004$ |
| Metadata OLS | 1.1237 | 0.8701 | 0.3888 | 1.1236 | 0.8695 | 0.3889 | $+0.0001$ |
| SBERT ridge | 1.1122 | 0.8618 | 0.4012 | 1.1238 | 0.8742 | 0.3887 | $-0.0125$ |
| SBERT + XGBoost | 1.0907 | 0.8355 | 0.4241 | 1.1026 | 0.8445 | 0.4114 | $-0.0127$ |
| **SBERT + meta + XGBoost** | **0.9652** | **0.7276** | **0.5490** | **0.9791** | **0.7309** | **0.5359** | **$-0.0131$** |

*P1 95% CI: $R^2 \in [0.5298, 0.5671]$; P2 95% CI: $R^2 \in [0.5162, 0.5537]$.*

**TABLE III.** Chronological Out-of-Distribution Holdout (P3: Train $\le 2015$, Val 2016–2019, Test 2020–2025; $n=515$).

| System | Test RMSE | Test MAE | Test $R^2$ | 95% Bootstrap CI ($R^2$) |
|:---|:---:|:---:|:---:|:---:|
| Constant mean | 1.2230 | 0.9762 | $-0.0947$ | $[-0.155, -0.041]$ |
| Metadata OLS | 1.0794 | 0.8384 | 0.1473 | $[0.061, 0.224]$ |
| SBERT ridge | 1.0719 | 0.8417 | 0.1591 | $[0.076, 0.236]$ |
| SBERT + XGBoost | 1.0007 | 0.7469 | 0.2671 | $[0.174, 0.354]$ |
| **SBERT + meta + XGBoost** | **0.9717** | **0.7232** | **0.3089** | **$[0.1985, 0.4038]$** |

*Target drift: Train mean $6.01 \pm 1.46$; Holdout mean $5.75 \pm 1.17$.*

Grouping by title family (P2) reduces $R^2$ by 0.0131 for the full model. Crucially, this performance drop occurs exclusively in models that read text: SBERT ridge loses 0.0125 and SBERT+XGBoost loses 0.0127, whereas metadata OLS is unaffected ($+0.0001$). Franchise leakage is entirely textual.

Enforcing chronological separation (P3) causes catastrophic collapse: the best model falls from $R^2=0.549$ to $R^2=0.309$, with non-overlapping bootstrap intervals. Metadata OLS collapses by 62% of its explanatory power (0.3889 to 0.1473), whereas text-only SBERT loses 35% (0.4114 to 0.2671). The features that fail most under temporal shift are those that have nothing to do with the script.

### B. Paired Statistical Comparisons
Table IV reports paired error comparisons against the full model (`SBERT + meta + XGBoost`).

**TABLE IV.** Paired Wilcoxon Signed-Rank Test Against Full System (`SBERT + meta + XGBoost`).

| Protocol | Baseline Model | $\Delta\text{MAE}$ | 95% CI | $p$-value |
|:---:|:---|:---:|:---:|:---:|
| **P1** | Constant mean | $+0.4485$ | $[+0.427, +0.471]$ | $2.2 \times 10^{-281}$ |
| | Metadata OLS | $+0.1425$ | $[+0.129, +0.157]$ | $1.8 \times 10^{-73}$ |
| | SBERT ridge | $+0.1342$ | $[+0.116, +0.150]$ | $7.0 \times 10^{-55}$ |
| | SBERT + XGBoost | $+0.1079$ | $[+0.093, +0.122]$ | $4.7 \times 10^{-46}$ |
| **P2** | Constant mean | $+0.4456$ | $[+0.424, +0.466]$ | $2.9 \times 10^{-280}$ |
| | Metadata OLS | $+0.1386$ | $[+0.124, +0.152]$ | $2.8 \times 10^{-73}$ |
| | SBERT ridge | $+0.1433$ | $[+0.127, +0.159]$ | $2.5 \times 10^{-61}$ |
| | SBERT + XGBoost | $+0.1136$ | $[+0.100, +0.127]$ | $2.1 \times 10^{-50}$ |
| **P3** | Constant mean | $+0.2530$ | $[+0.195, +0.313]$ | $5.7 \times 10^{-15}$ |
| | Metadata OLS | $+0.1152$ | $[+0.063, +0.166]$ | $2.0 \times 10^{-4}$ |
| | SBERT ridge | $+0.1185$ | $[+0.064, +0.176]$ | $1.3 \times 10^{-4}$ |
| | SBERT + XGBoost | $+0.0236$ | $[-0.020, +0.074]$ | **$0.273$** |

Under P1 and P2, adding metadata to SBERT embeddings yields a decisive error reduction ($\Delta\text{MAE} \approx 0.11$, $p < 10^{-45}$). Under the chronological split P3, however, adding metadata yields an insignificant $\Delta\text{MAE} = 0.0236$ ($p=0.273$, $W=62,734$, matched-pairs $r=+0.056$). The metadata block provides no statistically measurable benefit on truly unseen future films.

### C. Lexical vs. Semantic Baseline Controls
On the 5-fold CV subset where raw text is fully available ($n=5,195$):
- Structural OLS (16 script features + metadata): RMSE $1.076 \pm 0.040$, MAE $0.833 \pm 0.031$, $R^2 = 0.440 \pm 0.007$.
- TF-IDF + XGBoost: RMSE $1.083 \pm 0.034$, MAE $0.819 \pm 0.022$, $R^2 = 0.433 \pm 0.014$.
- SBERT + XGBoost: RMSE $0.958 \pm 0.033$, MAE $0.720 \pm 0.027$, $R^2 = 0.556 \pm 0.009$.

8,000 TF-IDF n-grams match 16 hand-crafted structural statistics within 0.007 $R^2$. SBERT outperforms TF-IDF by $\Delta\text{MAE} = 0.0514$ ($p = 2.6 \times 10^{-5}$). Ensembling across all four model types via Ridge stacking achieves $R^2 = 0.577$, and Bayesian hyperparameter tuning (100-trial TPE) reaches $R^2 = 0.581$. These tuning gains ($+0.032$ $R^2$) are negligible compared to the $-0.240$ $R^2$ drop caused by shifting from random to chronological evaluation. Ensembling moves the third decimal; the split protocol moves the first.

---

## VI. Feature Attribution and Mechanism Analysis

### A. Split Gain vs. Out-of-Sample Utility
In the fitted gradient-boosting model:
- 384 SBERT embedding dimensions account for **55.45%** of total split gain (mean 0.14% per dimension; max 1.68%).
- 3 metadata features account for **36.57%** (`movie_length` 25.50%, `year` 11.01%, `decade_encoded` 0.06%).
- 16 structural statistics account for only **7.98%** (`char_count` 2.41%, `word_count` 1.56%, `line_count` 0.75%).

Length proxies dominate the structural block, while dramaturgical features (`action_density` 0.10%, `dialogue_density` 0.09%, `unique_characters` 0.05%) contribute virtually nothing.

### B. Block Permutation on the Temporal Holdout
Permuting feature blocks across the 515 holdout films (20 iterations; baseline $\text{MAE} = 0.7232$) exposes the mechanics of temporal failure:
- Permuting all metadata: $\Delta\text{MAE} = \mathbf{+0.2168 \pm 0.0250}$
- Permuting `movie_length` alone: $\Delta\text{MAE} = \mathbf{+0.1988 \pm 0.0185}$
- Permuting 384 SBERT embeddings: $\Delta\text{MAE} = \mathbf{+0.1950 \pm 0.0217}$
- Permuting `year`: $\Delta\text{MAE} = \mathbf{+0.0000 \pm 0.0000}$
- Permuting `decade_encoded`: $\Delta\text{MAE} = \mathbf{+0.0000 \pm 0.0000}$

The exact zero increase for `year` and `decade` is structural. Because the model was trained on films up to 2015, all 2020–2025 release years exceed the maximal split thresholds in the trees. Every test instance traverses the identical terminal leaf branch regardless of permutation. A feature commanding 11% of in-sample gain becomes entirely inert at deployment. In contrast, film runtime (`movie_length`) transfers robustly across eras, matching the explanatory capacity of the entire 384-dimensional embedding space.

### C. Limits of Mean Pooling
While embeddings contribute $+0.1950$ MAE under permutation, their utility is diffuse: no dimension exceeds 1.7% gain. Mean pooling across 36 unit-norm vectors compresses script representations toward the corpus centroid, suppressing localized dialogue dynamics and plot pivots. Coupled with 31.6% silent token truncation, the semantic arm operates on a heavily smoothed, lossy view of the screenplay.

---

## VII. Threats to Validity and Limitations

1. *Subset text audit:* Byte SHA-256 deduplication was validated on the 356 distributed raw texts (0 collisions). While metadata keys confirm zero duplicate IDs across all 5,204 records, unobserved text-level duplicates among non-distributed files remain possible.
2. *IMSDb curation bias:* The corpus rating mean declines monotonically from 7.87 in the 1920s to 5.48 in the 2010s, reflecting community upload preferences rather than true historical film quality. Models learning `year` exploit this curation artifact.
3. *Variance compression in temporal holdout:* Rating standard deviation falls from 1.46 (pre-2020) to 1.17 (2020–2025). Since $R^2$ scales with target variance, part of the $R^2$ decline is mathematical; MAE remains flat at $0.7232$.
4. *Deployment availability of runtime:* Runtime is the primary driver of temporal transfer, yet runtime is unknown during pre-production spec-script evaluation. When inference falls back on metadata defaults, deployed performance reverts to the text-only baseline ($R^2=0.267$).
5. *Observational ceiling:* An IMDb rating aggregates casting, direction, marketing, and cultural timing. A screenplay model establishes a lower bound on text predictive power, not an upper bound on film success forecasting.

---

## VIII. Conclusion

Evaluating screenplay-to-rating models under standard shuffled partitions produces illusory predictive confidence. On 5,164 audited IMSDb scripts, random 5-fold CV yields $R^2 = 0.549$; grouping by franchise reduces this to $0.536$; and holding out future release years drops performance to $R^2 = 0.309$. Block permutation demonstrates that calendar features supply 11% of in-sample gain but zero out-of-sample utility, while runtime matches the entire 384-dimensional semantic embedding block. Furthermore, naive word chunking silently discards 31.6% of tokens before encoder ingestion.

For researchers and practitioners building screenplay prediction pipelines, we propose five operational rules:
1. **Group by title family:** Prevent sequel and remake leakage using title-stem grouping.
2. **Evaluate chronologically:** Report performance on future holdout windows to reflect deployment conditions.
3. **Report MAE alongside $R^2$:** Avoid misleading variance-compression artifacts when target distributions drift.
4. **Tokenize before chunking:** Align sliding window budgets to WordPiece token limits rather than whitespace words.
5. **Attach immutable record keys:** Never cache intermediate embeddings by row offset alone.

### Reproducibility
All audit, token verification, evaluation, and plotting code are released in the repository (`audit_corpus.py`, `audit_tokens.py`, `leakage_eval.py`, `analysis_xai.py`). Every table and metric is regenerated with fixed random seed 42 and $B=1,000$ bootstrap iterations.

---

## References

[1] I. Beltagy, M. E. Peters, and A. Cohan, "Longformer: The long-document transformer," *arXiv:2004.05150*, 2020.  
[2] N. Reimers and I. Gurevych, "Sentence-BERT: Sentence embeddings using Siamese BERT-networks," in *Proc. EMNLP*, 2019.  
[3] J. Eliashberg, S. K. Hui, and Z. J. Zhang, "Assessing box office performance using movie scripts: A kernel-based approach," *IEEE Trans. Knowl. Data Eng.*, vol. 26, no. 11, 2014.  
[4] S. D. Hunter, S. Smith, and S. Singh, "Predicting box office from the screenplay: A text analytical approach," West East Institute, 2016.  
[5] A. J. Reagan, L. Mitchell, D. Kiley, C. M. Danforth, and P. S. Dodds, "The emotional arcs of stories are dominated by six basic shapes," *EPJ Data Science*, vol. 5, no. 31, 2016.  
[6] A. Ramakrishna, V. R. Martínez, N. Malandrakis, K. Singla, and S. Narayanan, "Linguistic analysis of differences in portrayal of movie characters," in *Proc. ACL*, 2017.  
[7] M. Shafaei, C. Smailis, I. Kakadiaris, and T. Solorio, "Age suitability rating: Predicting MPAA ratings based on movie dialogues," in *Proc. LREC*, 2020.  
[8] Y. Zhang, M. Shafaei, and T. Solorio, "From none to severe: Predicting severity in movie scripts," in *Findings of EMNLP*, 2021.  
[9] E. Chu, D. Roy, and J. Glass, "Audio-visual sentiment analysis for learning emotional arcs in movies," in *Proc. ICCV*, 2017.  
[10] R. Sharda and D. Delen, "Predicting box-office success of motion pictures with neural networks," *Expert Systems with Applications*, vol. 30, no. 2, 2006.  
[11] C. T. Madongo, Z. Tang, and J. Hassan, "Movie box-office revenue prediction model by mining deep features from trailers using recurrent neural networks," *J. Adv. Inf. Technol.*, 2024.  
[12] A. Bhadrashetty and S. Patil, "Movie success and rating prediction using data mining," *J. Sci. Res. Technol.*, 2024.  
[13] M. Z. Naeem, F. Rustam, A. Mehmood, I. Ashraf, and G. S. Choi, "Classification of movie reviews using term frequency-inverse document frequency and optimized machine learning algorithms," *PeerJ Comput. Sci.*, 2022.  
[14] I. Chalkidis, X. Dai, M. Fergadiotis, P. Malakasiotis, and D. Elliott, "An exploration of hierarchical attention transformers for efficient long document classification," *arXiv:2210.05529*, 2022.  
[15] S. Asur and B. A. Huberman, "Predicting the future with social media," in *Proc. IEEE/WIC/ACM Web Intelligence*, 2010.  
[16] A. Oghina, M. Breuss, M. Tsagkias, and M. de Rijke, "Predicting IMDb movie ratings using social media," in *Proc. ECIR*, 2012.  
[17] G. Mishne and N. Glance, "Predicting movie sales from blogger sentiment," in *AAAI Spring Symposium*, 2006.  
[18] M. Mestyán, T. Yasseri, and J. Kertész, "Early prediction of movie box office success based on Wikipedia activity big data," *PLoS ONE*, vol. 8, no. 8, 2013.  
[19] H. L. Vogel, *Entertainment Industry Economics*, 10th ed. Cambridge, U.K.: Cambridge Univ. Press, 2020.  
[20] W. R. Bristi, Z. Zaman, and N. Sultana, "Predicting IMDb rating of movies by machine learning techniques," in *Proc. ICCCNT*, 2019.  
[21] A. L. Gomes, M. Ribeiro, and P. Quaresma, "Predicting IMDb rating of TV series with deep learning: The case of Arrow," *arXiv:2211.03388*, 2022.  
[22] J. Ramos, A. Pereira, and L. Cruz, "Movie rating prediction using sentiment features," in *Proc. SALLD*, 2022.  
[23] V. Udandarao, A. Gupta, and S. Bhattacharya, "Movie revenue prediction using machine learning models," *arXiv:2405.11651*, 2024.  
[24] W. Xie, Y. Liu, and H. Chen, "Predicting movie success with multi-task learning: GPT-based sentiment and SIR propagation," *arXiv preprint*, 2025.  
[25] E. Mohamed, M. Shafaei, and T. Solorio, "A first dataset for film age appropriateness investigation," in *Proc. LREC*, 2020.  
[26] S. Kar, S. Maharjan, and T. Solorio, "Folksonomication: Predicting tags for movies from plot synopses using emotion flow encoded neural network," in *Proc. COLING*, 2018.  
[27] M. C. Chiu, S. Lin, and W. Chen, "Screenplay quality assessment: Can we predict who gets nominated?" Nat. Univ. Singapore, Tech. Rep., 2020.  
[28] K. Gorman and S. Bedrick, "We need to talk about standard splits," in *Proc. ACL*, 2019.  
[29] A. Søgaard, S. Ebert, J. Bastings, and K. Filippova, "We need to talk about random splits," in *Proc. EACL*, 2021.  
[30] P. Lewis, P. Stenetorp, and S. Riedel, "Question and answer test-train overlap in open-domain question answering datasets," in *Proc. EACL*, 2021.  
[31] A. Elangovan, J. He, and K. Verspoor, "Memorization vs. generalization: Quantifying data leakage in NLP performance evaluation," in *Proc. EACL*, 2021.  
[32] S. Kapoor and A. Narayanan, "Leakage and the reproducibility crisis in machine-learning-based science," *Patterns*, vol. 4, no. 9, 2023.  
[33] A. Lazaridou *et al.*, "Mind the gap: Assessing temporal generalization in neural language models," in *Proc. NeurIPS*, 2021.  
[34] E. M. Bender and B. Friedman, "Data statements for natural language processing," *TACL*, vol. 6, 2018.  
[35] J. Dodge *et al.*, "Documenting large webtext corpora: A case study on the Colossal Clean Crawled Corpus," in *Proc. EMNLP*, 2021.
