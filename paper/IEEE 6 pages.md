# What Survives a Strict Split? An Empirical Audit of Long-Document Screenplay Modelling for IMDb Rating Prediction

**Anonymous authors**

***Abstract**—Screenplay-to-rating regression is usually reported on a random split of an archival corpus. We argue that this number is not the number a reader wants, and we measure the gap. Working from a corpus of 5,204 IMSDb screenplays joined to IMDb metadata, we first run a record-identity audit and publish the resulting exclusion ledger, which retains 5,164 records. We then quantify a defect that is invisible at the API surface: the pipeline slides a 256-*word* window over each script, but the encoder admits only 254 WordPiece *tokens*, so 98.45% of windows are cut and 31.60% of all tokens never reach the model. Finally we evaluate identical representations and learners under three partitions — random $k$-fold, title-family GroupKFold, and a chronological 2020–2025 holdout. The best system reports $R^{2}=0.549$ under random folds, 0.536 once franchises are held together, and 0.309 when the test set lies in the future. Release year and decade, which jointly supply 36.6% of in-sample split gain, contribute exactly $\Delta\mathrm{MAE}=0.000$ once the test years fall outside the training range, while runtime alone (+0.199) matches the entire 384-dimensional sentence-embedding block (+0.195). Semantic embeddings survive the strict split; the calendar confound does not.*

***Index Terms**—data leakage, evaluation protocol, sentence embeddings, long-document NLP, screenplay analysis, temporal generalisation, grouped cross-validation.*


## I. Introduction

A feature-film screenplay runs to roughly 25,000 words. Sentence encoders read 256 tokens. The usual answer — slide a window, embed each window, average the results — is cheap and widely used. What is far less often examined is whether the evaluation wrapped around that pipeline measures anything a practitioner could rely on, because an archival screenplay corpus assembled from a community repository carries two structural hazards that a shuffled train/test split will silently convert into apparent skill.

The first hazard is duplication. Franchises share vocabulary and structure across installments, and remakes reuse plot and dialogue wholesale. Under a random partition, a near-copy in the training set makes its twin in the test set trivially predictable. The second hazard is time. A model given a release-year feature can recover a large share of target variance from the calendar alone, without reading a single line of dialogue. That shortcut evaporates the moment the test films are drawn from years the model never saw, which is the only setting in which a screenplay-stage predictor would actually be deployed.

Layered underneath both is a quieter defect: chunking is almost always specified in words; truncation always happens in subword tokens, and nobody is told when the budget is exceeded. This paper is an audit, not a leaderboard entry. We take an existing SBERT-plus-gradient-boosting pipeline and ask what its reported performance is actually made of. Three contributions follow.

- **C1 — A record-identity protocol and its exclusion ledger.** We define film identity through four keys — content SHA-256 digest, IMDb identifier, title$\times$year, and a normalised title-family stem — audit all 5,204 joined records against each, and report every exclusion (Section III).
- **C2 — A measurement of silent truncation.** Running the encoder's own tokeniser over the pipeline's own windows, we find that 98.45% of 256-word windows exceed the 254-token content budget and that 31.60% of WordPiece tokens are discarded before the encoder sees them (Section IV-B).
- **C3 — A three-protocol evaluation with paired inference.** Holding representation and learner fixed, we vary only the partition: random $k$-fold, title-family GroupKFold, and a chronological 2020–2025 holdout. We report bootstrap intervals, paired Wilcoxon signed-rank tests with matched-pairs effect sizes, and a block-permutation attribution that isolates which feature families survive temporal extrapolation (Sections V and VI).

We do not claim a new state of the art. We claim that the difference between $R^{2}=0.549$ and $R^{2}=0.309$ on the same corpus, the same features, and the same learner is the finding.


## II. Related Work

**Screenplays and film outcomes.** Early work treated the script as a feature source for commercial forecasting, predicting return on investment or box-office revenue from genre, content, and semantic descriptors [3, 4]. A second strand reads narrative structure rather than commercial outcome: emotional arc shapes [5, 9], character portrayal [6], MPAA age-suitability and severity rating [7, 25, 8], and tag assignment from plot synopses [26]. Screenplay quality assessment in the awards-nomination framing [27] is the closest analogue to rating regression, and it shares our target's central difficulty: the label is an aggregate social judgement, only partly a function of the text. Outside the script itself, forecasting has drawn on production metadata [10], social-media and Wikipedia activity [15, 16, 17, 18], trailer content [11], and review sentiment [13, 22]; IMDb rating regression from tabular and textual metadata is by now a standard applied exercise [12, 20, 21, 23, 24]. The reported figures span a wide range, and the partitions behind them are rarely specified beyond a random seed. Vogel's industry account [19] is a useful corrective on the ceiling: theatrical outcomes are dominated by release strategy, competition, and marketing spend, none of which is legible in the script.

**Representations for long documents.** Sentence-BERT [2] produces dense sentence-level embeddings from a Siamese objective and is the encoder we audit. Its native context is short, so document-level use requires either hierarchical attention [14], sparse attention over longer sequences [1], or chunk-and-pool aggregation. Chunk-and-pool is the pragmatic default: it needs no fine-tuning, runs on CPU, and scales linearly. Its cost is that the aggregation function is fixed rather than learned, and — as we measure in Section IV-B — that the chunk boundary is usually specified in a unit the encoder does not use.

**Leakage and split design.** The methodological literature we lean on is not about films. System rankings reorder under alternative random splits, and split designs should stress the generalisation actually claimed [28, 29]; train–test overlap has been measured and formalised for NLP benchmarks [30, 31], and leakage taxonomies across disciplines argue for pre-registration of the split [32]. Temporal generalisation has its own line [33], as does documentation of corpus provenance [34, 35]. Applying these instruments to a screenplay corpus is, to our knowledge, new.


## III. Corpus Integrity and Exclusion Audit


### A. Source and identity keys

The corpus joins full screenplay texts scraped from the Internet Movie Script Database (IMSDb) to IMDb metadata on a per-film basis. Each of the 5,204 joined records carries a title, release year, decade bin, IMDb identifier, runtime in minutes, an IMDb rating on the 1.0–10.0 scale at one-decimal resolution, and a path to the script file. Ratings have mean 5.98, median 5.80, and standard deviation 1.44.

Deduplication is only as good as the notion of identity behind it, so we audit four keys of decreasing strictness. *Content digest:* for a script file with byte content $b$, the key is $\mathrm{SHA256}(b)$, which catches the re-upload-under-a-new-name case that no metadata key can see; collisions are resolved by retaining the earliest record in load order. *IMDb identifier:* the canonical film key. *Title$\times$year:* a fallback for records whose identifier is missing or malformed. *Title family:* the loosest key, and the one that matters for split design. We normalise a title $t$ by case-folding, stripping non-alphanumeric characters, removing a leading article, and iteratively deleting a trailing sequel marker — an Arabic numeral, a Roman numeral up to $\mathrm{X}$, or a `part` $n$ construction — to a fixed point. *Toy Story* and *Toy Story 2* therefore map to the single family `toy story`, the grouping variable for Section IV-D.


### B. The exclusion flow

Let $\mathcal{R}_0$ be the joined record set, $\vert \mathcal{R}_0\vert = 5{,}204$. The pipeline is a composition of four filters applied in a fixed order, each defined by a predicate on a record $r$:

$$
\pi_{\text{len}}(r) \equiv \vert \mathrm{text}(r)\vert \ge 1000 \ \text{characters}, \tag{1}
$$

$$
\pi_{\text{hash}}(r) \equiv \mathrm{SHA256}\bigl(\mathrm{bytes}(r)\bigr) \notin \mathcal{H}_{<r}, \tag{2}
$$

$$
\pi_{\text{year}}(r) \equiv \mathrm{yr}(r) \in [1900,\, 2025], \tag{3}
$$

$$
\pi_{\text{id}}(r) \equiv \text{positional identity of } r \text{ is unique}, \tag{4}
$$

where $\mathcal{H}_{<r}$ is the set of digests already seen in load order. The surviving sets are nested, $\mathcal{R}_{k} = \{\, r \in \mathcal{R}_{k-1} : \pi_k(r) \,\}$, and the realised cardinality chain on this corpus is

$$
5{,}204 \;\xrightarrow{\ \pi_{\text{len}}\ } 5{,}195 \;\xrightarrow{\ \pi_{\text{hash}}\ } 5{,}195 \;\xrightarrow{\ \pi_{\text{year}}\ } 5{,}191 \;\xrightarrow{\ \pi_{\text{id}}\ } 5{,}164 . \tag{5}
$$

The second arrow is an identity map *on the subset over which the digest could be evaluated*, not a corpus-wide null result: content hashing requires the raw text, and 356 of the 5,204 files are distributed with the code. The metadata keys, computable for all records, bound — without eliminating — the residual duplication that a full-corpus pass could still uncover.


### C. Audit results

**TABLE I.** Corpus integrity audit and exclusion ledger. Percentages are of the 5,204 joined records.

| **Check / exclusion** | **Count** | **%** |
|---|---|---|
| Joined records (IMSDb $\bowtie$ IMDb) | 5,204 | 100.00 |
| Distinct IMDb identifiers | 5,204 | 100.00 |
| Duplicate title$\times$year pairs | 0 | 0.00 |
| Title collisions (same title, other year) | 80 / 165 rec. | 3.17 |
| SHA-256 duplicate digests (audited subset) | 0 | — |
| Script text $<1{,}000$ characters | 9 | 0.17 |
| Release year outside [1900, 2025] | 4 | 0.08 |
| Unresolvable positional identity | 27 | 0.52 |
| **Analysis corpus** | 5,164 | **99.23** |
| Title families / multi-member | 4,979 / 164 | — |

Table I is the full ledger for the flow of Section III-B. All 5,204 identifiers are distinct and no title$\times$year pair repeats. This is a genuinely clean join, and it means the leakage channel in this corpus is not naive record duplication. Eighty titles appear more than once, spanning 165 records, but every collision pairs distinct identifiers and distinct years — these are remakes and same-title unrelated films, not duplicates. We retain them in a shared family. Four records carry release years of 1080 or 1088, transparently mis-typed 1980 and 1988; we drop them rather than silently repair them. The derived decade column is worse: the shipped label encoder holds 14 decade strings, among them `1080s` and `1996s`, encoded as an *ordinal* integer.

Twenty-seven further records are excluded for a reproducibility reason. The archived embedding matrix stores rows in loader order without a record key, so row $i$ of the matrix is not record $i$ of the spreadsheet. We reconstruct the offset function from 1,559 anchors in the committed split bookkeeping; twenty-seven positions inside the reconstruction gaps admit more than one consistent assignment, and we drop them rather than guess.


## IV. Methodology and Representations


### A. Structural screenplay statistics

Sixteen statistics are computed from the raw, uncleaned script text: six size measures (`char_count`, `word_count`, `line_count`, `sentence_count`, `scene_count`, `words_per_scene`); three describing lexical sophistication (`avg_word_length`, `unique_word_ratio`, `long_word_ratio`); two describing sentence geometry (`avg_sentence_length`, `sentence_length_std`); two describing dialogue (`dialogue_density`, the share of lines that match an ALL-CAPS speaker-cue pattern, and `unique_characters`); two emotional proxies (`exclamation_ratio`, `question_ratio`); and one stage-direction proxy (`action_density`). Two of the sixteen are close to degenerate on this corpus. Mean `dialogue_density` is 0.0087 with standard deviation 0.0361, and mean `unique_characters` is 3.41. A feature film does not have three speaking parts. The speaker-cue regular expression assumes a `NAME:` convention that most IMSDb transcripts do not follow, so both features encode transcription format rather than dramaturgy.


### B. Token-aligned chunking and mean pooling

Let a screenplay be a word sequence $D = (w_1, \dots, w_N)$ obtained by whitespace splitting after light normalisation that removes bracketed stage directions, ALL-CAPS speaker cues, `INT.`/`EXT.` markers, and timestamps. With window length $L = 256$ and overlap $o = 50$, the stride is $s = L - o = 206$, and the $j$-th window is

$$
C_j \;=\; \bigl(w_{(j-1)s+1}, \dots, w_{\min\{(j-1)s+L,\; N\}}\bigr), \tag{6}
$$

for $j = 1, \dots, M$ with $M = 1$ if $N \le L$ and $M = \lceil (N - L)/s \rceil + 1$ otherwise. Each window is encoded by $f: \mathcal{C} \to \mathbb{R}^{d}$ with $d = 384$, and the document representation is the arithmetic mean

$$
e(D) \;=\; \frac{1}{M} \sum_{j=1}^{M} f(C_j) . \tag{7}
$$

Because the encoder L2-normalises its output, every $f(C_j)$ lies on the unit sphere, so

$$
\bigl\Vert e(D) \bigr\Vert _2^2 = \frac{1}{M^2}\Bigl( M + 2\!\!\sum_{1 \le i < j \le M}\!\! \langle f(C_i), f(C_j)\rangle \Bigr), \tag{8}
$$

which equals 1 only when all windows embed identically and falls toward $1/\sqrt{M}$ as they decorrelate. With a median of 36 windows per script, no scene survives pooling as a distinguishable signal.

**The word/token mismatch.** The encoder's transformer accepts $T_{\max} = 256$ positions, two of which are consumed by `[CLS]` and `[SEP]`. The content budget is therefore $B = T_{\max} - 2 = 254$ WordPiece tokens, while $C_j$ is defined by a count of 256 *words*. Writing $\tau(\cdot)$ for the tokeniser, the encoder in fact sees $\tau(C_j)$ truncated to its first $B$ entries, and the discarded fraction for a document is

$$
\rho(D) \;=\; 1 - \frac{\sum_{j=1}^{M} \min\bigl\{\vert \tau(C_j)\vert ,\, B\bigr\}} {\sum_{j=1}^{M} \vert \tau(C_j)\vert } . \tag{9}
$$

Running the encoder's own tokeniser over the pipeline's own windows on the 356 available scripts yields 14,558 windows with median length 371 tokens, 90th percentile 405, and maximum 613. The consequences: 98.45% of windows exceed $B$, and $\rho$ evaluated corpus-wide is 0.3160, with a worst case of 0.4443. Roughly one word in three is written to a window that the encoder never reads. Nothing in the pipeline reports this.


### C. Lexical control and learners

The lexical control is TF-IDF over unigrams and bigrams, capped at 8,000 features, with English stop words removed, minimum document frequency 3, maximum document frequency 0.85, and sublinear term frequency $1 + \log \mathrm{tf}$. It is fed to the same gradient-boosted regressor as the semantic arm. This pairing is deliberate: SBERT and TF-IDF differ in representation and in nothing else, so their gap is attributable to semantics rather than to the learner.

The regressor throughout is gradient-boosted trees with 500 rounds, learning rate 0.05, maximum depth 6, histogram splitting, and early stopping after 20 rounds without validation improvement; predictions are clipped to [1, 10]. Two linear probes provide reference points: ordinary least squares on the three metadata columns, and ridge regression ($\alpha = 1.5$) on standardised embeddings. A constant-mean predictor anchors the bottom. For the stacked ensemble, a ridge meta-regressor is fitted over four base models — metadata OLS, structural OLS, TF-IDF+XGBoost, SBERT+XGBoost — using inner five-fold out-of-fold meta-features nested within each outer fold.


### D. Partition protocols

Let $\mathcal{I} = \{1, \dots, n\}$ index the analysis corpus, $n = 5{,}164$.

**P1, random.** A shuffled $K$-fold partition, $K = 5$: disjoint $F_1, \dots, F_K$ with $\bigcup_k F_k = \mathcal{I}$, assignment independent of every record attribute.

**P2, grouped.** Let $\gamma: \mathcal{I} \to \mathcal{G}$ map each record to its title family, $\vert \mathcal{G}\vert = 4{,}979$. GroupKFold constructs disjoint folds subject to the group constraint

$$
\forall g \in \mathcal{G}\;\; \exists!\, k \in \{1,\dots,K\}:\; \gamma^{-1}(g) \subseteq F_k , \tag{10}
$$

so no family is split across the boundary. Because 4,979 families cover 5,164 records, the constraint is nearly free: realised fold sizes are 1,032–1,033, identical to P1.

**P3, chronological.** With $\mathrm{yr}(i)$ the release year, the test set is $2020 \le \mathrm{yr}(i) \le 2025$, validation $c \le \mathrm{yr}(i) \le 2019$, and training $\mathrm{yr}(i) < c$, where $c$ is the 85th percentile of the pre-2020 year distribution, here $c = 2016$. Early stopping is thereby decided on films that are themselves in the future relative to the training set, so no forward-looking information enters through model selection. The realised sizes are 3,918 / 731 / 515.

A partition's leakage can be measured directly. Define the family-overlap rate of a test fold as the share of its records whose family also occurs in the corresponding training pool,

$$
\lambda(F_k) = \frac{1}{\vert F_k\vert } \bigl\vert \{\, i \in F_k : \gamma(i) \in \gamma(\mathcal{I} \setminus F_k) \,\}\bigr\vert . \tag{11}
$$

Averaged over folds, $\lambda = 5.50\%$ under P1 and, by construction, $\lambda = 0$ under P2. The chronological holdout sits at $\lambda = 6.41\%$ — a sequel released in 2021 may well have a parent film in the training years.


### E. Inference

Point estimates carry percentile bootstrap intervals from $B = 1000$ resamples of the test set. Model comparison is paired and distribution-free. For predictions $\hat{y}^{A}$, $\hat{y}^{B}$ on a common test set, form per-record absolute errors $e^{A}_i = \vert \hat{y}^{A}_i - y_i\vert$ and $e^{B}_i$ likewise, take $d_i = e^{A}_i - e^{B}_i$, discard the $n_0$ exact ties, rank the remaining $\vert d_i\vert$, and accumulate

$$
R^{+} = \!\!\sum_{i:\, d_i > 0}\!\! \mathrm{rank}(\vert d_i\vert ), \qquad R^{-} = \!\!\sum_{i:\, d_i < 0}\!\! \mathrm{rank}(\vert d_i\vert ) , \tag{12}
$$

with test statistic $W = \min(R^{+}, R^{-})$. The signed-rank test is non-parametric and has no degrees-of-freedom parameter; the reporting quantities are the number of pairs $n$, the number of non-tied pairs $n - n_0$, the statistic $W$, and the matched-pairs rank-biserial effect size

$$
r \;=\; \frac{R^{+} - R^{-}}{R^{+} + R^{-}} \;\in\; [-1, 1] , \tag{13}
$$

which we report alongside $p$ so that a small effect at large $n$ is not mistaken for a large one. Differences in mean absolute error carry their own paired bootstrap intervals.


## V. Experimental Results


### A. The cost of a stricter partition

Table II isolates the cost of the group constraint; Table III isolates the cost of the temporal one. Five systems, one corpus, one learner configuration throughout.

**TABLE II.** Grouped five-fold results. P1 shuffles records; P2 enforces the title-family constraint (10). Pooled out-of-fold point estimates over 5,164 records. $\Delta R^{2}$ is the explained variance lost to the constraint, i.e. the share of P1 skill that franchise overlap was supplying.

|  | **P1 random $k$-fold** ($\lambda = 5.50\%$) | **P2 grouped by family** ($\lambda = 0\%$) |  |  |  |  |  |
|---|---|---|---|---|---|---|---|
| **System** | RMSE | MAE | $R^{2}$ | RMSE | MAE | $R^{2}$ | $\Delta R^{2}$ |
| Constant mean | 1.4373 | 1.1761 | -0.0000 | 1.4375 | 1.1765 | -0.0004 | -0.0004 |
| Metadata OLS | 1.1237 | 0.8701 | 0.3888 | 1.1236 | 0.8695 | 0.3889 | +0.0001 |
| SBERT ridge | 1.1122 | 0.8618 | 0.4012 | 1.1238 | 0.8742 | 0.3887 | -0.0125 |
| SBERT + XGBoost | 1.0907 | 0.8355 | 0.4241 | 1.1026 | 0.8445 | 0.4114 | -0.0127 |
| SBERT + meta + XGBoost | **0.9652** | **0.7276** | **0.5490** | **0.9791** | **0.7309** | **0.5359** | -0.0131 |
| Best system: P1 $R^{2}\in[0.5298,0.5671]$, P2 $R^{2}\in[0.5162,0.5537]$. Every arm that reads text loses $R^{2}$; the metadata-only probe does not. |  |  |  |  |  |  |  |

**TABLE III.** Temporal holdout. Fit on films released up to 2015, early stopping on 2016–2019, tested on the 515 films released 2020–2025.

| **System** | RMSE | MAE | $R^{2}$ |
|---|---|---|---|
| Constant mean | 1.2230 | 0.9762 | -0.0947 |
| Metadata OLS | 1.0794 | 0.8384 | 0.1473 |
| SBERT ridge | 1.0719 | 0.8417 | 0.1591 |
| SBERT + XGBoost | 1.0007 | 0.7469 | 0.2671 |
| SBERT + meta + XGBoost | **0.9717** | **0.7232** | **0.3089** |
| Best system $R^{2}\in[0.1985,0.4038]$; $\lambda = 6.41\%$. |  |  |  |
| Target: train $6.0119 \pm 1.4619$, holdout $5.7546 \pm 1.1701$. |  |  |  |

Grouping costs little. The best system moves from $R^{2} = 0.5490$ to 0.5359, a drop of 0.0131, and the two bootstrap intervals overlap substantially. That is the expected result for a corpus in which only 164 families have more than one member and the largest has four. The interesting part is that the effect is consistently negative across every system that reads text — SBERT ridge loses 0.0125 $R^{2}$, SBERT+XGBoost loses 0.0127 — while the metadata-only probe is unmoved ($0.3888 \to 0.3889$). Franchise leakage is a *textual* channel, and a metadata model cannot exploit it because it never sees the words.

Chronology costs a great deal. The same best system falls to $R^{2} = 0.3089$, and the interval [0.1985, 0.4038] no longer overlaps either in-corpus estimate. The constant-mean predictor goes from $R^{2} \approx 0$ to -0.0947, which is a pure distribution-shift effect: the training mean (6.11) is simply wrong for the holdout (5.75). And the metadata probe collapses hardest of all, from 0.3889 to 0.1473, losing 62% of its explanatory power, while the text-only SBERT model loses 35% ($0.4114 \to 0.2671$). The representation that generalises worst across time is the one that was never about the film.


### B. Paired comparisons

Table IV reports every baseline against the best system on matched records.

**TABLE IV.** Paired comparison against SBERT + metadata + XGBoost. $\Delta\mathrm{MAE}$ is positive when the reference system has lower error. $p$ from the two-sided Wilcoxon signed-rank test.

| **Prot.** | **Baseline** | $\Delta\mathrm{MAE}$ | **95% CI** | $p$ |
|---|---|---|---|---|
| P1 | Constant mean | +0.4485 | [+0.427, +0.471] | $2.2\!\times\!10^{-281}$ |
|  | Metadata OLS | +0.1425 | [+0.129, +0.157] | $1.8\!\times\!10^{-73}$ |
|  | SBERT ridge | +0.1342 | [+0.116, +0.150] | $7.0\!\times\!10^{-55}$ |
|  | SBERT + XGB | +0.1079 | [+0.093, +0.122] | $4.7\!\times\!10^{-46}$ |
| P2 | Constant mean | +0.4456 | [+0.424, +0.466] | $2.9\!\times\!10^{-280}$ |
|  | Metadata OLS | +0.1386 | [+0.124, +0.152] | $2.8\!\times\!10^{-73}$ |
|  | SBERT ridge | +0.1433 | [+0.127, +0.159] | $2.5\!\times\!10^{-61}$ |
|  | SBERT + XGB | +0.1136 | [+0.100, +0.127] | $2.1\!\times\!10^{-50}$ |
| P3 | Constant mean | +0.2530 | [+0.195, +0.313] | $5.7\!\times\!10^{-15}$ |
|  | Metadata OLS | +0.1152 | [+0.063, +0.166] | $2.0\!\times\!10^{-4}$ |
|  | SBERT ridge | +0.1185 | [+0.064, +0.176] | $1.3\!\times\!10^{-4}$ |
|  | SBERT + XGB | +0.0236 | [-0.020, +0.074] | 0.273 |

The final row is the one that changes the story. Under P1 and P2, adding the three metadata columns to the embedding buys a highly significant 0.108–0.114 rating points of mean absolute error. Under P3 the same addition buys 0.0236 points, with an interval straddling zero and $p = 0.273$; the full statistics are $W = 62{,}734$ over $n = 515$ pairs with no ties, rank-biserial $r = +0.056$. An effect size of 0.056 is negligible by any convention. The metadata block, worth two orders of magnitude in $p$-value under a random split, is not measurably useful when the test films are genuinely unseen.


### C. The lexical and structural arms

The lexical control and the structural-feature model can be evaluated only under the random protocol on this corpus, because reconstructing either representation requires the raw script text; the figures below are from archived 5-fold runs over 5,195 records and are *not* comparable to Tables II– III. Structural OLS reaches $R^{2} = 0.440 \pm 0.007$ and TF-IDF+XGBoost $0.433 \pm 0.014$ — 8,000 TF-IDF weights buy almost exactly what sixteen hand-counted statistics buy. The semantic arm beats the lexical one by $\Delta\mathrm{MAE} = 0.0514$ [0.034, 0.067], $p = 2.6\times10^{-5}$, a real but modest margin. Inverse-frequency sample weighting, applied in the original pipeline to compensate for the rating imbalance, *hurts*: removing it improves MAE by 0.0482 [0.039, 0.057] at $p = 2.5\times10^{-26}$. The nested ridge stack ($R^{2} = 0.577 \pm 0.010$) improves on its own strongest base by $\Delta\mathrm{MAE} = 0.0134$ [0.008, 0.019], $p = 4.2\times10^{-6}$, and a 100-trial TPE search on a single 80/20 split over the boosting head reaches test $R^{2} = 0.5811$ [0.5471, 0.6182]. Both gains are real and both are small next to the 0.240 $R^{2}$ that separates the random and chronological protocols. Ensembling and tuning move the third decimal; the split moves the first.


### D. Temporal shift in detail

The holdout target is compressed relative to the training pool: standard deviation falls from 1.4619 to 1.1701, while the mean falls from 6.0119 to 5.7546. Part of the $R^{2}$ collapse is therefore arithmetic — the same absolute error against a narrower target yields a smaller $R^{2}$ — and part is genuine degradation. Mean absolute error separates them: it is 0.7276 under P1 and 0.7232 under P3, statistically indistinguishable. What collapses is how much of the (now narrower) variance it can order. Regressing prediction on truth over the holdout gives slope 0.4634 $(\mathrm{s.e.}\ 0.0278)$ and intercept 3.1745, against an ideal slope of 1. The system is heavily shrunk toward the centre — it identifies which films are above or below average and declines to commit to how far.


## VI. What the Features Are Doing


### A. In-sample gain is not out-of-sample use

In the shipped model's total split gain, the 384 embedding dimensions take 55.5%, the three metadata columns take 36.6%, and the sixteen structural statistics take 8.0%. Per dimension the picture inverts: the single largest embedding dimension contributes 1.68% and the mean dimension 0.14%, whereas `movie_length` alone contributes 25.50% and `year` 11.01%. One tabular column outweighs all sixteen structural statistics by a factor of three.


### B. Block permutation on the temporal holdout

Gain is computed on training splits and therefore inherits every in-sample advantage. Permutation on the holdout does not. We shuffle each feature block across the 515 holdout films, re-predict, and record the increase in MAE over 20 repetitions; the baseline is $\mathrm{MAE} = 0.7232$:

- permuting all three metadata columns: $+0.2168 \pm 0.0250$;
- permuting `movie_length` alone: $+0.1988 \pm 0.0185$;
- permuting the entire 384-dimensional embedding: $+0.1950 \pm 0.0217$;
- permuting `year`: $+0.0000 \pm 0.0000$;
- permuting `decade_encoded`: $+0.0000 \pm 0.0000$.

The two zeros are not rounding. Every holdout film was released in 2020–2025, and the model was fitted on films up to 2015, so every holdout value of `year` lies beyond the largest training split threshold. A tree ensemble routes all of them down the same branch; shuffling values that already share a leaf changes nothing. The feature carrying 11% of in-sample gain becomes a constant at deployment.

Runtime is the opposite case. It transfers cleanly, because a 130-minute film in 2023 occupies the same region of feature space as a 130-minute film in 1994, and it alone reproduces the predictive value of the entire semantic block. That the embedding block contributes as much as it does, after 31.6% truncation and the smoothing of Eq. (8), is the more surprising half of the result.


## VII. Threats to Validity and Limitations

TF-IDF and the structural statistics require raw text and were evaluated only under P1. Selection effects in IMSDb are severe and not correctable: the mean rating falls monotonically from 7.87 in the 1920s to 5.48 in the 2010s, which is a statement about who uploads what, not about film quality, and P3 is the protocol that refuses to reward it. The holdout is one window; rolling-origin evaluation would estimate the decay rate. A screenplay-stage predictor does not know the runtime, because the film has not been cut — or shot. Finally, we audit only mean-pooled `all-MiniLM-L6-v2`; holding the representation fixed is what makes the protocol comparison clean.


## VIII. Conclusion

We audited a screenplay-to-rating pipeline instead of extending it. The corpus turned out to be clean at the record level, but to carry 164 multi-member title families, four corrupt release years, and a corrupt ordinal decade vocabulary that reaches the feature matrix intact. The chunking configuration turned out to discard 31.60% of WordPiece tokens without warning. And the headline metric turned out to depend on the partition far more than on the model: $R^{2} = 0.549$ under random folds, 0.536 once franchises are kept together, 0.309 when the test films are drawn from years the model never saw. Runtime transfers; semantics transfer; the calendar does not. For anyone building a screenplay-stage predictor, the operational recommendations are narrow and concrete: group by title family, hold out by release year, report MAE next to $R^{2}$ when the target distribution shifts, align the chunk to the tokeniser rather than to whitespace, and store an immutable record key beside every cached representation.


##  Reproducibility

All audit and evaluation code is released with the paper; every number is regenerated from the committed JSON artifacts. The random seed is 42 throughout.


## References

[1] I. Beltagy, M. E. Peters, and A. Cohan, “Longformer: The long-document transformer,” *arXiv:2004.05150*, 2020.
[2] N. Reimers and I. Gurevych, “Sentence-BERT: Sentence embeddings using Siamese BERT-networks,” in *Proc. EMNLP*, 2019.
[3] J. Eliashberg, S. K. Hui, and Z. J. Zhang, “Assessing box office performance using movie scripts: A kernel-based approach,” *IEEE Trans. Knowl. Data Eng.*, vol. 26, no. 11, 2014.
[4] S. D. Hunter, S. Smith, and S. Singh, “Predicting box office from the screenplay: A text analytical approach,” West East Institute, 2016.
[5] A. J. Reagan, L. Mitchell, D. Kiley, C. M. Danforth, and P. S. Dodds, “The emotional arcs of stories are dominated by six basic shapes,” *EPJ Data Science*, vol. 5, no. 31, 2016.
[6] A. Ramakrishna, V. R. Martínez, N. Malandrakis, K. Singla, and S. Narayanan, “Linguistic analysis of differences in portrayal of movie characters,” in *Proc. ACL*, 2017.
[7] M. Shafaei, C. Smailis, I. Kakadiaris, and T. Solorio, “Age suitability rating: Predicting MPAA ratings based on movie dialogues,” in *Proc. LREC*, 2020.
[8] Y. Zhang, M. Shafaei, and T. Solorio, “From none to severe: Predicting severity in movie scripts,” in *Findings of EMNLP*, 2021.
[9] E. Chu, D. Roy, and J. Glass, “Audio-visual sentiment analysis for learning emotional arcs in movies,” in *Proc. ICCV*, 2017.
[10] R. Sharda and D. Delen, “Predicting box-office success of motion pictures with neural networks,” *Expert Systems with Applications*, vol. 30, no. 2, 2006.
[11] C. T. Madongo, Z. Tang, and J. Hassan, “Movie box-office revenue prediction model by mining deep features from trailers using recurrent neural networks,” *J. Adv. Inf. Technol.*, 2024.
[12] A. Bhadrashetty and S. Patil, “Movie success and rating prediction using data mining,” *J. Sci. Res. Technol.*, 2024.
[13] M. Z. Naeem, F. Rustam, A. Mehmood, I. Ashraf, and G. S. Choi, “Classification of movie reviews using term frequency-inverse document frequency and optimized machine learning algorithms,” *PeerJ Comput. Sci.*, 2022.
[14] I. Chalkidis, X. Dai, M. Fergadiotis, P. Malakasiotis, and D. Elliott, “An exploration of hierarchical attention transformers for efficient long document classification,” *arXiv:2210.05529*, 2022.
[15] S. Asur and B. A. Huberman, “Predicting the future with social media,” in *Proc. IEEE/WIC/ACM Web Intelligence*, 2010.
[16] A. Oghina, M. Breuss, M. Tsagkias, and M. de Rijke, “Predicting IMDb movie ratings using social media,” in *Proc. ECIR*, 2012.
[17] G. Mishne and N. Glance, “Predicting movie sales from blogger sentiment,” in *AAAI Spring Symposium*, 2006.
[18] M. Mestyán, T. Yasseri, and J. Kertész, “Early prediction of movie box office success based on Wikipedia activity big data,” *PLoS ONE*, vol. 8, no. 8, 2013.
[19] H. L. Vogel, *Entertainment Industry Economics*, 10th ed. Cambridge, U.K.: Cambridge Univ. Press, 2020.
[20] W. R. Bristi, Z. Zaman, and N. Sultana, “Predicting IMDb rating of movies by machine learning techniques,” in *Proc. ICCCNT*, 2019.
[21] A. L. Gomes, M. Ribeiro, and P. Quaresma, “Predicting IMDb rating of TV series with deep learning: The case of Arrow,” *arXiv:2211.03388*, 2022.
[22] J. Ramos, A. Pereira, and L. Cruz, “Movie rating prediction using sentiment features,” in *Proc. SALLD*, 2022.
[23] V. Udandarao, A. Gupta, and S. Bhattacharya, “Movie revenue prediction using machine learning models,” *arXiv:2405.11651*, 2024.
[24] W. Xie, Y. Liu, and H. Chen, “Predicting movie success with multi-task learning: GPT-based sentiment and SIR propagation,” *arXiv preprint*, 2025.
[25] E. Mohamed, M. Shafaei, and T. Solorio, “A first dataset for film age appropriateness investigation,” in *Proc. LREC*, 2020.
[26] S. Kar, S. Maharjan, and T. Solorio, “Folksonomication: Predicting tags for movies from plot synopses using emotion flow encoded neural network,” in *Proc. COLING*, 2018.
[27] M. C. Chiu, S. Lin, and W. Chen, “Screenplay quality assessment: Can we predict who gets nominated?” Nat. Univ. Singapore, Tech. Rep., 2020.
[28] K. Gorman and S. Bedrick, “We need to talk about standard splits,” in *Proc. ACL*, 2019.
[29] A. Søgaard, S. Ebert, J. Bastings, and K. Filippova, “We need to talk about random splits,” in *Proc. EACL*, 2021.
[30] P. Lewis, P. Stenetorp, and S. Riedel, “Question and answer test-train overlap in open-domain question answering datasets,” in *Proc. EACL*, 2021.
[31] A. Elangovan, J. He, and K. Verspoor, “Memorization vs. generalization: Quantifying data leakage in NLP performance evaluation,” in *Proc. EACL*, 2021.
[32] S. Kapoor and A. Narayanan, “Leakage and the reproducibility crisis in machine-learning-based science,” *Patterns*, vol. 4, no. 9, 2023.
[33] A. Lazaridou *et al.*, “Mind the gap: Assessing temporal generalization in neural language models,” in *Proc. NeurIPS*, 2021.
[34] E. M. Bender and B. Friedman, “Data statements for natural language processing,” *TACL*, vol. 6, 2018.
[35] J. Dodge *et al.*, “Documenting large webtext corpora: A case study on the Colossal Clean Crawled Corpus,” in *Proc. EMNLP*, 2021.
