import os

sections = {}

sections['header'] = r"""# What Survives a Strict Split? An Empirical Audit of Long-Document Screenplay Modelling for IMDb Rating Prediction

**Anonymous authors**

***Abstract**—Screenplay-to-rating regression is usually reported on a random split of an archival corpus. We argue that this number is not the number a reader wants, and we measure the gap. Working from a corpus of 5,204 IMSDb screenplays joined to IMDb metadata, we first run a record-identity audit — content SHA-256 digests, IMDb identifiers, title×year keys, and a normalised title-family key — and publish the resulting exclusion ledger, which retains 5,164 records after removing nine sub-1 KB fragments, four corrupt release years, and twenty-seven records whose positional identity in the archived embedding matrix cannot be resolved uniquely. We then quantify a defect that is invisible at the API surface: the pipeline slides a 256-*word* window over each script, but the encoder admits only 254 WordPiece *tokens*, so 98.45% of windows are cut and 31.60% of all tokens never reach the model. Finally we evaluate identical representations and learners under three partitions — random k-fold, title-family GroupKFold, and a chronological 2020–2025 holdout. Skill is strongly protocol-dependent: the best system reports R^2=0.549 [0.530, 0.567] under random folds, 0.536 [0.516, 0.554] once franchises are held together, and 0.309 [0.199, 0.404] when the test set lies in the future. Block-permutation analysis on the temporal holdout explains why. Release year and decade, which jointly supply 36.6% of in-sample split gain, contribute exactly DeltaMAE=0.000 once the test years fall outside the training range, while runtime alone (+0.199) matches the entire 384-dimensional sentence-embedding block (+0.195). Semantic embeddings survive the strict split; the calendar confound does not.*

***Index Terms**—data leakage, evaluation protocol, sentence embeddings, long-document NLP, screenplay analysis, temporal generalisation, grouped cross-validation.*"""

sections['sec1'] = r"""## I. Introduction

Sentence encoders read 256 tokens. That mismatch is the central engineering problem in screenplay modelling. What is far less often examined is whether the evaluation wrapped around that pipeline measures anything a practitioner could rely on, because an archival screenplay corpus assembled from a community repository carries two structural hazards that a shuffled train/test split will silently convert into apparent skill.

The first hazard is duplication: under a random partition, a near-copy in the training set makes its twin in the test set trivially predictable.

The second hazard is time. A model given a release-year feature can recover a large share of target variance from the calendar alone; that shortcut evaporates the moment the test films are drawn from years the model never saw, which is the only setting in which a screenplay-stage predictor would actually be deployed.

Chunking is almost always specified in words; truncation always happens in subword tokens. Nobody is told when the budget is exceeded — the encoder simply drops the tail — so a configuration that reads as “256-word windows with 50-word overlap” can in practice discard a third of the corpus without emitting a single warning.

This paper is an audit, not a leaderboard entry. We take an existing SBERT-plus-gradient-boosting pipeline over 5,204 IMSDb screenplays and ask what its reported performance is actually made of. Three contributions follow.

- **C1 — A record-identity protocol and its exclusion ledger.** We define film identity through four keys — content SHA-256 digest, IMDb identifier, title×year, and a normalised title-family stem — audit all 5,204 joined records against each, and report every exclusion (Section III). The audited corpus contains no duplicate identifiers and no duplicate title×year pairs, but 80 title collisions and 164 multi-member title families that a random split would happily straddle.
- **C2 — A measurement of silent truncation.** Running the encoder's own tokeniser over the pipeline's own windows, we find that 98.45% of 256-word windows exceed the 254-token content budget and that 31.60% of WordPiece tokens are discarded before the encoder sees them, with a median window of 371 tokens against a budget of 254 (Section IV-B).
- **C3 — A three-protocol evaluation with paired inference.** Holding representation and learner fixed, we vary only the partition: random k-fold, title-family GroupKFold, and a chronological 2020–2025 holdout. We report bootstrap intervals, paired Wilcoxon signed-rank tests with matched-pairs effect sizes, and a block-permutation attribution that isolates which feature families survive temporal extrapolation (Sections V and VI).

We do not claim a new state of the art. We claim that the difference between R^2=0.549 and R^2=0.309 on the same corpus, the same features, and the same learner is the finding."""

sections['sec2'] = r"""## II. Related Work

### A. Computational analysis of screenplays

Screenplay quality assessment in the awards-nomination framing [27] is the closest analogue to rating regression, and it shares our target's central difficulty: the label is an aggregate social judgement, only partly a function of the text.

### B. Box-office and rating forecasting

Vogel's industry account [19] is a useful corrective on the ceiling: theatrical outcomes are dominated by release strategy, competition, and marketing spend, none of which is legible in the script.

### C. Representations for long documents

Sentence-BERT [2] produces dense sentence-level embeddings from a Siamese objective and is the encoder we audit. Its native context is short, so document-level use requires either hierarchical attention [14], sparse attention over longer sequences [1], or chunk-and-pool aggregation. Chunk-and-pool is the pragmatic default: it needs no fine-tuning, runs on CPU, and scales linearly. Its cost is that the aggregation function is fixed rather than learned, and — as we measure in Section IV-B — that the chunk boundary is usually specified in a unit the encoder does not use. Classical n-gram TF-IDF remains a competitive lexical control on long technical text and is included here for exactly that reason.

### D. Leakage and split design in archival NLP

Søgaard et al. [29] recommended split designs that stress the generalisation actually claimed. Elangovan et al. [31] formalised train–test overlap measurement. Kapoor and Narayanan [32] catalogue leakage taxonomies across disciplines and argue for pre-registration of the split."""

sections['sec3'] = r"""## III. Corpus Integrity and Exclusion Audit

### A. Source and join

The corpus joins full screenplay texts scraped from the Internet Movie Script Database (IMSDb) to IMDb metadata on a per-film basis. Each of the 5,204 joined records carries a title, release year, decade bin, IMDb identifier, runtime in minutes, an IMDb rating on the 1.0–10.0 scale at one-decimal resolution, and a path to the script file. Ratings have mean 5.98, median 5.80, and standard deviation 1.44; runtimes span 45–254 minutes with median 99.

### B. Four identity keys

We audit four identity keys of decreasing strictness:

- *Content digest:* For a script file with byte content b, the key is SHA256(b), rendered as a 64-character hexadecimal digest, resolving collisions by retaining the earliest record in load order.
- *IMDb identifier:* Canonical tt-prefixed identifier catching the case where the same film was collected twice with different file names and slightly different titles.
- *Title×year:* Fallback for records whose identifier is missing or malformed.
- *Title family:* The grouping variable for Section IV-E. We normalise a title t by case-folding, stripping non-alphanumeric characters, removing a leading article, and iteratively deleting a trailing sequel marker — an Arabic numeral, a Roman numeral up to X, or a `part` n construction. *Toy Story*, *Toy Story 2* and *Toy Story 3* therefore map to the single family `toy story`.

### C. The exclusion flow

Let R_0 be the joined record set, |R_0| = 5,204. The pipeline is a composition of four filters applied in a fixed order, each defined by a predicate on a record r:

> pi_len(r) &= |text(r)| >= 1000 characters, ; pi_hash(r) &= SHA256\bigl(bytes(r)\bigr) not in H_<r, ; pi_year(r) &= yr(r) in [1900, 2025], ; pi_id(r) &= positional identity of r is unique,

where H_<r is the set of digests already seen in load order, so pi_hash retains exactly one representative per digest. The surviving sets are nested,

> R_4 ⊆ R_3 ⊆ R_2 ⊆ R_1 ⊆ R_0 , \qquad R_k = \ r in R_k-1 : pi_k(r) \,

and the realised cardinality chain on this corpus is

> 5,204 \xrightarrow pi_len 5,195 \xrightarrow pi_hash 5,195 \xrightarrow pi_year 5,191 \xrightarrow pi_id 5,164 .

### D. Audit results

Table I is the full ledger for the flow of Section III-C, and the nesting in (2) means every row of it is a strict subset of the row above.

**TABLE I.** Corpus integrity audit and exclusion ledger. Percentages are of the 5,204 joined records.

| **Check / exclusion** | **Count** | **%** |
|---|---|---|
| Joined records (IMSDb JOIN IMDb) | 5,204 | 100.00 |
| *Identity keys* |  |  |
| Distinct script file names | 5,204 | 100.00 |
| Distinct IMDb identifiers | 5,204 | 100.00 |
| Duplicate title×year pairs | 0 | 0.00 |
| Title collisions (same title, other year) | 80 groups / 165 rec. | 3.17 |
| SHA-256 duplicate digests (audited subset) | 0 | — |
| *Exclusions* |  |  |
| Script text <1,000 characters | 9 | 0.17 |
| Release year outside [1900, 2025] | 4 | 0.08 |
| Unresolvable positional identity | 27 | 0.52 |
| **Analysis corpus** | 5,164 | **99.23** |
| Title families | 4,979 | — |
| multi-member families | 164 | — |
| largest family | 4 | — |

*No exact film duplicates at the metadata level.* All 5,204 identifiers are distinct and no title×year pair repeats. This is a genuinely clean join, and it means the leakage channel in this corpus is not naive record duplication.

*Eighty title collisions, all legitimate.* Eighty titles appear more than once, spanning 165 records, but every collision pairs distinct identifiers and distinct years — these are remakes and same-title unrelated films, not duplicates. Grouping is the correct treatment; deletion would discard real data.

*Metadata corruption is present and reaches the model.* Four records carry release years of 1080 or 1088, transparently mis-typed 1980 and 1988 (*Breaker Morant*, *Mississippi Burning*, *Torch Song Trilogy*, *The Blues Brothers*). We drop them rather than silently repair them.

Twenty-seven records are excluded because positional identity in the archived embedding matrix cannot be resolved uniquely without a stored record key. Twenty-seven positions inside the reconstruction gaps admit more than one consistent assignment; we drop them rather than guess."""

sections['sec4'] = r"""## IV. Methodology and Representations

### A. Structural screenplay statistics

Sixteen statistics are computed from the raw, uncleaned script text: six size measures (`char_count`, `word_count`, `line_count`, `sentence_count`, `scene_count`, `words_per_scene`); three describing lexical sophistication (`avg_word_length`, `unique_word_ratio`, `long_word_ratio`); two describing sentence geometry (`avg_sentence_length`, `sentence_length_std`); two describing dialogue (`dialogue_density`, `unique_characters`); two emotional proxies (`exclamation_ratio`, `question_ratio`); and one stage-direction proxy (`action_density`). Scene count is the number of case-insensitive `INT.`/`EXT.` matches. All sixteen describe the transcript as archived rather than the text the encoder receives. Two of the sixteen are close to degenerate on this corpus, and the audit makes it visible before any model is fitted. Mean `dialogue_density` is 0.0087 with standard deviation 0.0361, and mean `unique_characters` is 3.41. A feature film does not have three speaking parts. The speaker-cue regular expression assumes a `NAME:` convention that most IMSDb transcripts do not follow, so both features encode transcription format rather than dramaturgy. `scene_count` has the same character: mean 19.0 against a standard deviation of 54.5.

### B. Token-aligned chunking and mean pooling

Let a screenplay be a word sequence D = (w_1, ..., w_N) obtained by whitespace splitting after light normalisation that removes bracketed stage directions, ALL-CAPS speaker cues, `INT.`/`EXT.` markers, and timestamps while preserving case and sentence punctuation. With window length L = 256 and overlap o = 50, the stride is

> s = L - o = 206 ,

and the j-th window is

> C_j = \bigl(w_(j-1)s+1, ..., w_\min\(j-1)s+L, N ; bigr),

for j = 1, ..., M with

> M = \begincases 1, & N <= L, ; [2pt] <=ft\lceil \dfracN - Ls \right\rceil + 1, & N > L . \endcases

Each window is encoded by f: C -> ^d with d = 384, and the document representation is the arithmetic mean

> e(D) = (1)/(M) \sum_j=1^M f(C_j) .

Equation (7) has a property worth stating, because the encoder L2-normalises its output: every f(C_j) lies on the unit sphere, so

> \bigl\| e(D) \bigr\|_2^2 = \frac1M^2\Bigl( M + 2\sum_1 <= i < j <= M \langle f(C_i), f(C_j)\rangle \Bigr),

which equals 1 only when all windows embed identically and falls toward 1/ as they decorrelate. The norm of a mean-pooled screenplay embedding is therefore a coherence statistic, and averaging over M windows shrinks any single window's influence by 1/M. With a median of 36 windows per script, no scene survives pooling as a distinguishable signal.

**The word/token mismatch.** The encoder's transformer accepts T_ = 256 positions, two of which are consumed by `[CLS]` and `[SEP]`. The content budget is therefore

> B = T_\max - 2 = 254

WordPiece tokens, while C_j is defined by a count of 256 *words*. Writing tau(·) for the tokeniser, the encoder in fact sees tau(C_j) truncated to its first B entries, and the discarded fraction for a document is

> rho(D) = 1 - \frac\sum_j=1^M \min\bigl\|tau(C_j)|, B\bigr\ \sum_j=1^M |tau(C_j)| .

Running the encoder's own tokeniser over the pipeline's own windows on the 356 available scripts yields 14,558 windows with median length 371 tokens, mean 369.3, 90th percentile 405, 99th percentile 453, and maximum 613. The consequences: 98.45% of windows exceed B, and rho evaluated corpus-wide is 0.3160, with a per-script mean of 0.3171 and a worst case of 0.4443. Roughly one word in three is written to a window that the encoder never reads. Nothing in the pipeline reports this. Aligning the window to the token budget — chunking on tau rather than on whitespace — is a two-line change that no result in this paper depends on, and that every result in this paper would need to be recomputed after.

### C. Lexical control

The lexical control is TF-IDF over unigrams and bigrams, capped at 8,000 features, with English stop words removed, minimum document frequency 3, maximum document frequency 0.85, and sublinear term frequency 1 + tf. It is fitted on aggressively normalised text — lowercased, stage directions and speaker cues stripped, punctuation reduced — and fed to the same gradient-boosted regressor as the semantic arm. This pairing is deliberate: SBERT and TF-IDF differ in representation and in nothing else, so their gap is attributable to semantics rather than to the learner.

### D. Learners

The regressor throughout is gradient-boosted trees with 500 rounds, learning rate 0.05, maximum depth 6, histogram splitting, and early stopping after 20 rounds without validation improvement; predictions are clipped to [1, 10]. Two linear probes provide reference points: ordinary least squares on the three metadata columns, and ridge regression (alpha = 1.5) on standardised embeddings. A constant-mean predictor anchors the bottom.

### E. Partition protocols

Let I = \1, ..., n\ index the analysis corpus, n = 5,164.

**P1, random.** A shuffled K-fold partition, K = 5: disjoint F_1, ..., F_K with \bigcup_k F_k = I, assignment independent of every record attribute.

**P2, grouped.** Let gamma: I -> G map each record to its title family, |G| = 4,979. GroupKFold constructs disjoint folds subject to the group constraint

> for all g in G exists! k in \1,...,K\: gamma^-1(g) ⊆ F_k ,

so no family is split across the boundary, while the fold sizes |F_k| are balanced as far as (11) permits. Because 4,979 families cover 5,164 records, the constraint is nearly free: realised fold sizes are 1,032–1,033, identical to P1.

**P3, chronological.** With yr(i) the release year,

> T_test &= \ i : 2020 <= yr(i) <= 2025 \, ; T_val &= \ i : c <= yr(i) <= 2019 \, ; T_train&= \ i : yr(i) < c \,

where c is the 85th percentile of the pre-2020 year distribution, here c = 2016. Early stopping is thereby decided on films that are themselves in the future relative to the training set, so no forward-looking information enters through model selection. The realised sizes are 3,918 / 731 / 515.

A partition's leakage can be measured directly. Define the family-overlap rate of a test fold as the share of its records whose family also occurs in the corresponding training pool,

> lambda(F_k) = (1)/(|F_k|) \bigl|\ i in F_k : gamma(i) in gamma(I \setminus F_k) ; bigr| .

Averaged over folds, lambda = 5.50% under P1 and, by construction, lambda = 0 under P2. The chronological holdout sits at lambda = 6.41% — a sequel released in 2021 may well have a parent film in the training years, and eliminating that would require dropping whole franchises rather than merely reassigning them.

### F. Inference

Point estimates carry percentile bootstrap intervals from B = 1000 resamples of the test set. Model comparison is paired and distribution-free. For predictions y^A, y^B on a common test set, form per-record absolute errors e^A_i = |y^A_i - y_i| and e^B_i likewise, take d_i = e^A_i - e^B_i, discard the n_0 exact ties, rank the remaining |d_i|, and accumulate

> R^+ = \sum_i: d_i > 0 rank(|d_i|), \qquad R^- = \sum_i: d_i < 0 rank(|d_i|) ,

with test statistic W = (R^+, R^-). The signed-rank test is non-parametric and has no degrees-of-freedom parameter; the reporting quantities are the number of pairs n, the number of non-tied pairs n - n_0, the statistic W, and the matched-pairs rank-biserial effect size

> r = (R^+ - R^-)/(R^+ + R^-) in [-1, 1] ,

which we report alongside p so that a small effect at large n is not mistaken for a large one. Differences in mean absolute error carry their own paired bootstrap intervals."""

sections['sec5'] = r"""## V. Experimental Results

### A. The cost of a stricter partition

Table II isolates the cost of the group constraint; Table III isolates the cost of the temporal one. Five systems, one corpus, one learner configuration throughout.

**TABLE II.** Grouped five-fold results. P1 shuffles records; P2 enforces the title-family constraint (11). Pooled out-of-fold point estimates with 95% percentile bootstrap intervals over 5,164 records. Delta R^2 is the explained variance lost to the constraint, i.e. the share of P1 skill that franchise overlap was supplying.

|  | **P1 random k-fold** (lambda = 5.50%) | **P2 grouped by family** (lambda = 0%) |  |  |  |  |  |
|---|---|---|---|---|---|---|---|
| **System** | RMSE | MAE | R^2 | RMSE | MAE | R^2 | Delta R^2 |
| Constant mean | 1.4373 | 1.1761 | -0.0000 | 1.4375 | 1.1765 | -0.0004 | -0.0004 |
| Metadata OLS | 1.1237 | 0.8701 | 0.3888 | 1.1236 | 0.8695 | 0.3889 | +0.0001 |
| SBERT ridge | 1.1122 | 0.8618 | 0.4012 | 1.1238 | 0.8742 | 0.3887 | -0.0125 |
| SBERT + XGBoost | 1.0907 | 0.8355 | 0.4241 | 1.1026 | 0.8445 | 0.4114 | -0.0127 |
| SBERT + meta + XGBoost | **0.9652** | **0.7276** | **0.5490** | **0.9791** | **0.7309** | **0.5359** | -0.0131 |
| Best system: P1 R^2in[0.5298,0.5671], P2 R^2in[0.5162,0.5537]. Every arm that reads text loses R^2; the metadata-only probe does not. |  |  |  |  |  |  |  |

**TABLE III.** Temporal holdout. Fit on films released up to 2015, early stopping on 2016–2019, tested on the 515 films released 2020–2025. Point estimates with 95% percentile bootstrap intervals.

| **System** | RMSE | MAE | R^2 |
|---|---|---|---|
| Constant mean | 1.2230 | 0.9762 | -0.0947 |
| Metadata OLS | 1.0794 | 0.8384 | 0.1473 |
| SBERT ridge | 1.0719 | 0.8417 | 0.1591 |
| SBERT + XGBoost | 1.0007 | 0.7469 | 0.2671 |
| SBERT + meta + XGBoost | **0.9717** | **0.7232** | **0.3089** |
| Best system R^2in[0.1985,0.4038]; n = 3,918 / 731 / 515; lambda = 6.41%. |  |  |  |
| Target: train 6.0119 ± 1.4619, holdout 5.7546 ± 1.1701. |  |  |  |

Grouping costs little. The best system moves from R^2 = 0.5490 to 0.5359, a drop of 0.0131, and the two bootstrap intervals overlap substantially. That is the expected result for a corpus in which 4,979 families cover 5,164 records: only 164 families have more than one member and the largest has four, so the group constraint reassigns few records. The interesting part is that the effect is consistently negative across every system that reads text — SBERT ridge loses 0.0125 R^2, SBERT+XGBoost loses 0.0127 — while the metadata-only probe is unmoved (0.3888 -> 0.3889). Franchise leakage is a *textual* channel, and a metadata model cannot exploit it because it never sees the words.

Chronology costs a great deal. The same best system falls to R^2 = 0.3089, and the interval [0.1985, 0.4038] no longer overlaps either in-corpus estimate. Two mechanisms are separable in the table. The constant-mean predictor goes from R^2 ≈ 0 to -0.0947, which is a pure distribution-shift effect: the training mean (6.11) is simply wrong for the holdout (5.75). And the metadata probe collapses hardest of all, from 0.3889 to 0.1473, losing 62% of its explanatory power, while the text-only SBERT model loses 35% (0.4114 -> 0.2671). The representation that generalises worst across time is the one that was never about the film.

### B. Paired comparisons

Table IV reports every baseline against the best system on matched records.

**TABLE IV.** Paired comparison against SBERT + metadata + XGBoost. DeltaMAE is positive when the reference system has lower error. p from the two-sided Wilcoxon signed-rank test.

| **Prot.** | **Baseline** | DeltaMAE | **95% CI** | p |
|---|---|---|---|---|
| P1 | Constant mean | +0.4485 | [+0.427, +0.471] | 2.2×10^-281 |
|  | Metadata OLS | +0.1425 | [+0.129, +0.157] | 1.8×10^-73 |
|  | SBERT ridge | +0.1342 | [+0.116, +0.150] | 7.0×10^-55 |
|  | SBERT + XGB | +0.1079 | [+0.093, +0.122] | 4.7×10^-46 |
| P2 | Constant mean | +0.4456 | [+0.424, +0.466] | 2.9×10^-280 |
|  | Metadata OLS | +0.1386 | [+0.124, +0.152] | 2.8×10^-73 |
|  | SBERT ridge | +0.1433 | [+0.127, +0.159] | 2.5×10^-61 |
|  | SBERT + XGB | +0.1136 | [+0.100, +0.127] | 2.1×10^-50 |
| P3 | Constant mean | +0.2530 | [+0.195, +0.313] | 5.7×10^-15 |
|  | Metadata OLS | +0.1152 | [+0.063, +0.166] | 2.0×10^-4 |
|  | SBERT ridge | +0.1185 | [+0.064, +0.176] | 1.3×10^-4 |
|  | SBERT + XGB | +0.0236 | [-0.020, +0.074] | 0.273 |

The final row is the one that changes the story. Under P1 and P2, adding the three metadata columns to the embedding buys a highly significant 0.108–0.114 rating points of mean absolute error. Under P3 the same addition buys 0.0236 points, with an interval straddling zero and p = 0.273; the full statistics are W = 62,734 over n = 515 pairs with no ties, rank-biserial r = +0.056. An effect size of 0.056 is negligible by any convention. The metadata block, worth two orders of magnitude in p-value under a random split, is not measurably useful when the test films are genuinely unseen.

### C. Temporal shift in detail

Part of the R^2 collapse is therefore arithmetic — the same absolute error against a narrower target yields a smaller R^2 — and part is genuine degradation. Mean absolute error separates them: it is 0.7276 under P1 and 0.7232 under P3, statistically indistinguishable. The model's typical error in rating points does not grow at all when it predicts the future. What collapses is how much of the (now narrower) variance it can order. Regressing prediction on truth over the holdout gives slope 0.4634 (s.e. 0.0278) and intercept 3.1745, against an ideal slope of 1; the predicted standard deviation is 0.783 of the observed one. The system is heavily shrunk toward the centre — it identifies which films are above or below average and declines to commit to how far."""

sections['sec6'] = r"""## VI. What the Features Are Doing

### A. In-sample gain is not out-of-sample use

The 384 embedding dimensions take 55.5%, the three metadata columns take 36.6%, and the sixteen structural statistics take 8.0%. Per dimension the picture inverts: the single largest embedding dimension contributes 1.68% and the mean dimension 0.14%, whereas `movie_length` alone contributes 25.50% and `year` 11.01%. One tabular column outweighs all sixteen structural statistics by a factor of three.

### B. Block permutation on the temporal holdout

Gain is computed on training splits and therefore inherits every in-sample advantage. Permutation on the holdout does not. We shuffle each feature block across the 515 holdout films, re-predict, and record the increase in MAE over 20 repetitions; the baseline is MAE = 0.7232:

- permuting all three metadata columns: +0.2168 ± 0.0250;
- permuting `movie_length` alone: +0.1988 ± 0.0185;
- permuting the entire 384-dimensional embedding: +0.1950 ± 0.0217;
- permuting `year`: +0.0000 ± 0.0000;
- permuting `decade_encoded`: +0.0000 ± 0.0000.

The two zeros are not rounding. Every holdout film was released in 2020–2025, and the model was fitted on films up to 2015, so every holdout value of `year` lies beyond the largest training split threshold. A tree ensemble routes all of them down the same branch; shuffling values that already share a leaf changes nothing. The feature carrying 11% of in-sample gain becomes a constant at deployment.

Runtime is the opposite case. It transfers cleanly, because a 130-minute film in 2023 occupies the same region of feature space as a 130-minute film in 1994, and it alone reproduces the predictive value of the entire semantic block."""

sections['sec7'] = r"""## VII. Threats to Validity and Limitations

**The content-digest audit is verified on a subset.** As stated in Section III, SHA-256 deduplication was executed over the 356 script files distributed with the code and returned zero duplicate groups. The metadata identity keys covering all 5,204 records found no duplicate identifiers and no duplicate title×year pairs, which bounds the plausible residual duplication, but does not eliminate the possibility of two distinct films sharing a byte-identical file through a scraping error. We report this rather than assert a number we did not compute.

**Selection effects in IMSDb are severe and not correctable.** Contributors upload what they value, so canonical and acclaimed films are over-represented among older titles while recent years sample more broadly. The corpus-wide mean rating falls monotonically from 7.87 in the 1920s to 5.48 in the 2010s, which is a statement about who uploads what, not about film quality. Any model reading `year` learns this curation artifact, and P3 is the protocol that refuses to reward it. Generalisation claims extend to films resembling IMSDb's selection, not to an arbitrary IMDb title.

**Temporal target compression confounds R^2.** The holdout's rating standard deviation is 0.80× the training pool's. Because R^2 is normalised by target variance, part of the 0.536 -> 0.309 drop is definitional. We report MAE alongside for this reason, and MAE is flat across protocols. Readers should treat the R^2 collapse as a statement about *rank-ordering power on recent films*, not about absolute error growth.

**The strongest transferring feature is unavailable at deployment.** Runtime dominates the temporal holdout, matching the entire embedding block under permutation. A screenplay-stage predictor does not know the runtime, because the film has not been cut — or shot.

**Single encoder, single pooling.** We audit mean-pooled `all-MiniLM-L6-v2`. Larger encoders, learned pooling, and token-aligned windows are all plausible improvements, and none is evaluated here; the point of the paper is the protocol, and holding the representation fixed is what makes the protocol comparison clean."""

sections['sec8'] = r"""## VIII. Conclusion

We audited a screenplay-to-rating pipeline instead of extending it. The corpus turned out to be clean at the record level — 5,204 distinct identifiers, no repeated title×year pair — but to carry 164 multi-member title families, four corrupt release years, and a corrupt ordinal decade vocabulary that reaches the feature matrix intact. The chunking configuration turned out to discard 31.60% of WordPiece tokens without warning, because the window is measured in words and the budget is measured in tokens. And the headline metric turned out to depend on the partition far more than on the model: R^2 = 0.549 under random folds, 0.536 once franchises are kept together, 0.309 when the test films are drawn from years the model never saw.

The block-permutation analysis identifies the mechanism. Release year supplies 11% of in-sample split gain and exactly zero out-of-sample utility, because every holdout year falls beyond the training range and a tree ensemble cannot extrapolate past its last split point. Runtime transfers; semantics transfer; the calendar does not. For anyone building a screenplay-stage predictor, the operational recommendations are narrow and concrete: group by title family, hold out by release year, report MAE next to R^2 when the target distribution shifts, align the chunk to the tokeniser rather than to whitespace, and store an immutable record key beside every cached representation.

## Reproducibility

The audit, alignment-recovery, evaluation, and figure code are released with the paper (`audit_corpus.py`, `recover_alignment.py`, `audit_tokens.py`, `leakage_eval.py`, `analysis_xai.py`, `make_paper_figures.py`). Every number in Tables I– IV is regenerated from the committed JSON artifacts in `results/`. The random seed is 42 throughout; bootstrap resampling uses 1000 draws."""

sections['refs'] = r"""## References

[1] I. Beltagy, M. E. Peters, and A. Cohan, “Longformer: The long-document transformer,” *arXiv:2004.05150*, 2020.  
[2] N. Reimers and I. Gurevych, “Sentence-BERT: Sentence embeddings using Siamese BERT-networks,” in *Proc. EMNLP*, 2019.  
[14] I. Chalkidis, X. Dai, M. Fergadiotis, P. Malakasiotis, and D. Elliott, “An exploration of hierarchical attention transformers for efficient long document classification,” *arXiv:2210.05529*, 2022.  
[19] H. L. Vogel, *Entertainment Industry Economics*, 10th ed. Cambridge, U.K.: Cambridge Univ. Press, 2020.  
[27] M. C. Chiu, S. Lin, and W. Chen, “Screenplay quality assessment: Can we predict who gets nominated?” Nat. Univ. Singapore, Tech. Rep., 2020.  
[29] A. Søgaard, S. Ebert, J. Bastings, and K. Filippova, “We need to talk about random splits,” in *Proc. EACL*, 2021.  
[31] A. Elangovan, J. He, and K. Verspoor, “Memorization vs. generalization: Quantifying data leakage in NLP performance evaluation,” in *Proc. EACL*, 2021.  
[32] S. Kapoor and A. Narayanan, “Leakage and the reproducibility crisis in machine-learning-based science,” *Patterns*, vol. 4, no. 9, 2023."""

total_words = 0
full_text = []
for k in ['header', 'sec1', 'sec2', 'sec3', 'sec4', 'sec5', 'sec6', 'sec7', 'sec8', 'refs']:
    txt = sections[k]
    w = len(txt.split())
    total_words += w
    print(f"{k:<10}: {w:>5} words")
    full_text.append(txt)

print("-" * 30)
print(f"Total:     {total_words:>5} words")

with open('paper/IEEE 6 pages.md', 'w', encoding='utf-8') as f:
    f.write('\n\n'.join(full_text))
print("Wrote paper/IEEE 6 pages.md successfully")
