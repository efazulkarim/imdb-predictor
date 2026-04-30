# Limitations

We close with an honest enumeration of the limitations that bear most
directly on how a reader should interpret our results. Several of these
are evidence we surface in §Dataset and §Results; we collect them here
so they cannot be missed.

## Selection bias in the corpus

The 5,195 screenplays in our corpus are drawn from the Internet Movie
Script Database (IMSDb), an editorially curated repository whose
inclusion criteria are not random. Two pieces of evidence make the
non-randomness concrete:

1. **Structural rating gaps.** Across the entire corpus there is **not
   a single film** with an IMDb rating in $[4.3, 5.0)$ or $[6.0, 7.0)$
   — intervals that together cover roughly 17% of IMDb's typical rating
   mass. The bimodality visible in our rating histogram is a sampling
   artifact, not a property of the true population.

2. **Year–rating confound.** Spearman $\rho \approx -0.93$ between
   decade midpoint and mean rating across decades (1930s mean 7.49 →
   2010s mean 5.48), most plausibly explained by survivorship bias:
   only canonical older films have made it into IMSDb.

Consequence: every metric we report should be interpreted as conditional
on IMSDb-style films, not on an arbitrary IMDb sample. A model deployed
to score films from outside this distribution — particularly
mid-quality, mid-budget contemporary films in the empty intervals
above — is operating outside its training support.

## Metadata dominates predictive signal

A linear regressor on three metadata features alone (`year`,
`movie_length`, `decade_encoded`) attains $R^{2} \approx 0.39$ on
five-fold cross-validation. Our full SBERT pipeline attains
$R^{2} \approx 0.56$. The marginal contribution of script content is
therefore $\Delta R^{2} \approx +0.17$ — statistically significant
($p \ll 10^{-30}$) but smaller than a casual reading of the headline
number would suggest.

We have not run a fully metadata-removed ablation in which the SBERT
system is denied year/length/decade entirely; the closest comparison
available within this work is `ols_structural`, which still includes
those three features. We name this as the most important missing
ablation for future work.

## Single-encoder, single-pooling SBERT configuration

For compute reasons we report a single SBERT configuration in the
main results: `all-MiniLM-L6-v2` with mean-pooling over 256-word
chunks (50-word overlap). Two alternatives we attempted but did not
complete:

- **Larger encoder** (`all-mpnet-base-v2`, 768-dim). Embedding the
  corpus required substantially more wall-clock time on CPU than was
  available; we leave the encoder-strength ablation to follow-up work.
- **Pooling-strategy ablation** (max-pooling, $\ell_2$-norm-weighted
  pooling). Each additional pooling strategy currently triggers a full
  re-encoding pass under our caching scheme; refactoring to share
  per-chunk embeddings across pooling strategies is straightforward
  and queued.

It remains possible that a stronger encoder or a non-mean pooling
operator yields a meaningfully different SBERT contribution.

## No properly-controlled long-document transformer comparison

An earlier iteration of this work compared SBERT + XGBoost against a
fine-tuned Longformer model. That comparison was not properly
controlled — the Longformer received only 3 metadata features (vs the
SBERT system's 19), was truncated to 2,048 of its 4,096 supported
tokens for memory reasons, and used a comparatively small
$771 \rightarrow 800 \rightarrow 128 \rightarrow 1$ MLP regression
head. Because all of these confounds favor SBERT, we do not present
that comparison here. A properly-matched study (frozen Longformer
embeddings + the same 19 features + an XGBoost head, full 4,096-token
context) requires GPU resources beyond what we have available and is
left to future work.

The efficiency comparison — frozen SBERT + XGBoost trains in minutes
on CPU and produces a $\sim 2$ MB model, whereas end-to-end
fine-tuning of a long-sequence transformer requires hours on a GPU
and produces a $\sim 500$ MB model — does not depend on accuracy
and stands as one of the contributions of this paper.

## Rating subjectivity and unobserved factors

IMDb ratings are subjective aggregates produced by self-selected
voters and reflect more than screenplay quality alone: production
values, marketing reach, cast and director reputation, festival
exposure, and release timing all influence ratings without appearing
in our feature set. Our model, like any text-only model of audience
reception, can at best predict an aggregate that the script does not
fully determine. The 44% of variance our pipeline does not explain
($1 - R^{2} \approx 0.44$) plausibly contains a sizeable irreducible
component.

## Statistical caveats

We report bootstrap 95% confidence intervals and paired Wilcoxon
signed-rank tests, but we do not adjust the latter for multiple
comparisons in the headline tables. With four baseline comparisons
the Bonferroni-corrected $\alpha$ is $\approx 0.0125$; the only
single-split comparison that is borderline at this level is
SBERT vs TF-IDF ($p = 0.014$). Under cross-validation the same
comparison is decisively significant ($p \approx 2.6 \times 10^{-5}$),
so we treat the conclusion as robust, but readers concerned with
multiple-testing should consult Table 4 rather than Table 2.

## Generalization

We do not evaluate on a held-out distribution outside IMSDb (e.g.
screenplays scraped from a different source, screenplays for
not-yet-released films, non-English screenplays). All claims in
this paper are about within-corpus performance.
