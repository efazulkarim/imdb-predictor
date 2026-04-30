# 5. Discussion

## What our results say (and don't say) about screenplay-driven rating prediction

The headline number — 5-fold $R^{2} \approx 0.56$ — is consistent with a
moderately useful predictor of audience reception from screenplay
content alone. It is not, however, evidence that an audience rating is
"in the script." Three metadata features (year, decade, runtime)
already explain $R^{2} \approx 0.38$ on the same corpus and splits.
The $\Delta R^{2} \approx +0.17$ that SBERT and the structural feature
set add, while statistically robust, places a hard ceiling on how
much of an audience rating any text-only pipeline can be claimed to
"recover from the screenplay."

Two corollaries follow. First, papers in this area should report
metadata-only OLS as a mandatory baseline; without it, reported R²
values systematically over-attribute predictive power to the text
component. Second, for downstream practitioners — for example,
studio readers using a script-quality score in development decisions
— the *delta* over a metadata baseline, not the absolute R², is the
relevant figure of merit.

## On the negative result for sample weighting

A common reflex when faced with a long-tailed regression target is to
upweight tail samples by inverse frequency. In our setting this
*hurts* significantly: at 5-fold pooled $n \approx 5{,}195$, the
paired Wilcoxon test yields $p \approx 3 \times 10^{-26}$ in favor of
unweighted training. Two factors plausibly explain the result. First,
the most up-weighted bucket (Excellent, weight 13.24×) contains only
147 training samples, so amplifying its gradients makes the regressor
sensitive to a high-leverage minority. Second, gradient-boosted trees
fit residuals; large weights on a small bucket produce large per-step
residuals there, which the next tree then over-corrects, propagating
instability across boosting rounds. We do not claim this generalizes
beyond gradient-boosted regression on screenplay-style features, but
we note that the practice was not load-bearing — and was actively
harmful — in our pipeline.

## On the dataset

The IMSDb corpus is the principal public source of feature-film
screenplays for academic work, and it is appropriate that it
continues to be used. The structural gaps we report ($[4.3, 5.0)$ and
$[6.0, 7.0)$ are empty intervals) are not a defect of IMSDb so much
as a property of the editorial process by which screenplays end up
there. What matters is that the property be **disclosed** and that
generalization claims be scoped accordingly. A film randomly drawn
from IMDb's full distribution will frequently land in one of those
empty intervals; a model trained without correction on IMSDb has no
basis on which to predict it.

A second, more subtle issue is the year–rating confound. The mean
rating in our 1930s sample is 7.49; in the 2010s it is 5.48. Any
predictor with access to release year will trivially exploit this
calendar trend, and the trend is not "older films are better" but
"older films that survived are better." We strongly recommend that
follow-on work include a metadata-removed ablation as a matter of
course.

## On the place of long-document transformers

We did not include a properly controlled long-document transformer
baseline (e.g. Longformer with the same 19 features, full 4,096-token
context, and a comparable head) because the GPU budget required to do
so was not available within the present scope. The efficiency
contribution of the SBERT pipeline — minutes on CPU, $\approx 2$ MB
final model — is, however, independent of any such comparison and
remains a defensible practical claim on its own.

# 7. Conclusion

We presented a screenplay-to-IMDb-rating predictor combining frozen
SBERT embeddings, hand-crafted structural features, gradient
boosting, and a Ridge meta-regressor stacked over four base models;
benchmarked it against the constituent baselines under both
single-split and 5-fold cross-validated protocols; and reported every
comparison with bootstrap confidence intervals and paired Wilcoxon
significance tests. The stacked system attains
$R^{2} = 0.577 \pm 0.010$ on 5-fold CV, significantly improving on
the strongest single base model
($p \approx 4 \times 10^{-6}$) and on every other baseline
($p \le 3 \times 10^{-5}$). A 100-trial Optuna search on the
XGBoost head independently confirms that the hand-picked defaults
are systematically under-regularized and recommends a substantially
shrunk learning rate ($\approx 0.011$) and an order-of-magnitude
larger $L_{1}$ penalty for production use.

We additionally documented (i) that approximately 70% of the
explained variance is recoverable from three metadata features
alone, (ii) that the legacy practice of inverse-frequency sample
weighting harms rather than helps performance in this setting,
(iii) that the constituent base models carry partially complementary
signal — confirmed by the stacked R² standard deviation being
*tighter* than any base model and by the negative Ridge weight on
the OLS-structural prediction once SBERT and TF-IDF predictions are
available — and (iv) that the underlying corpus exhibits structural
sampling artifacts that warrant disclosure in any future study using
IMSDb-derived data.

Our pipeline is fully reproducible from a single command, runs on
commodity CPU hardware, and produces a deployable 2 MB model. Code,
cached embeddings, per-model predictions, and a dataset snapshot are
released alongside this paper.

Future work should (a) include a metadata-removed ablation to
isolate the SBERT contribution from era and length, (b) benchmark a
properly controlled long-document transformer baseline at full
context with matched feature budgets, (c) investigate alternative
chunk-pooling operators (max, $\ell_2$-weighted) using the refactored
caching scheme we already release that reuses per-chunk embeddings
across pooling strategies, and (d) evaluate on a held-out
distribution outside the IMSDb editorial selection.
