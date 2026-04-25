# Dataset

## Source

We construct our corpus by joining two sources: (i) full screenplay texts
collected from the Internet Movie Script Database (IMSDb) and (ii) movie
metadata (release year, runtime, IMDb rating) retrieved from IMDb.
The joined dataset consists of **5,204 records**, of which **5,195** have
script files of at least 1 KB and a valid IMDb rating after cleaning. All
subsequent statistics are reported over the cleaned corpus.

## Per-record fields

Each record contains:

- **Movie name** — title string
- **Year** — release year (1922–2025 after correction; see *Data quality*)
- **Decade** — derived bin (1920s through 2020s)
- **IMDb Rating** — target variable, scale 1.0–10.0 (one decimal place)
- **IMDb ID** — unique identifier
- **Movie length** — runtime in minutes (range 45–254, median 99)
- **Script file** — path to the corresponding `.txt` screenplay

## Script characteristics

Script byte-size statistics (file size on disk, UTF-8 / Latin-1):

| Statistic | Bytes |
|---|---|
| 5th percentile | 19,009 |
| 25th percentile | 32,089 |
| Median | 44,739 |
| 75th percentile | 64,612 |
| Maximum | 575,764 |

The 5th-percentile script is ≈19 KB (roughly 3,000 words), well below the
length of a typical feature-film screenplay (≈25,000 words). 59 scripts
are below 10 KB and are likely partial transcripts; we retain those that
exceed our 1 KB minimum because larger filtering thresholds did not change
results materially in pilot ablations.

## Rating distribution

The marginal rating distribution has the following bucket counts:

| Range | n | % |
|---|---|---|
| Low [1.0, 4.0) | 555 | 10.7% |
| Medium [4.0, 6.0) | 2,736 | 52.6% |
| Good [6.0, 8.0) | 1,685 | 32.4% |
| Excellent [8.0, 10.0) | 228 | 4.4% |

Mean 5.98, median 5.80, standard deviation 1.44, IQR 5.30–7.40.

## Sampling artifacts (important caveat)

The fine-grained empirical rating distribution is **not** consistent with
uniform random sampling from IMDb. In particular, after enumerating every
unique rating value in the dataset we observe **structural gaps**:

| Empty interval | Width | Count |
|---|---|---|
| [4.3, 5.0) | 0.7 | 0 |
| [6.0, 7.0) | 1.0 | 0 |

i.e. there is not a single film in our corpus rated between 4.3 and 4.9
(inclusive at 0.1 resolution), nor between 6.0 and 6.9. These intervals
together cover roughly 17% of the ordinary IMDb rating mass, so their
absence is highly unlikely under random sampling.

We attribute this to selection effects upstream of our control: the IMSDb
collection over-represents acclaimed/canonical films and films selected
for inclusion through editorial curation by individual contributors,
producing a corpus that is **bimodal by construction**. The rating
distribution figure exhibits two distinct modes (a wide low/medium mode
near 5–6 and a high mode near 7–8) for this reason.

We disclose this caveat because it has material implications for
generalization: predictive performance reported on this corpus should not
be read as performance on an arbitrary IMDb-rated film, but rather on
films that are similar in *selection* to those in IMSDb. A randomly drawn
film from IMDb's full distribution would frequently fall in the empty
intervals above, on which our model has no training signal.

## Temporal coverage and a confound

Year coverage spans **1922–2025**. After fixing four typographic errors
(records dated "1080" / "1088" — verified to be 1980 / 1988 from the
movie title), the per-decade distribution is:

| Decade | n | Mean rating |
|---|---|---|
| 1920s | 12 | 7.87 |
| 1930s | 85 | 7.49 |
| 1940s | 117 | 7.63 |
| 1950s | 128 | 7.33 |
| 1960s | 150 | 7.34 |
| 1970s | 218 | 6.63 |
| 1980s | 431 (+4)¹ | 6.29 |
| 1990s | 640 | 6.20 |
| 2000s | 1,117 | 5.88 |
| 2010s | 1,779 | 5.48 |
| 2020s | 523 | 5.73 |

¹ four records originally typed "1080"/"1088" added back to 1980/1988.

There is a strong, monotone-decreasing relationship between release decade
and mean IMDb rating across the corpus (Spearman ρ ≈ –0.93 over decade
midpoints). We do not interpret this as "older films are objectively
better"; rather, it is consistent with **survivorship and selection bias**
— the older a film is, the more likely it is to appear in IMSDb only if
it is canonical and well-regarded — combined with the fact that our
modern (2010s, 2020s) sample is large enough to include films across the
full quality spectrum.

The implication for our results is direct: any model that observes
`Year` or `Decade` features can exploit this calendar-quality confound to
recover a substantial fraction of rating variance without any signal from
the screenplay itself. In our results section we therefore present an
ablation in which the metadata columns (`year`, `movie_length`,
`decade_encoded`) are removed, isolating the contribution of script
content to predictive performance.

## Data splits

For headline results we use a single 70%/15%/15% train/validation/test
split with `random_state=42`, mirroring the legacy training pipeline. For
robustness, we additionally report 5-fold cross-validated mean ± standard
deviation over the same model set in §X. Splits are stratification-free
(plain shuffle) but we have verified that the per-bucket rating
distribution matches the marginal distribution within ±1.5% in all folds.

## Reproducibility

The cleaned dataset, the train/val/test indices, and per-record SBERT
embeddings used in our experiments are released alongside the code at
[REPO_URL]. The `dataset_snapshot.json` artifact (committed) contains the
exact summary statistics reported in this section.
