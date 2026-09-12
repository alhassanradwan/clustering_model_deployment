# RFM Customer Segmentation

K-Means segmentation of an online retailer's customers using the Recency,
Frequency and Monetary model, reproducing Chen et al. (2012), *Data mining for
the online retail industry* (Journal of Database Marketing & Customer Strategy
Management, 19(3), 197-208).

Built as a reproducible DVC pipeline with Hydra configuration and MLflow
experiment tracking on DagsHub.

## Data

`Online Retail.xlsx` — 541,909 transactions from a UK gift retailer, Dec 2010
to Dec 2011. Tracked by DVC, not git.

After preprocessing: **3,902 UK customers**.

## Results

Five segments, k chosen for interpretability (the elbow is gentle — see below):

| Cluster | Customers | Recency | Frequency | Monetary | % customers | % sales |
|---|---|---|---|---|---|---|
| 1 — champions | 318 | 14 d | 20.1 | £10,909 | 8.1% | **51.3%** |
| 4 — loyal | 812 | 17 d | 5.7 | £1,922 | 20.8% | 23.1% |
| 2 — slipping | 901 | 96 d | 3.4 | £1,279 | 23.1% | 17.0% |
| 3 — new / light | 791 | 30 d | 1.6 | £353 | 20.3% | 4.1% |
| 0 — lapsed | 1,080 | 214 d | 1.2 | £276 | 27.7% | 4.4% |

**8.1% of customers generate 51.3% of revenue.** The paper found 5% driving
25%; this dataset concentrates value more sharply.

### Is the model sound?

| metric | value | reading |
|---|---|---|
| silhouette | 0.322 | moderate separation (random baseline: -0.011) |
| Davies-Bouldin | 0.987 | good (below 1.0) |
| Calinski-Harabasz | 3002.2 | — |
| misfit rate | 1.3% | few customers closer to a neighbouring cluster |
| **stability (ARI, 10 seeds)** | **0.930** | **same grouping every time** |

Silhouette of 0.32 is honest rather than impressive: RFM data is a continuum,
not naturally separated blobs, so K-Means is drawing sensible lines through a
smooth cloud. The stability figure is the stronger evidence — re-running from
different seeds, or dropping 20% of customers, reproduces the same segments.

## Preprocessing decisions

Three choices worth knowing, each of which changes the answer:

**Cancellations are never deleted.** Invoice numbers starting with `C` are
refunds. Deleting them removes the refund but keeps the original order, which
invents revenue that never happened — 1,382 customers and £528,823 in this
data. Customer 12346 ordered 74,215 jars and cancelled all of them 16 minutes
later; delete the `C` row and they look like a £77,183 customer instead of a £0
one. Instead, Recency and Frequency come from purchases only, while Monetary
sums everything so refunds subtract.

**Outliers are kept and log-transformed, not removed.** The IQR rule flags 555
customers (14.2%), but these are wholesale buyers ordering ~3.5x the units per
line — a real segment, and the one the analysis exists to find. `log1p` drops
Monetary's skew from 23.2 to 0.35 without deleting anyone.

**IQR over Z-score for outlier detection.** Z-score uses the mean and standard
deviation, both inflated by the outliers themselves. On this data it flags 69
customers against IQR's 555, and flags *zero* on Recency — the variable is
bounded at 374 days, so it can never reach z > 3.

## Pipeline

```
featurize  ->  train  ->  evaluate
```

| stage | in | out |
|---|---|---|
| `featurize` | `Online Retail.xlsx` | `data/target_dataset.csv`, `data/features.csv` |
| `train` | `data/features.csv` | `models/kmeans.joblib` |
| `evaluate` | model + features | `metrics.json`, `reports/cluster_profile.csv`, `data/segmented_customers.csv` |

Splitting `featurize` out matters: DVC caches it, so changing `k` re-runs only
train and evaluate — about 11 seconds instead of 90.

## Running it

```powershell
uv sync                     # install dependencies
uv run dvc repro            # run the pipeline
uv run dvc metrics show     # see the metrics
uv run dvc dag              # draw the pipeline
```

Always prefix with `uv run`. DVC launches the stages as subprocesses, and bare
`python` resolves to the system interpreter, which has none of the dependencies.

### Experiments

```powershell
uv run python src/train.py model.n_clusters=6
uv run python src/evaluate.py model.n_clusters=6

uv run python src/train.py --multirun model.n_clusters=3,4,5,6,7,8
```

Every setting in `conf/config.yaml` can be overridden this way. Runs are logged
to MLflow on DagsHub.

### Credentials

```powershell
copy .env.example .env
```

Then put your DagsHub token in `.env`. It is gitignored. Without it, MLflow
falls back to a local SQLite database.

## Layout

```
conf/config.yaml          all settings; also read by DVC as its params file
src/featurize.py          cleaning + RFM aggregation
src/train.py              K-Means fit
src/evaluate.py           metrics, cluster profile, MLflow logging
preprocessing.ipynb       exploratory analysis the pipeline was derived from
dvc.yaml                  stage definitions
metrics.json              tracked in git so `dvc metrics diff` works
```

`preprocessing.ipynb` and `src/` contain the same logic. The notebook is the
exploration and the reasoning; `src/` is what actually produces the results.
If they disagree, `src/` is authoritative.
