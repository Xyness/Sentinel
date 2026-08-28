# Training

Reads the features Spark wrote and fits the Isolation Forest the API serves.

Batch training, streaming inference. The model is fitted over whatever has
accumulated, written to a joblib bundle, and the API picks it up from there.

## What it does

`load_dataset.py` globs every Parquet file under `FEATURES_PATH` and concatenates
them, skipping the zero-byte ones Spark leaves behind on empty batches.

`preprocess.py` drops rows with a null in any feature and splits off the labels
if there are any. The simulator marks the anomalies it injects, so simulated runs
are labelled and Binance runs are not. Labels are used to set the contamination
and to score the model, never to fit it: the fit is unsupervised either way,
which is the point. A feature store that doesn't carry the columns this expects
is an error naming the columns, rather than a KeyError three frames down.

`train_isolation_forest.py` puts a StandardScaler in front of an Isolation Forest
with 200 estimators, and writes `{"model", "scaler", "features", "metrics"}` to
the path in `MODEL_PATH`. On labelled data it holds out 20 % and prints a
classification report. On unlabelled data there is nothing to report against, and
it says so instead of inventing a number.

`train_runner.py` is what the container runs: it waits for `MIN_PARQUET_FILES` to
appear, trains, then keeps going and refits every `RETRAIN_INTERVAL_SECONDS`.

## Contamination is measured, not assumed

It used to be pinned at 1 %. The simulator draws once per event, so a one-minute
window at one event a second is 60 draws and a per-event 0.01 marked 45 % of
windows anomalous. Telling the forest to flag 1 % of rows while 45 % carried the
label caps recall at about 2 %, whatever the features do.

So it's read off the labels, clamped to `[0.005, 0.25]`. Outside that range the
number says the generator wants turning down rather than the model, and the run
logs a warning saying so. With no labels there is nothing to measure and
`CONTAMINATION` is used as the stated assumption it is.

## Retraining

The first model out of a cold stack is fitted on the two minutes of data that
existed when it started. It used to be the model served for the rest of the run,
because the job trained once and exited.

Now it loops. Each round checks whether the feature store actually moved and
skips the run if it didn't. A refit that throws is logged and slept off: the
model on disk is still the last one that worked and the API is still serving it,
so dying here would trade a stale model for no model.

The bundle is written beside the live path and renamed over it. The API reloads
on the file's timestamp, and a `joblib.dump` straight onto that path would let it
read a half-written bundle.

## What the bundle carries

Beyond the model and the scaler: the feature names it was fitted on, and the
metrics the run measured. `/model-info` serves those and `sentinel status`
prints them. A precision that only ever existed in a log nobody kept is a
precision nobody can check.

```bash
python evaluation/evaluate.py
```

`evaluation/evaluate.py` prints the holdout out of the bundle, then scores the
model over everything currently on disk. Those two numbers are not
interchangeable and it says which is which: the second one includes rows the
model was fitted on.

## Configuration

| variable | default | |
|---|---|---|
| `FEATURES_PATH` | `data/features` | what Spark writes to |
| `MODEL_PATH` | `models/isolation_forest.joblib` | where the bundle goes |
| `MIN_PARQUET_FILES` | `3` | how much data before training starts |
| `CHECK_INTERVAL_SECONDS` | `15` | how often to look while waiting |
| `MAX_WAIT_SECONDS` | `600` | give up waiting |
| `RETRAIN_INTERVAL_SECONDS` | `300` | 0 trains once and exits |
| `CONTAMINATION` | `0.05` | only used when there are no labels |
| `N_ESTIMATORS` | `200` | trees in the forest |

## Running it on its own

```bash
pip install -r requirements.txt
FEATURES_PATH=../data/features MODEL_PATH=../models/isolation_forest.joblib \
  RETRAIN_INTERVAL_SECONDS=0 python train_runner.py
```

## The awkward part

This talks to Spark only through Parquet files on a shared volume, so the feature
set is written down in `FeatureAssembler.java`, `preprocess.py`, `api/store.py`
and the scorer. They have to agree, and `preprocess` is the one that will tell
you when they don't.
