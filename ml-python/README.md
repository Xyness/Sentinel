# Training

Reads the features Spark wrote and fits the Isolation Forest the API serves.

Batch, not streaming. The model is fitted once over whatever has accumulated,
written to a joblib bundle, and the API picks it up from there. Nothing retrains
on its own yet.

## What it does

`load_dataset.py` globs every Parquet file under `FEATURES_PATH` and concatenates
them, skipping the zero-byte ones Spark leaves behind on empty batches.

`preprocess.py` drops rows with a null in any feature and splits off the labels
if there are any. The simulator marks the anomalies it injects, so simulated runs
are labelled and Binance runs are not. Labels are used to score the model and
never to fit it: the fit is unsupervised either way, which is the point.

`train_isolation_forest.py` puts a StandardScaler in front of an Isolation Forest
with 200 estimators and 1% contamination, and writes `{"model", "scaler"}` to the
path in `MODEL_PATH`. On labelled data it holds out 20% and prints a
classification report. On unlabelled data there is nothing to report against, and
it says so instead of inventing a number.

`train_runner.py` is what the container runs: it waits for `MIN_PARQUET_FILES` to
appear before starting, gives up after `MAX_WAIT_SECONDS`, then trains once and
exits.

`evaluation/evaluate.py` is not part of that. It is a hand-run tool that scores an
already-trained model against the features on disk, for when you want to know
whether a change helped without waiting for a full stack restart:

```bash
python evaluation/evaluate.py
```

## Configuration

| variable | default | |
|---|---|---|
| `FEATURES_PATH` | `data/features` | what Spark writes to |
| `MODEL_PATH` | `models/isolation_forest.joblib` | where the bundle goes |
| `MIN_PARQUET_FILES` | `3` | how much data before training starts |
| `CHECK_INTERVAL_SECONDS` | `15` | how often to look while waiting |
| `MAX_WAIT_SECONDS` | `600` | give up waiting |

## Running it on its own

```bash
pip install -r requirements.txt
FEATURES_PATH=../data/features MODEL_PATH=../models/isolation_forest.joblib \
  python train_runner.py
```

## The awkward part

This talks to Spark only through Parquet files on a shared volume, so the feature
set is written down in three places: `RollingFeatures.java`, `preprocess.py` and
the API schema. Change one and the other two fail at inference time, confusingly.
