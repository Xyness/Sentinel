# Scorer

Reads the feature rows Spark writes and sends them to the API, so the prediction
history fills from the pipeline rather than only from whoever happens to call
the endpoint by hand.

It is the piece that makes `sentinel feed` show live data.

## How it works

Poll the features directory, take any Parquet file it has not read, pull the five
feature columns out of it, and POST the whole file to `/predict/batch` in one
call. The symbol comes off the path: Spark partitions by it, so it is a directory
name rather than a column, and reading a single file on its own would otherwise
lose it.

It used to POST a row at a time, which was one HTTP round trip and one
single-row scaler call per window. Files larger than `BATCH_SIZE` are split, and
the API answers in the order it was given, so nothing has to be matched back up.

Rows with a null in them are dropped. They come from windows Spark could not
complete, the API would reject them, and a partial vector is not a prediction
worth having.

While training is still running the API answers 503, and the file is left
unmarked so it gets picked up on the next pass instead of being lost.

## Configuration

| variable | default | |
|---|---|---|
| `API_BASE_URL` | `http://localhost:8000` | where the API is |
| `FEATURES_PATH` | `data/features` | what Spark writes to |
| `SCORE_INTERVAL_SECONDS` | `15` | how often to look for new files |

## Running it on its own

```bash
pip install -r requirements.txt
API_BASE_URL=http://localhost:8000 FEATURES_PATH=../data/features python scorer.py
```

## What it does not do

It keeps the set of files it has read in memory, so restarting it replays
whatever is still on disk. Remembering a cursor across restarts would need a
writable volume, and now that the predictions themselves are in the API's
database a replay costs duplicate rows rather than a gap.
