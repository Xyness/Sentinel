# Contributing

## Running it

You need Docker and Python 3.11+.

```bash
git clone https://github.com/<your-username>/Sentinel.git
cd Sentinel
docker compose up --build
```

The unit tests don't need any of that running:

```bash
pip install -r tests/requirements.txt
pytest
```

They're pure unit tests: no Kafka, no Spark, no network. If you find yourself needing to
spin up infrastructure to test something, that's usually a sign the logic wants pulling out
into a function that takes plain data.

The Spark tests do stand a local SparkSession up, over batch DataFrames rather than streams:

```bash
cd spark-java && mvn test
```

And there's an end to end check that wants the stack actually running. CI runs it on
twenty-second windows so it fits inside a job:

```bash
docker compose up -d --build
python scripts/smoke_test.py
```

## Where things are

`data-generator/` produces market events, `spark-java/` consumes them and does the feature
engineering (Java, built with Maven), `ml-python/` trains the model, `api/` serves it and stores
what it scored, `scorer/` puts the rows Spark wrote back through it, and `cli/` is the terminal
client. Dockerfiles are in `docker/`, one per service; the client is not one of them, it installs
on your machine and talks to the API over HTTP.

The awkward seam is between Spark and everything downstream: they only talk through Parquet files
on a shared volume, so the feature set is written down in four places. Change one and change all
four:

- `spark-java/.../FeatureAssembler.java`, which has the list as a constant
- `ml-python/training/preprocess.py`
- `api/store.py`, which is where the API's schema and its table both get it from
- `scorer/scorer.py`

`preprocess` raises naming the columns it expected when they drift, and `scripts/smoke_test.py`
compares the list the model was fitted on against the one it expects, so a drift shows up in CI
rather than as a confusing 422 at inference time.

## If you touch the features

Read `spark-java/README.md` first, the section on what was wrong before. The short version is
that a feature normalised by a statistic the anomaly itself moves cannot see the anomaly, and
that mistake took a while to spot because everything downstream of it looked fine.

New features want to be dimensionless and order independent. Dimensionless because one model
covers three pairs two orders of magnitude apart in price. Order independent because an
aggregation shuffles, so `first()` and `last()` are not the row you think they are.

And say what your change did to the results. Training puts precision, recall and f1 in the
bundle, so `sentinel status` after a run is the number to quote. A change that improves precision
and quietly destroys recall looks identical in a diff.

## Style

Python: PEP 8, type hints on function signatures. Java: whatever your IDE does by default.

Keep configuration in environment variables. Every service reads its settings that way and
adding a hardcoded host or credential breaks the Compose setup for everyone.

## Commits and PRs

Conventional Commits, with the component as the scope:

```
feat(api): add batch prediction endpoint
fix(cli): keep the feed columns aligned on wide symbols
test(ml): add scaler normalization tests
```

Branch off `main`, keep `pytest` green, add a line to `CHANGELOG.md` under `[Unreleased]`,
and open the PR.

## Issues

Bugs: what you did, what happened, what you expected. If it's about a detection, the input
feature vector is the useful part.
