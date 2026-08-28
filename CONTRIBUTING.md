# Contributing

## Running it

You need Docker and Python 3.11+.

```bash
git clone https://github.com/<your-username>/Sentinel.git
cd Sentinel
docker compose up --build
```

The tests don't need any of that running:

```bash
pip install -r tests/requirements.txt
pytest
```

They're pure unit tests: no Kafka, no Spark, no network. If you find yourself needing to
spin up infrastructure to test something, that's usually a sign the logic wants pulling out
into a function that takes plain data.

## Where things are

`data-generator/` produces market events, `spark-java/` consumes them and does the feature
engineering (Java, built with Maven), `ml-python/` trains the model, `api/` serves it, `scorer/`
puts the rows Spark wrote back through it, and `cli/` is the terminal client. Dockerfiles are in
`docker/`, one per service; the client is not one of them, it installs on your machine and talks
to the API over HTTP.

The awkward seam is between Spark and the training job: they only talk through Parquet files
on a shared volume, so if you change the feature set you have to change it in
`RollingFeatures.java`, in `preprocess.py`, and in the API schema, or things fail in
confusing ways at inference time.

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
and open the PR. If you touched the model or the features, say what it did to the results.
A change that improves precision and quietly destroys recall looks identical in a diff.

## Issues

Bugs: what you did, what happened, what you expected. If it's about a detection, the input
feature vector is the useful part.
