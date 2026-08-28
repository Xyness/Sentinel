# Sentinel CLI

The terminal client. It talks to the API over HTTP and to nothing else, so it runs from the
host against the compose stack, or against an API anywhere else with `--api`.

```bash
pip install -e .
```

That puts a `sentinel` command on your path.

## Commands

```bash
sentinel status                                # what is up, and what the model looks like
sentinel feed                                  # follow predictions as they land
sentinel stats                                 # the buffer, aggregated
sentinel predict --preset flash-crash          # score a feature vector by hand
sentinel export -f csv -o predictions.csv      # the buffer as a file
sentinel version
```

`--api URL` points it somewhere else, `API_BASE_URL` does the same from the environment, and
`--plain` drops the colour. `--json` on any command prints the raw payload instead of the
rendered view.

## Layout

```
sentinel/
  cli.py        command surface
  client.py     the API, as five calls
  render.py     everything the terminal prints
```

`client.py` never turns a failure into an empty default. A command that cannot answer says why
and exits non-zero; `status` is the one caller that catches the error, because reporting the
outage is the job it was given.

## Tests

The suite lives with the rest of the project, in `../tests/test_cli.py`. It runs offline: the
API is stood up in-process with an `httpx` mock transport, so nothing binds a port.

```bash
cd .. && pytest tests/test_cli.py
```
