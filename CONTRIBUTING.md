# Contributing to Sentinel

Thank you for your interest in contributing to Sentinel! This guide will help you get started.

## Prerequisites

- [Docker](https://docs.docker.com/get-docker/) & Docker Compose
- Python 3.11+
- Git

## Getting Started

### 1. Fork & clone

```bash
git clone https://github.com/<your-username>/Sentinel.git
cd Sentinel
```

### 2. Run the full stack with Docker

```bash
docker compose up --build
```

### 3. Run unit tests

```bash
pip install -r tests/requirements.txt
pytest
```

Tests run without Docker/Kafka/Spark — they are pure unit tests.

## Project Structure

| Directory          | Description                              |
|--------------------|------------------------------------------|
| `api/`             | FastAPI anomaly detection service         |
| `dashboard/`       | Streamlit visualization (4 pages)         |
| `data-generator/`  | Simulated & Binance market data producers |
| `ml-python/`       | Isolation Forest training & evaluation    |
| `spark-java/`      | Spark Structured Streaming (Java/Maven)   |
| `docker/`          | Dockerfiles for each service              |
| `tests/`           | Unit tests (pytest)                       |
| `docs/`            | Technical report & design decisions       |

## Commit Conventions

We follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <short description>
```

**Types**: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`, `ci`

**Scopes**: `api`, `dashboard`, `generator`, `spark`, `ml`, `docker`, `tests`, `docs`

Examples:

```
feat(api): add batch prediction endpoint
fix(dashboard): correct chart refresh interval
docs(readme): update architecture diagram
test(ml): add scaler normalization tests
```

## Pull Request Process

1. Create a feature branch from `main`:
   ```bash
   git checkout -b feat/my-feature
   ```

2. Make your changes and add tests if applicable.

3. Run the test suite:
   ```bash
   pytest
   ```

4. Update `CHANGELOG.md` under the `[Unreleased]` section.

5. Open a Pull Request against `main` using the PR template.

6. Wait for CI to pass and a maintainer review.

## Coding Standards

### Python
- Follow [PEP 8](https://peps.python.org/pep-0008/)
- Use type hints for function signatures
- Keep functions focused and under 50 lines when possible

### Java (Spark)
- Follow standard Java conventions
- Use meaningful variable names

### General
- No hardcoded credentials or secrets
- Use environment variables for configuration
- Write descriptive commit messages

## Reporting Issues

Use the [issue templates](.github/ISSUE_TEMPLATE/) to report bugs or request features.

## Code of Conduct

This project follows the [Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md). By participating, you agree to uphold it.
