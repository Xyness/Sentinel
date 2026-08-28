"""Every call that leaves the process."""

from __future__ import annotations

import os
from typing import Any

import httpx

DEFAULT_API = os.environ.get("API_BASE_URL", "http://localhost:8000")

FEATURE_FIELDS = (
    "z_score_price",
    "z_score_log_return",
    "z_score_volume",
    "rolling_price_std",
    "rolling_volume_std",
)


class ApiError(RuntimeError):
    """The API could not be reached, or answered with something unusable."""


class Client:
    """The API, as five calls.

    Nothing here swallows a failure into a default. The old dashboard returned
    empty stats when the API was down, which drew a page of zeros that looked
    like a quiet market rather than an outage. A command that cannot answer
    says so and exits non-zero; `status` is the one caller that catches the
    error, because reporting the outage is its job.
    """

    # /system-status probes Spark, Kafka and Zookeeper one after another with
    # its own timeouts, which comes to about seven seconds when all three are
    # down. Ten leaves no margin for the outage it exists to report.
    def __init__(self, base_url: str = DEFAULT_API, timeout: float = 20.0, transport=None):
        self.base_url = base_url.rstrip("/")
        self._http = httpx.Client(base_url=self.base_url, timeout=timeout, transport=transport)

    def __enter__(self) -> Client:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def close(self) -> None:
        self._http.close()

    # -- calls -------------------------------------------------------------

    def health(self) -> dict[str, Any]:
        return self._get("/health")

    def system_status(self) -> dict[str, Any]:
        return self._get("/system-status")

    def stats(self) -> dict[str, Any]:
        return self._get("/stats")

    def model_info(self) -> dict[str, Any]:
        return self._get("/model-info")

    def latest_predictions(self, limit: int = 100, symbol: str | None = None,
                           after: int | None = None) -> list[dict[str, Any]]:
        return self._get("/latest-predictions", limit=limit, symbol=symbol, after=after)

    def predict(self, features: dict[str, Any]) -> dict[str, Any]:
        return self._request("POST", "/predict", json=features)

    # -- plumbing ----------------------------------------------------------

    def _get(self, path: str, **params) -> Any:
        return self._request("GET", path, params={k: v for k, v in params.items()
                                                  if v is not None})

    def _request(self, method: str, path: str, **kwargs) -> Any:
        try:
            response = self._http.request(method, path, **kwargs)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as error:
            raise ApiError(_explain(error.response)) from error
        except httpx.HTTPError as error:
            raise ApiError(f"{self.base_url} is not answering: {error}") from error
        except ValueError as error:
            raise ApiError(f"{self.base_url}{path} did not return JSON") from error


def _explain(response: httpx.Response) -> str:
    """The API's own reason for a refusal, rather than the status code alone.

    503 from /predict is the ordinary case on a cold start: the model has not
    finished training. Printing the detail turns a number into that sentence.
    """
    try:
        detail = response.json().get("detail")
    except ValueError:
        detail = None
    if isinstance(detail, list):
        detail = "; ".join(str(item.get("msg", item)) for item in detail)
    if detail:
        return f"{response.status_code} {detail}"
    return f"{response.request.url.path} returned {response.status_code}"
