"""Smoke-check the serving API: hit ``/health`` and ``/predict`` with sample inputs.

Start the API first, then run this against it::

    uv run uvicorn bike_rental.serving.app:app --port 8000   # in one terminal
    uv run python scripts/check_api.py [base_url]            # in another

``base_url`` defaults to ``http://localhost:8000``. Exits non-zero if any check
fails (server down, /predict error), so it doubles as a CI/health probe.
"""

import json
import sys
import urllib.error
import urllib.request

BASE_URL = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"

# Future-dated queries (2013 — past the 2011–2012 training span, so the model
# extrapolates the growth trend): summer holiday / winter night / mild evening.
SAMPLE_REQUESTS = [
    {"timestamp": "2013-07-04T08:00:00", "conditions": "clouds", "temperature_c": 25,
     "perceived_temperature_c": 27, "humidity": 60, "windspeed_kmh": 12},
    {"timestamp": "2013-01-15T03:00:00", "conditions": "heavy_rain", "temperature_c": -2,
     "perceived_temperature_c": -6, "humidity": 90, "windspeed_kmh": 30},
    {"timestamp": "2013-09-12T18:00:00", "conditions": "clear", "temperature_c": 22,
     "perceived_temperature_c": 22, "humidity": 50, "windspeed_kmh": 8},
]


def _get(path: str) -> tuple[int, dict]:
    """GET ``path`` and return ``(status_code, parsed_json)``."""
    with urllib.request.urlopen(f"{BASE_URL}{path}", timeout=10) as response:
        return response.status, json.loads(response.read())


def _post(path: str, payload: dict) -> tuple[int, dict]:
    """POST ``payload`` as JSON to ``path``; return ``(status_code, parsed_json)``."""
    request = urllib.request.Request(
        f"{BASE_URL}{path}",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return response.status, json.loads(response.read())


def main() -> int:
    """Run the health and prediction checks; return a process exit code."""
    print(f"API @ {BASE_URL}")

    try:
        status, health = _get("/health")
    except urllib.error.URLError as error:
        print(f"  /health unreachable: {error} — is the server running?")
        return 1
    print(f"  /health [{status}] model v{health['model_version']} "
          f"data_commit={(health['data_commit'] or '')[:12]}")

    failed = False
    for payload in SAMPLE_REQUESTS:
        try:
            status, body = _post("/predict", payload)
            print(f"  /predict [{status}] {payload['timestamp']} {payload['conditions']:>10} "
                  f"-> {body['predicted_rentals']:>4} rentals")
        except urllib.error.HTTPError as error:
            print(f"  /predict FAILED [{error.code}] {payload['timestamp']}: "
                  f"{error.read().decode()[:200]}")
            failed = True

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
