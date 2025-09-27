import time
import json
import statistics
from typing import Tuple
import requests


API_BASE = "http://localhost:8000"


def time_request(method: str, path: str, **kwargs) -> Tuple[float, int, dict]:
    url = f"{API_BASE}{path}"
    start = time.perf_counter()
    resp = requests.request(method, url, timeout=60, **kwargs)
    elapsed = (time.perf_counter() - start) * 1000.0
    payload = {}
    try:
        payload = resp.json()
    except Exception:
        payload = {"text": resp.text[:2000]}
    return elapsed, resp.status_code, payload


def bench_query(iterations=5, k=10):
    latencies = []
    for i in range(iterations):
        ms, code, _ = time_request(
            "POST",
            "/query",
            json={"query": "What are common API testing defects?", "k": k},
        )
        print(f"Query {i+1}/{iterations}: {ms:.1f} ms (HTTP {code})")
        latencies.append(ms)
    p50 = statistics.median(latencies)
    p95 = statistics.quantiles(latencies, n=20)[18] if len(latencies) >= 2 else latencies[-1]
    return {"iterations": iterations, "k": k, "p50_ms": p50, "p95_ms": p95, "samples": latencies}


def main():
    print("Checking health...")
    ms, code, health = time_request("GET", "/health")
    print(f"Health: {code} in {ms:.1f} ms -> {health.get('status')}")

    print("Checking status...")
    ms, code, status = time_request("GET", "/status")
    print(f"Status: {code} in {ms:.1f} ms -> ready={status.get('system_ready')}")

    print("Benchmarking /query...")
    result = bench_query()
    print("\nResults:")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
