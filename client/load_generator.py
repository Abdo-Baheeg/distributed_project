"""
Multi-threaded HTTP client simulating 1000+ concurrent users (POST /task, poll GET).
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import requests

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

DEFAULT_QUERIES = [
    "What is Redis Streams?",
    "Explain least connections load balancing.",
    "How does RAG improve LLM answers?",
    "What is fault tolerance in distributed systems?",
]


def _session() -> requests.Session:
    s = requests.Session()
    a = requests.adapters.HTTPAdapter(pool_connections=200, pool_maxsize=200)
    s.mount("http://", a)
    s.mount("https://", a)
    return s


def _one_user(
    base_url: str,
    user_id: int,
    timeout_submit: float,
    timeout_poll: float,
    poll_interval: float,
    max_polls: int,
    queries: list[str],
    latencies_submit: list,
    latencies_e2e: list,
    errors: defaultdict[str, int],
    lock: threading.Lock,
) -> None:
    s = _session()
    q = random.choice(queries)
    t0 = time.perf_counter()
    try:
        r = s.post(
            f"{base_url.rstrip('/')}/task",
            json={"query": q, "metadata": {"user_id": user_id}},
            timeout=timeout_submit,
        )
        t1 = time.perf_counter()
        with lock:
            latencies_submit.append(t1 - t0)
        if r.status_code != 200:
            with lock:
                errors[f"submit_{r.status_code}"] += 1
            return
        task_id = r.json().get("task_id")
        if not task_id:
            with lock:
                errors["no_task_id"] += 1
            return

        for _ in range(max_polls):
            time.sleep(poll_interval)
            gr = s.get(
                f"{base_url.rstrip('/')}/task/{task_id}",
                timeout=timeout_poll,
            )
            if gr.status_code == 404:
                with lock:
                    errors["poll_404"] += 1
                return
            if gr.status_code != 200:
                with lock:
                    errors[f"poll_{gr.status_code}"] += 1
                return
            body = gr.json()
            st = body.get("status")
            if st in ("done", "error"):
                t2 = time.perf_counter()
                with lock:
                    latencies_e2e.append(t2 - t0)
                return
        with lock:
            errors["poll_timeout"] += 1
    except Exception as e:
        with lock:
            errors[type(e).__name__] += 1


def _percentile(sorted_vals: list[float], p: float) -> float:
    if not sorted_vals:
        return 0.0
    k = (len(sorted_vals) - 1) * p / 100.0
    f = int(k)
    c = min(f + 1, len(sorted_vals) - 1)
    return sorted_vals[f] + (k - f) * (sorted_vals[c] - sorted_vals[f])


def run_load_test(
    base_url: str | None = None,
    num_users: int = 1000,
    max_workers: int = 500,
    timeout_submit: float = 30.0,
    timeout_poll: float = 120.0,
    poll_interval: float = 0.5,
    max_polls: int = 240,
    queries: list[str] | None = None,
) -> dict[str, Any]:
    base = base_url or os.environ.get("BASE_URL", "http://127.0.0.1:8000")
    qs = queries or DEFAULT_QUERIES
    latencies_submit: list[float] = []
    latencies_e2e: list[float] = []
    errors: defaultdict[str, int] = defaultdict(int)
    lock = threading.Lock()

    t_start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = [
            ex.submit(
                _one_user,
                base,
                i,
                timeout_submit,
                timeout_poll,
                poll_interval,
                max_polls,
                qs,
                latencies_submit,
                latencies_e2e,
                errors,
                lock,
            )
            for i in range(num_users)
        ]
        for _ in as_completed(futs):
            pass
    elapsed = time.perf_counter() - t_start

    latencies_submit.sort()
    latencies_e2e.sort()
    report = {
        "base_url": base,
        "num_users": num_users,
        "elapsed_sec": round(elapsed, 3),
        "throughput_rps": round(num_users / elapsed, 3) if elapsed else 0,
        "submit_p50_ms": round(_percentile(latencies_submit, 50) * 1000, 2),
        "submit_p95_ms": round(_percentile(latencies_submit, 95) * 1000, 2),
        "e2e_p50_ms": round(_percentile(latencies_e2e, 50) * 1000, 2),
        "e2e_p95_ms": round(_percentile(latencies_e2e, 95) * 1000, 2),
        "completed_e2e": len(latencies_e2e),
        "errors": dict(errors),
    }
    return report


def main() -> None:
    p = argparse.ArgumentParser(description="Distributed AI load generator")
    p.add_argument("--base-url", default=os.environ.get("BASE_URL", "http://127.0.0.1:8000"))
    p.add_argument("--users", type=int, default=int(os.environ.get("LOAD_USERS", "1000")))
    p.add_argument("--max-workers", type=int, default=int(os.environ.get("LOAD_MAX_WORKERS", "500")))
    args = p.parse_args()
    r = run_load_test(base_url=args.base_url, num_users=args.users, max_workers=args.max_workers)
    print(r)


if __name__ == "__main__":
    main()
