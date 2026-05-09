#!/usr/bin/env python3
"""
Entry point (PDF): `python main.py [api|worker|load]`

- api: run FastAPI control plane (or use: uvicorn master.scheduler:app --host 0.0.0.0 --port 8000)
- worker: run GPU worker (set RAY_DISABLE=1 for plain process)
- load: run threaded load generator
"""

from __future__ import annotations

import argparse
import os
import sys


def main() -> None:
    root = os.path.dirname(os.path.abspath(__file__))
    if root not in sys.path:
        sys.path.insert(0, root)

    from common.config import load_env

    load_env()

    p = argparse.ArgumentParser(description="Distributed AI system")
    p.add_argument("mode", choices=["api", "worker", "load"], nargs="?", default="api")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    args, rest = p.parse_known_args()

    if args.mode == "api":
        import uvicorn

        uvicorn.run("master.scheduler:app", host=args.host, port=args.port, reload=False)
    elif args.mode == "worker":
        from workers.gpu_worker import main as worker_main

        worker_main()
    else:
        from client.load_generator import main as load_main

        sys.argv = [sys.argv[0]] + rest
        load_main()


if __name__ == "__main__":
    main()
