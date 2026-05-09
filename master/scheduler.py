"""
FastAPI control plane plus **Scheduler** — XADD to Redis Streams, consumer-group setup, result polling.

Redis acts as the **message broker** (typically deployed on AWS EC2/VPC/ElastiCache); set **REDIS_URL**
to match that broker — not necessarily the nginx edge host.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from contextlib import asynccontextmanager

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from common.config import load_env

load_env()

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from common.models import TaskRequest, TaskResponse, TaskStatus
from common.redis_io import (
    STREAM_KEY,
    connect,
    enqueue_task as redis_enqueue_task,
    ensure_consumer_group,
    get_meta_json,
    get_result_json,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Scheduler:
    """
    Master-side task scheduling: persists queue metadata and **XADD** payloads
    to the Redis Stream (`tasks:stream`). Workers read via **consumer groups** (PEL).
    """

    def __init__(self, redis_url: str | None = None) -> None:
        url = redis_url if redis_url is not None else os.environ.get("REDIS_URL", "redis://127.0.0.1:6379/0")
        self.redis_url = url
        self._redis = connect(url)
        ensure_consumer_group(self._redis)
        logger.info("Scheduler ready; stream=%s url=%s", STREAM_KEY, url)

    @property
    def redis(self):  # noqa: ANN201
        return self._redis

    def enqueue_task(self, req: TaskRequest) -> str:
        """Schedule work: **XADD** stream + queued meta. Returns **task_id**."""
        tid = redis_enqueue_task(self._redis, req)
        logger.debug("XADD task_id=%s", tid)
        return tid

    def read_result_json(self, task_id: str) -> str | None:
        return get_result_json(self._redis, task_id)

    def read_meta_json(self, task_id: str) -> str | None:
        return get_meta_json(self._redis, task_id)


_scheduler: Scheduler | None = None


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global _scheduler
    _scheduler = Scheduler()
    logger.info("Scheduler connected to Redis; stream=%s", STREAM_KEY)
    yield


app = FastAPI(title="Distributed AI Control Plane", version="2.0.0", lifespan=lifespan)


class SubmitBody(BaseModel):
    query: str
    metadata: dict = Field(default_factory=dict)


@app.post("/task")
def submit_task(body: SubmitBody) -> dict:
    assert _scheduler is not None
    req = TaskRequest(query=body.query, metadata=body.metadata)
    task_id = _scheduler.enqueue_task(req)
    logger.info("Enqueued task_id=%s", task_id)
    return {"task_id": task_id, "stream": STREAM_KEY}


@app.get("/task/{task_id}")
def get_task(task_id: str) -> dict:
    assert _scheduler is not None
    result_raw = _scheduler.read_result_json(task_id)
    if result_raw:
        return TaskResponse.from_json(result_raw).to_dict()

    meta_raw = _scheduler.read_meta_json(task_id)
    if not meta_raw:
        raise HTTPException(status_code=404, detail="Unknown task_id")

    try:
        meta = json.loads(meta_raw)
        status = meta.get("status", TaskStatus.QUEUED.value)
        return {
            "task_id": task_id,
            "answer": meta.get("answer", ""),
            "status": status,
            "context_snippets": meta.get("context_snippets") or [],
            "error": meta.get("error"),
        }
    except (json.JSONDecodeError, KeyError) as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/health")
def health() -> dict:
    try:
        if _scheduler is None:
            return {"ok": False, "redis": False, "error": "not connected"}
        _scheduler.redis.ping()
        return {"ok": True, "redis": True}
    except Exception as e:
        logger.exception("health fail")
        return {"ok": False, "redis": False, "error": str(e)}
