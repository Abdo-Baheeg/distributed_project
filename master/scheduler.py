"""
FastAPI control plane: enqueue tasks to Redis Streams, poll results.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from contextlib import asynccontextmanager

# Allow `python master/scheduler.py` and `uvicorn master.scheduler:app`
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from common.models import TaskRequest, TaskResponse, TaskStatus
from common.redis_io import (
    STREAM_KEY,
    connect,
    enqueue_task,
    ensure_consumer_group,
    get_meta_json,
    get_result_json,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_redis_url = os.environ.get("REDIS_URL", "redis://127.0.0.1:6379/0")
_r = None


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global _r
    _r = connect(_redis_url)
    ensure_consumer_group(_r)
    logger.info("Scheduler connected to Redis; stream=%s", STREAM_KEY)
    yield


app = FastAPI(title="Distributed AI Control Plane", version="1.0.0", lifespan=lifespan)


class SubmitBody(BaseModel):
    query: str
    metadata: dict = Field(default_factory=dict)


@app.post("/task")
def submit_task(body: SubmitBody) -> dict:
    assert _r is not None
    req = TaskRequest(query=body.query, metadata=body.metadata)
    task_id = enqueue_task(_r, req)
    logger.info("Enqueued task_id=%s", task_id)
    return {"task_id": task_id, "stream": STREAM_KEY}


@app.get("/task/{task_id}")
def get_task(task_id: str) -> dict:
    """Return final result if ready, else status from meta."""
    assert _r is not None
    result_raw = get_result_json(_r, task_id)
    if result_raw:
        tr = TaskResponse.from_json(result_raw)
        return tr.to_dict()

    meta_raw = get_meta_json(_r, task_id)
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
        if _r is None:
            return {"ok": False, "redis": False, "error": "not connected"}
        _r.ping()
        return {"ok": True, "redis": True}
    except Exception as e:
        logger.exception("health fail")
        return {"ok": False, "redis": False, "error": str(e)}
