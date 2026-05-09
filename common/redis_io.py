"""
Redis Streams wiring: task queue, consumer group, result keys.

Stream payload fields:
  - payload: JSON-serialized TaskRequest (with task_id set)
"""

from __future__ import annotations

import logging
import uuid

import redis

from common.models import TaskRequest, TaskResponse, TaskStatus, meta_key, result_key

logger = logging.getLogger(__name__)

STREAM_KEY = "tasks:stream"
CONSUMER_GROUP = "workers"
RESULT_TTL_SEC = 86400


def connect(url: str | None = None) -> redis.Redis:
    import os

    u = url or os.environ.get("REDIS_URL", "redis://127.0.0.1:6379/0")
    return redis.from_url(u, decode_responses=True)


def ensure_consumer_group(r: redis.Redis) -> None:
    try:
        r.xgroup_create(STREAM_KEY, CONSUMER_GROUP, id="0", mkstream=True)
        logger.info("Created stream and consumer group %s", CONSUMER_GROUP)
    except redis.ResponseError as e:
        if "BUSYGROUP" in str(e):
            return
        raise


def enqueue_task(r: redis.Redis, req: TaskRequest) -> str:
    """XADD task; set meta queued. Returns task_id."""
    tid = req.task_id or str(uuid.uuid4())
    body = TaskRequest(query=req.query, task_id=tid, metadata=req.metadata)
    r.set(
        meta_key(tid),
        TaskResponse(
            task_id=tid,
            answer="",
            status=TaskStatus.QUEUED,
        ).to_json(),
        ex=RESULT_TTL_SEC,
    )
    r.xadd(STREAM_KEY, {"payload": body.to_json()})
    return tid


def set_processing(r: redis.Redis, task_id: str) -> None:
    r.set(
        meta_key(task_id),
        TaskResponse(
            task_id=task_id,
            answer="",
            status=TaskStatus.PROCESSING,
        ).to_json(),
        ex=RESULT_TTL_SEC,
    )


def store_result(r: redis.Redis, response: TaskResponse) -> None:
    r.set(result_key(response.task_id), response.to_json(), ex=RESULT_TTL_SEC)
    r.set(meta_key(response.task_id), response.to_json(), ex=RESULT_TTL_SEC)


def get_result_json(r: redis.Redis, task_id: str) -> str | None:
    return r.get(result_key(task_id))


def get_meta_json(r: redis.Redis, task_id: str) -> str | None:
    return r.get(meta_key(task_id))
