"""
GPU / Colab workers: **GPUWorker** pulls from Redis Streams (`XREADGROUP`), runs FAISS RAG + Ollama,
**XACK** after completion.

**Fault tolerance (PEL)**:
  Unacknowledged messages remain in the group's **Pending Entries List**. Stale entries
  (**idle** > threshold) are reclaimed with **XAUTOCLAIM** so another consumer can finish the work.
"""

from __future__ import annotations

import logging
import os
import socket
import sys
import time
import uuid

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from common.config import load_env

load_env()

from common.redis_io import CONSUMER_GROUP, STREAM_KEY

logger = logging.getLogger(__name__)

CLAIM_MIN_IDLE_MS = int(os.environ.get("REDIS_CLAIM_IDLE_MS", "120000"))
BLOCK_MS = int(os.environ.get("REDIS_BLOCK_MS", "5000"))


def _default_consumer_name() -> str:
    return os.environ.get("WORKER_ID", f"{socket.gethostname()}-{uuid.uuid4().hex[:8]}")


class GPUWorker:
    """
    Consumer-group worker: **XREADGROUP … BLOCK** new messages, **XAUTOCLAIM** stuck PEL entries.
    """

    def __init__(
        self,
        redis_url: str | None = None,
        consumer_name: str | None = None,
    ) -> None:
        self.redis_url = redis_url or os.environ.get("REDIS_URL", "redis://127.0.0.1:6379/0")
        self.consumer_name = consumer_name or _default_consumer_name()

    def _ensure_group(self, r) -> None:
        try:
            r.xgroup_create(STREAM_KEY, CONSUMER_GROUP, id="0", mkstream=True)
            logger.info("Created stream %s group %s", STREAM_KEY, CONSUMER_GROUP)
        except Exception as e:
            if "BUSYGROUP" not in str(e):
                logger.debug("xgroup_create: %s", e)

    def _process_message(self, r, fields: dict, engine) -> None:
        from common.models import TaskRequest, TaskResponse, TaskStatus
        from common.redis_io import set_processing, store_result
        from rag.retriever import FaissRetriever

        raw = fields.get("payload")
        if not raw:
            raise ValueError("missing payload")

        req = TaskRequest.from_json(raw)
        tid = req.task_id
        if not tid:
            raise ValueError("missing task_id")

        set_processing(r, tid)
        retriever = FaissRetriever()
        contexts = retriever.retrieve_context(req.query, k=int(os.environ.get("RAG_TOP_K", "4")))
        answer = engine.answer_with_rag(
            req.query,
            contexts,
            max_new_tokens=int(os.environ.get("LLM_MAX_NEW_TOKENS", "256")),
            temperature=float(os.environ.get("LLM_TEMPERATURE", "0.7")),
        )
        tr = TaskResponse(
            task_id=tid,
            answer=answer,
            status=TaskStatus.DONE,
            context_snippets=contexts,
        )
        store_result(r, tr)
        logger.info("Done task_id=%s", tid)

    def _handle_error(self, r, raw_payload: str | None, exc: Exception) -> None:
        from common.models import TaskRequest, TaskResponse, TaskStatus
        from common.redis_io import store_result

        tid = None
        try:
            if raw_payload:
                req = TaskRequest.from_json(raw_payload)
                tid = req.task_id
        except Exception:
            pass
        if not tid:
            logger.warning("Unrecoverable task error: %s", exc)
            return
        store_result(
            r,
            TaskResponse(task_id=tid, answer="", status=TaskStatus.ERROR, error=str(exc)),
        )

    def reclaim_stale_pending(self, r, engine) -> None:
        """PEL recovery: idle pending messages → this consumer (**XAUTOCLAIM**), then process + **XACK**."""
        import redis

        try:
            claim = r.xautoclaim(
                STREAM_KEY,
                CONSUMER_GROUP,
                self.consumer_name,
                CLAIM_MIN_IDLE_MS,
                "0-0",
                count=50,
            )
        except redis.ResponseError as e:
            if "NOGROUP" in str(e):
                self._ensure_group(r)
            else:
                logger.debug("xautoclaim: %s", e)
            return

        if not claim or len(claim) < 2:
            return
        messages = claim[1] or []
        for item in messages:
            if not item or len(item) < 2:
                continue
            msg_id, fields = item[0], item[1]
            raw = fields.get("payload") if isinstance(fields, dict) else None
            try:
                self._process_message(r, fields, engine)
                r.xack(STREAM_KEY, CONSUMER_GROUP, msg_id)
            except Exception as e:
                logger.exception("Reclaimed message failed id=%s: %s", msg_id, e)
                self._handle_error(r, raw, e)
                try:
                    r.xack(STREAM_KEY, CONSUMER_GROUP, msg_id)
                except Exception:
                    pass

    def poll_new_messages(self, r, engine) -> None:
        """Long poll: **XREADGROUP … BLOCK** for `>` deliveries."""
        import redis

        streams = r.xreadgroup(
            CONSUMER_GROUP,
            self.consumer_name,
            {STREAM_KEY: ">"},
            count=1,
            block=BLOCK_MS,
        )
        if not streams:
            return
        for _sname, messages in streams:
            for msg_id, fields in messages:
                raw = fields.get("payload") if isinstance(fields, dict) else None
                try:
                    self._process_message(r, fields, engine)
                    r.xack(STREAM_KEY, CONSUMER_GROUP, msg_id)
                except Exception as e:
                    logger.exception("Task failed id=%s: %s", msg_id, e)
                    self._handle_error(r, raw, e)
                    try:
                        r.xack(STREAM_KEY, CONSUMER_GROUP, msg_id)
                    except Exception:
                        pass

    def run_forever(self) -> None:
        import redis

        from common.redis_io import connect
        from llm.inference import get_engine

        r = connect(self.redis_url)
        self._ensure_group(r)
        logger.info(
            "GPUWorker consumer=%s stream=%s group=%s Redis=%s",
            self.consumer_name,
            STREAM_KEY,
            CONSUMER_GROUP,
            self.redis_url,
        )

        engine = get_engine()

        while True:
            try:
                self.reclaim_stale_pending(r, engine)
                self.poll_new_messages(r, engine)
            except KeyboardInterrupt:
                logger.info("Interrupted — exiting")
                break
            except redis.ConnectionError as e:
                logger.warning("Redis connection: %s — retry", e)
                time.sleep(2)
            except Exception:
                logger.exception("Worker loop error; backing off")
                time.sleep(2)


def worker_loop() -> None:
    GPUWorker().run_forever()


def _ray_num_gpus() -> float:
    v = os.environ.get("RAY_NUM_GPUS", "1")
    try:
        return float(v)
    except ValueError:
        return 1.0


_ray_pkg = None
ray_gpu_worker = None

try:
    import ray as _ray_pkg

    @_ray_pkg.remote(num_gpus=_ray_num_gpus())
    def _ray_gpu_worker_impl() -> None:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
        worker_loop()

    ray_gpu_worker = _ray_gpu_worker_impl
except ImportError:
    pass


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    use_ray = os.environ.get("RAY_DISABLE", "").lower() not in ("1", "true", "yes")

    if use_ray and _ray_pkg is not None and callable(ray_gpu_worker):
        _ray_pkg.init(ignore_reinit_error=True)
        _ray_pkg.get(ray_gpu_worker.remote())
        return

    worker_loop()


if __name__ == "__main__":
    main()
