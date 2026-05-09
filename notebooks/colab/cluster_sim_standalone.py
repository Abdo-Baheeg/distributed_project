"""
Inlined bundle for Ray_4GPU_Cluster.ipynb — models, Redis Streams, FAISS RAG, Ollama HTTP.

In production Redis lives on AWS VPS (optionally reachable from Colab via Cloudflare Tunnel
hostnames/VPN — see deploy/CLOUDFLARE_TUNNEL.md); set REDIS_URL accordingly.
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Mapping

import redis
import numpy as np
import faiss

logger = logging.getLogger("cluster_sim")

# --- Domain ----------------------------------------------------------
class TaskStatus(str, Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    DONE = "done"
    ERROR = "error"


STREAM_KEY = "tasks:stream"
CONSUMER_GROUP = "workers"
RESULT_TTL = 86400


@dataclass(frozen=True)
class TaskRequest:
    query: str
    task_id: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        d = asdict(self)
        d["metadata"] = dict(self.metadata)
        return json.dumps(d, ensure_ascii=False)

    @classmethod
    def from_json(cls, raw: str | bytes) -> TaskRequest:
        d = json.loads(raw)
        m = d.get("metadata") or {}
        return cls(query=d["query"], task_id=d.get("task_id"), metadata=dict(m) if isinstance(m, Mapping) else {})


@dataclass(frozen=True)
class TaskResponse:
    task_id: str
    answer: str
    status: TaskStatus
    context_snippets: list[str] = field(default_factory=list)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "answer": self.answer,
            "status": self.status.value if isinstance(self.status, TaskStatus) else str(self.status),
            "context_snippets": list(self.context_snippets),
            "error": self.error,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def from_json(cls, raw: str | bytes) -> TaskResponse:
        d = json.loads(raw)
        st = d.get("status", TaskStatus.DONE.value)
        if isinstance(st, str):
            st = TaskStatus(st)
        return cls(
            task_id=d["task_id"],
            answer=d.get("answer", ""),
            status=st,
            context_snippets=list(d.get("context_snippets") or []),
            error=d.get("error"),
        )


def result_key(task_id: str) -> str:
    return f"result:{task_id}"


def meta_key(task_id: str) -> str:
    return f"task:meta:{task_id}"


def redis_connect(url: str | None = None) -> redis.Redis:
    u = url or os.environ["REDIS_URL"]
    return redis.from_url(u, decode_responses=True)


def ensure_consumer_group(r: redis.Redis) -> None:
    try:
        r.xgroup_create(STREAM_KEY, CONSUMER_GROUP, id="0", mkstream=True)
        logger.info("Created stream group %s", CONSUMER_GROUP)
    except redis.ResponseError as e:
        if "BUSYGROUP" not in str(e):
            raise


def enqueue_task(r: redis.Redis, req: TaskRequest) -> str:
    tid = req.task_id or str(uuid.uuid4())
    body = TaskRequest(query=req.query, task_id=tid, metadata=req.metadata)
    r.set(
        meta_key(tid),
        TaskResponse(task_id=tid, answer="", status=TaskStatus.QUEUED).to_json(),
        ex=RESULT_TTL,
    )
    r.xadd(STREAM_KEY, {"payload": body.to_json()})
    return tid


def set_processing(r: redis.Redis, task_id: str) -> None:
    r.set(
        meta_key(task_id),
        TaskResponse(task_id=task_id, answer="", status=TaskStatus.PROCESSING).to_json(),
        ex=RESULT_TTL,
    )


def store_result(r: redis.Redis, tr: TaskResponse) -> None:
    r.set(result_key(tr.task_id), tr.to_json(), ex=RESULT_TTL)
    r.set(meta_key(tr.task_id), tr.to_json(), ex=RESULT_TTL)


def get_result_json(r: redis.Redis, task_id: str) -> str | None:
    return r.get(result_key(task_id))


def get_meta_json(r: redis.Redis, task_id: str) -> str | None:
    return r.get(meta_key(task_id))


def poll_task_until_done(
    redis_url: str,
    task_id: str,
    timeout_sec: float = 180.0,
    interval: float = 0.4,
) -> dict[str, Any]:
    r = redis_connect(redis_url)
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        raw = get_result_json(r, task_id)
        if raw:
            return TaskResponse.from_json(raw).to_dict()
        mj = get_meta_json(r, task_id)
        if mj:
            td = TaskResponse.from_json(mj).to_dict()
            if td.get("status") == TaskStatus.ERROR.value:
                return td
        time.sleep(interval)
    return {"task_id": task_id, "status": "timeout", "error": "poll timeout"}


def xpending_summary(r: redis.Redis) -> dict[str, Any]:
    """PEL size + consumer breakdown (fault observability)."""
    try:
        p = r.xpending(STREAM_KEY, CONSUMER_GROUP)
    except Exception as e:
        return {"error": str(e)}
    if not p:
        return {}
    pending_count = p["pending"]
    lowest = p.get("min") or ""
    consumers = []
    try:
        for row in r.xpending_range(STREAM_KEY, CONSUMER_GROUP, "-", "+", count=50):
            consumers.append(dict(row))
    except Exception:
        pass
    return {"pending_total": pending_count, "pel_sample": consumers[:10], "lowest_id": lowest}


# --- RAG (FAISS) per process -----------------------------------------
def _chunk_text(text: str, max_chars: int = 400) -> list[str]:
    text = re.sub(r"\s+", " ", text.strip())
    if not text:
        return []
    return [text[i : i + max_chars] for i in range(0, len(text), max_chars)]


class FaissRetriever:
    """Each Ray actor holds its own instance (separate process = separate RAG state)."""

    def __init__(self, index_dir: str | Path | None = None) -> None:
        self.index_dir = Path(index_dir or os.environ.get("RAG_INDEX_DIR", "/tmp/rag_index_sim"))
        self.embed_model = os.environ.get("RAG_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        self._model = None
        self._index: faiss.Index | None = None
        self._chunks: list[str] = []

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.embed_model)
        return self._model

    def ensure_loaded(self, sample_corpus: str | None = None) -> None:
        index_path = self.index_dir / "index.faiss"
        meta_path = self.index_dir / "meta.json"
        self.index_dir.mkdir(parents=True, exist_ok=True)

        if index_path.exists() and meta_path.exists():
            self._index = faiss.read_index(str(index_path))
            self._chunks = json.loads(meta_path.read_text(encoding="utf-8"))["chunks"]
            return

        text = sample_corpus or (
            "Redis Streams consumer groups track a Pending Entry List PEL for each message. "
            "XAUTOCLAIM reassigns idle pending messages to another consumer. "
            "Ray actors can simulate multiple GPU workers on one machine. "
            "Ollama serves Llama models over HTTP for Colab and edge inference."
        )
        chunks = _chunk_text(text)
        model = self._get_model()
        emb = model.encode(chunks, convert_to_numpy=True, show_progress_bar=False)
        dim = emb.shape[1]
        index = faiss.IndexFlatIP(dim)
        faiss.normalize_L2(emb)
        index.add(emb.astype(np.float32))
        faiss.write_index(index, str(index_path))
        meta_path.write_text(json.dumps({"chunks": chunks}, ensure_ascii=False), encoding="utf-8")
        self._index = index
        self._chunks = chunks

    def retrieve_context(self, query: str, k: int = 4) -> list[str]:
        self.ensure_loaded()
        if self._index is None or not self._chunks:
            return []
        model = self._get_model()
        q = model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(q)
        k = min(k, len(self._chunks))
        _, indices = self._index.search(q.astype(np.float32), k)
        out: list[str] = []
        for idx in indices[0]:
            i = int(idx)
            if 0 <= i < len(self._chunks):
                out.append(self._chunks[i])
        return out


# --- Ollama HTTP (LLM per logical node) --------------------------------
def ollama_rag_chat(
    query: str,
    contexts: list[str],
    base_url: str | None = None,
    model: str | None = None,
    max_tokens: int = 256,
    temperature: float = 0.7,
) -> str:
    import requests

    base = (base_url or os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434")).rstrip("/")
    model = model or os.environ.get("OLLAMA_MODEL", "llama3.2:1b")
    ctx = "\n".join(f"- {c}" for c in contexts) if contexts else "(no context)"
    system = (
        "You are a helpful assistant. Use the context when relevant.\n\nContext:\n" + ctx
    )
    timeout = float(os.environ.get("OLLAMA_TIMEOUT_SEC", "600"))
    r = requests.post(
        f"{base}/api/chat",
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": query},
            ],
            "stream": False,
            "options": {"num_predict": max_tokens, "temperature": temperature},
        },
        timeout=timeout,
    )
    r.raise_for_status()
    data = r.json()
    msg = data.get("message") or {}
    content = msg.get("content") or ""
    return str(content).strip()


def _claim_min_idle_ms() -> int:
    return int(os.environ.get("REDIS_CLAIM_IDLE_MS", "60000"))


def _block_ms() -> int:
    return int(os.environ.get("REDIS_BLOCK_MS", "4000"))


def _rag_top_k() -> int:
    return int(os.environ.get("RAG_TOP_K", "4"))


def _process_one_message(
    r: redis.Redis,
    fields: dict,
    retriever: FaissRetriever,
    ollama_base: str,
    ollama_model: str,
) -> None:
    raw = fields.get("payload")
    if not raw:
        raise ValueError("missing payload")
    req = TaskRequest.from_json(raw)
    tid = req.task_id
    if not tid:
        raise ValueError("missing task_id")

    set_processing(r, tid)
    contexts = retriever.retrieve_context(req.query, k=_rag_top_k())
    ans = ollama_rag_chat(req.query, contexts, base_url=ollama_base, model=ollama_model)
    store_result(
        r,
        TaskResponse(
            task_id=tid,
            answer=ans,
            status=TaskStatus.DONE,
            context_snippets=contexts,
        ),
    )
    logger.info("[%s] done %s", os.environ.get("NODE_LABEL", "?"), tid)


def _handle_err(r: redis.Redis, raw_payload: str | None, exc: Exception) -> None:
    tid = None
    try:
        if raw_payload:
            tid = TaskRequest.from_json(raw_payload).task_id
    except Exception:
        pass
    if not tid:
        return
    store_result(r, TaskResponse(task_id=tid, answer="", status=TaskStatus.ERROR, error=str(exc)))


def reclaim_and_process_batch(
    r: redis.Redis,
    consumer_name: str,
    retriever: FaissRetriever,
    ollama_base: str,
    ollama_model: str,
) -> None:
    """PEL reclaim: stale pending → this consumer (**XAUTOCLAIM**)."""
    try:
        claim = r.xautoclaim(
            STREAM_KEY,
            CONSUMER_GROUP,
            consumer_name,
            _claim_min_idle_ms(),
            "0-0",
            count=20,
        )
    except redis.ResponseError as e:
        if "NOGROUP" in str(e):
            ensure_consumer_group(r)
        return
    if not claim or len(claim) < 2:
        return
    for item in claim[1] or []:
        if not item or len(item) < 2:
            continue
        mid, flds = item[0], item[1]
        raw = flds.get("payload") if isinstance(flds, dict) else None
        try:
            _process_one_message(r, flds, retriever, ollama_base, ollama_model)
            r.xack(STREAM_KEY, CONSUMER_GROUP, mid)
        except Exception as e:
            logger.exception("reclaim failed %s", mid)
            _handle_err(r, raw, e)
            try:
                r.xack(STREAM_KEY, CONSUMER_GROUP, mid)
            except Exception:
                pass


def poll_new_and_process(
    r: redis.Redis,
    consumer_name: str,
    retriever: FaissRetriever,
    ollama_base: str,
    ollama_model: str,
) -> None:
    streams = r.xreadgroup(
        CONSUMER_GROUP, consumer_name, {STREAM_KEY: ">"}, count=1, block=_block_ms()
    )
    if not streams:
        return
    for _, messages in streams:
        for mid, flds in messages:
            raw = flds.get("payload") if isinstance(flds, dict) else None
            try:
                _process_one_message(r, flds, retriever, ollama_base, ollama_model)
                r.xack(STREAM_KEY, CONSUMER_GROUP, mid)
            except Exception as e:
                logger.exception("task failed %s", mid)
                _handle_err(r, raw, e)
                try:
                    r.xack(STREAM_KEY, CONSUMER_GROUP, mid)
                except Exception:
                    pass


def thread_consumer_forever(
    redis_url: str,
    consumer_name: str,
    node_label: str,
    rag_index_dir: str,
    stop_event: threading.Event,
    ollama_base: str,
    ollama_model: str,
) -> None:
    """Runs inside a daemon thread inside each Ray InferenceNode."""
    os.environ["NODE_LABEL"] = node_label
    r = redis_connect(redis_url)
    ensure_consumer_group(r)
    retr = FaissRetriever(Path(rag_index_dir))
    retr.ensure_loaded()
    logger.info("node=%s consumer=%s rag_dir=%s", node_label, consumer_name, rag_index_dir)

    while not stop_event.is_set():
        try:
            reclaim_and_process_batch(r, consumer_name, retr, ollama_base, ollama_model)
            poll_new_and_process(r, consumer_name, retr, ollama_base, ollama_model)
        except redis.ConnectionError:
            time.sleep(2)
        except Exception:
            logger.exception("loop backoff")
            time.sleep(1)


