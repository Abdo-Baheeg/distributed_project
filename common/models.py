from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Mapping


class TaskStatus(str, Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    DONE = "done"
    ERROR = "error"


@dataclass(frozen=True)
class TaskRequest:
    """Inbound work unit (also exposed as **`Request`** for skeleton naming)."""

    query: str
    task_id: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["metadata"] = dict(self.metadata)
        return d

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> TaskRequest:
        meta = data.get("metadata") or {}
        return cls(
            query=data["query"],
            task_id=data.get("task_id"),
            metadata=dict(meta) if isinstance(meta, Mapping) else {},
        )

    @classmethod
    def from_json(cls, raw: str | bytes) -> TaskRequest:
        return cls.from_dict(json.loads(raw))


@dataclass(frozen=True)
class TaskResponse:
    """Pollable outcome (also exposed as **`Response`** for skeleton naming)."""

    task_id: str
    answer: str
    status: TaskStatus
    context_snippets: list[str] = field(default_factory=list)
    error: str | None = None

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "answer": self.answer,
            "status": self.status.value if isinstance(self.status, TaskStatus) else str(self.status),
            "context_snippets": list(self.context_snippets),
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> TaskResponse:
        status = data.get("status", TaskStatus.DONE.value)
        if isinstance(status, str):
            status = TaskStatus(status)
        return cls(
            task_id=data["task_id"],
            answer=data.get("answer", ""),
            status=status,
            context_snippets=list(data.get("context_snippets") or []),
            error=data.get("error"),
        )

    @classmethod
    def from_json(cls, raw: str | bytes) -> TaskResponse:
        return cls.from_dict(json.loads(raw))


def result_key(task_id: str) -> str:
    return f"result:{task_id}"


def meta_key(task_id: str) -> str:
    return f"task:meta:{task_id}"


# Skeleton aliases
Request = TaskRequest
Response = TaskResponse


__all__ = [
    "TaskStatus",
    "TaskRequest",
    "TaskResponse",
    "Request",
    "Response",
    "result_key",
    "meta_key",
]
