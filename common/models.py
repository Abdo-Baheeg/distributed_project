from dataclasses import dataclass, asdict
from enum import Enum
from typing import Optional
import json

class TaskStatus(str, Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    DONE = "done"
    ERROR = "error"

@dataclass
class TaskRequest:
    id: int
    query: str

@dataclass
class TaskResponse:
    id: int
    result: Optional[str] = None
    status: TaskStatus = TaskStatus.QUEUED
    latency: float = 0.0
    error: Optional[str] = None

def to_json(obj):
    return json.dumps(asdict(obj))
