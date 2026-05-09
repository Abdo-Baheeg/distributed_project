from dataclasses import dataclass, asdict
import json

@dataclass
class Request:
    id: int
    query: str

@dataclass
class Response:
    id: int
    result: str
    latency: float

# Helper to serialize dataclasses for Redis
def to_json(obj):
    return json.dumps(asdict(obj))