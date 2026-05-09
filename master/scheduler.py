import os
import redis
from fastapi import FastAPI, HTTPException
from common.models import Request, Response
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="CSE354 Distributed AI Master")

# Initialize Redis client (Broker)
r = redis.Redis(
    host=os.getenv("REDIS_HOST", "localhost"),
    port=6379,
    password=os.getenv("REDIS_PASSWORD"),
    decode_responses=True
)

@app.post("/ask")
async def handle_request(req: Request):
    """
    Receives request and dispatches it to the Redis Message Broker [cite: 223-225].
    """
    try:
        # Pushing to Redis Stream 'task_stream' for cluster nodes to claim [cite: 68-71]
        task_data = {"id": req.id, "query": req.query}
        r.xadd("task_stream", task_data)
        
        print(f"[Scheduler] Dispatched request {req.id} to Redis")
        return {"status": "queued", "task_id": req.id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health_check():
    return {"status": "active", "redis_connected": r.ping()}