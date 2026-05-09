import os
import json
import asyncio
import redis.asyncio as redis # Using async redis for high concurrency
from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from common.models import TaskRequest, TaskResponse, TaskStatus, to_json
from dotenv import load_dotenv

load_dotenv()

# --- FAULT TOLERANCE MONITOR ---
async def fault_tolerance_monitor(redis_client):
    """
    Background loop to detect failed nodes and reassign tasks.
    """
    while True:
        try:
            # XAUTOCLAIM: Reclaim tasks idle for > 2 minutes (120000ms)
            # This ensures 'no request is lost' if a Colab worker fails.
            await redis_client.xautoclaim(
                name="task_stream",
                groupname="workers",
                consumername="Monitor_Node",
                min_idle_time=120000,
                start_id="0-0"
            )
        except Exception as e:
            print(f"[Monitor Error] {e}")
        await asyncio.sleep(60)


# --- LIFESPAN MANAGER ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.redis = redis.from_url(os.getenv("REDIS_URL", "redis://localhost:6379/0"))
    try:
        # Create Consumer Group 'workers' to distribute tasks
        await app.state.redis.xgroup_create("task_stream", "workers", id="0", mkstream=True)
    except redis.exceptions.ResponseError:
        pass # Group already exists
    
    monitor_task = asyncio.create_task(fault_tolerance_monitor(app.state.redis))
    yield
    # Shutdown
    monitor_task.cancel()
    await app.state.redis.close()

app = FastAPI(title="CSE354 Distributed Controller", lifespan=lifespan)

# --- ROUTES ---

@app.post("/ask", response_model=dict)
async def submit_task(req: TaskRequest):
    """
    Receives user query and dispatches to the GPU Cluster queue.
    """
    try:
        # 1. Initialize metadata in Redis
        task_meta = TaskResponse(id=req.id, status=TaskStatus.QUEUED)
        await app.state.redis.set(f"task:meta:{req.id}", to_json(task_meta))
        
        # 2. Add to Redis Stream (The Message Broker)
        task_payload = {"id": req.id, "query": req.query}
        await app.state.redis.xadd("task_stream", task_payload)
        
        return {"status": "accepted", "task_id": req.id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Broker Error: {str(e)}")

@app.get("/task/{task_id}", response_model=TaskResponse)
async def get_task_result(task_id: int):
    """
    Polls for the result of a specific AI request.
    """
    # 1. Check for finished result
    result_data = await app.state.redis.get(f"result:{task_id}")
    if result_data:
        return TaskResponse(**json.loads(result_data))
    
    # 2. If not finished, check current metadata status
    meta_data = await app.state.redis.get(f"task:meta:{task_id}")
    if meta_data:
        return TaskResponse(**json.loads(meta_data))
    
    raise HTTPException(status_code=404, detail="Task not found")

@app.get("/health")
async def health():
    """System monitoring endpoint."""
    is_alive = await app.state.redis.ping()
    return {"status": "active", "redis_connected": is_alive}
