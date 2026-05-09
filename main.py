import uvicorn
import os

if __name__ == "__main__":
    # Assign specific port 8000 for the FastAPI Controller
    uvicorn.run(
        "master.scheduler:app", 
        host="0.0.0.0", 
        port=8000, 
        reload=True
    )
