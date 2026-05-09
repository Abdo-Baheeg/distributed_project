import uvicorn
import os

if __name__ == "__main__":
    # Assigning Port 8000 as the entry point for the Master Node [cite: 301-303]
    port = int(os.getenv("PORT", 8000))
    print(f"Starting Master Controller on port {port}...")
    
    uvicorn.run(
        "master.scheduler:app", 
        host="0.0.0.0", 
        port=port, 
        reload=True
    )