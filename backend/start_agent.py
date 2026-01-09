import os
import subprocess
import threading
import time
from fastapi import FastAPI
import uvicorn
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

@app.get("/")
def root():
    return {"status": "agent_running", "service": "livekit-agent"}

@app.get("/health")
def health():
    return {"status": "healthy"}

def start_agent():
    """Start the LiveKit agent in a subprocess"""
    try:
        # Change to backend directory and run the agent
        os.chdir(os.path.dirname(os.path.abspath(__file__)))
        subprocess.run(["python", "main.py", "start"])
    except Exception as e:
        print(f"Error starting agent: {e}")

if __name__ == "__main__":
    # Start agent in background thread
    agent_thread = threading.Thread(target=start_agent, daemon=True)
    agent_thread.start()
    
    # Give agent a moment to start
    time.sleep(2)
    
    # Start HTTP server to keep Render happy (it needs an open port)
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
