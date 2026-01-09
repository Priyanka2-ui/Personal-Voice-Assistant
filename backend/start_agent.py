import os
import subprocess
import threading
import time
import sys
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

# Global variable to store the agent process
agent_process = None

def start_agent():
    """Start the LiveKit agent in a subprocess"""
    global agent_process
    try:
        # Get the directory where this script is located
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Start the agent process (non-blocking)
        agent_process = subprocess.Popen(
            [sys.executable, "main.py", "start"],
            cwd=script_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=1,
            universal_newlines=True
        )
        
        # Log output in real-time
        def log_output(pipe, prefix):
            for line in iter(pipe.readline, ''):
                if line:
                    print(f"[{prefix}] {line.strip()}")
            pipe.close()
        
        # Start threads to log stdout and stderr
        stdout_thread = threading.Thread(
            target=log_output, 
            args=(agent_process.stdout, "AGENT"),
            daemon=True
        )
        stderr_thread = threading.Thread(
            target=log_output, 
            args=(agent_process.stderr, "AGENT-ERROR"),
            daemon=True
        )
        stdout_thread.start()
        stderr_thread.start()
        
        print(f"Agent process started with PID: {agent_process.pid}")
        
    except Exception as e:
        print(f"Error starting agent: {e}")
        import traceback
        traceback.print_exc()

def cleanup():
    """Clean up the agent process on exit"""
    global agent_process
    if agent_process:
        print("Stopping agent process...")
        agent_process.terminate()
        try:
            agent_process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            agent_process.kill()
        print("Agent process stopped")

# Register cleanup on exit
import atexit
atexit.register(cleanup)

if __name__ == "__main__":
    # Start agent in background thread
    agent_thread = threading.Thread(target=start_agent, daemon=False)
    agent_thread.start()
    
    # Give agent more time to start (LiveKit agents need time to initialize)
    time.sleep(5)
    
    # Start HTTP server to keep Render happy (it needs an open port)
    port = int(os.getenv("PORT", 8000))
    
    try:
        uvicorn.run(app, host="0.0.0.0", port=port)
    except KeyboardInterrupt:
        cleanup()
        sys.exit(0)
