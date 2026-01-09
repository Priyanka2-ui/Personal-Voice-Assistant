from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uuid
import logging
import asyncio
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Try importing from backend.main first (for when running from root), 
# otherwise import from main (for when running from backend directory)
try:
    from backend.main import Assistant, AgentSession, RoomInputOptions, RoomOutputOptions, noise_cancellation, openai, TurnDetection, metrics
except ImportError:
    from main import Assistant, AgentSession, RoomInputOptions, RoomOutputOptions, noise_cancellation, openai, TurnDetection, metrics

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# In-memory session store
sessions = {}

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str

class VoiceAgentSession:
    def __init__(self):
        self.session = AgentSession(
            llm=openai.realtime.RealtimeModel.with_azure(
                azure_deployment="gpt-4o-realtime-preview",
                azure_endpoint=os.getenv("AZURE_OPENAI_REALTIME_ENDPOINT"),
                api_key=os.getenv("AZURE_OPENAI_REALTIME_API_KEY"),
                api_version=os.getenv("AZURE_OPENAI_REALTIME_API_VERSION"),
                turn_detection=TurnDetection(
                    type="server_vad",
                    threshold=0.5,
                    prefix_padding_ms=300,
                    silence_duration_ms=500,
                    create_response=True,
                    interrupt_response=True,
                )
            ),
        )
        self.assistant = Assistant()
        self.room = type("DummyRoom", (), {"name": f"api-room-{uuid.uuid4()}"})()
        self.usage_collector = metrics.UsageCollector()
        self.session.on("metrics_collected")(self._on_metrics_collected)
        self.session_started = False

    def _on_metrics_collected(self, ev):
        metrics.log_metrics(ev.metrics)
        self.usage_collector.collect(ev.metrics)

    async def start(self):
        if not self.session_started:
            await self.session.start(
                agent=self.assistant,
                room=self.room,
                room_input_options=RoomInputOptions(
                    noise_cancellation=noise_cancellation.BVC(),
                ),
                room_output_options=RoomOutputOptions(transcription_enabled=True),
            )
            self.session_started = True

    async def send_message(self, message: str) -> str:
        await self.start()
        handle = self.session.generate_reply(user_input=message)
        await handle
        assistant_messages = [item for item in handle.chat_items if getattr(item, 'type', None) == 'message' and getattr(item, 'role', None) == 'assistant']
        if assistant_messages:
            return assistant_messages[-1].text_content or ""
        return "[No response]"

    def close(self):
        pass

@app.post("/session/start")
async def start_session():
    try:
        session_id = str(uuid.uuid4())
        sessions[session_id] = VoiceAgentSession()
        logger.info(f"Started new session: {session_id}")
        return {"session_id": session_id}
    except Exception as e:
        logger.error(f"Error starting session: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to start session: {str(e)}")

@app.post("/session/{session_id}/chat", response_model=ChatResponse)
async def chat(session_id: str, request: ChatRequest):
    try:
        session = sessions.get(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        logger.info(f"Processing chat message for session: {session_id}")
        response = await session.send_message(request.message)
        return ChatResponse(response=response)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing chat message: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to process message: {str(e)}")

@app.post("/session/{session_id}/end")
async def end_session(session_id: str):
    try:
        session = sessions.pop(session_id, None)
        if session:
            session.close()
            logger.info(f"Ended session: {session_id}")
        return {"status": "ended"}
    except Exception as e:
        logger.error(f"Error ending session: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to end session: {str(e)}")

@app.get("/")
async def root():
    return {"message": "Voice Assistant API is running", "status": "healthy"}

@app.get("/health")
async def health():
    return {"status": "healthy", "service": "voice-assistant-api"} 