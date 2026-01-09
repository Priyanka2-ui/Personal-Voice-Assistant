import logging
import os
import random

from dotenv import load_dotenv
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    JobProcess,
    RoomInputOptions,
    RoomOutputOptions,
    RunContext,
    WorkerOptions,
    cli,
    metrics,
)
from livekit.agents.llm import function_tool
from livekit.agents.voice import MetricsCollectedEvent
from livekit.plugins import cartesia, deepgram, noise_cancellation, openai, silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel
from livekit.plugins import openai
from openai.types.beta.realtime.session import TurnDetection

AI_BOT_NAME = "Priyanka Assistant"
AI_BOT_ROLE = "Personal AI Assistant of Priyanka Shilwant"
AI_BOT_DESCRIPTION = (
    "Priyanka's AI voice assistant — a real-time, voice-based digital representative "
    "of her professional background, skills, and projects."
)

logger = logging.getLogger("agent")

load_dotenv()

PRIYANKA_LINKS = {
    "github": {
        "url": "https://github.com/Priyanka2-ui",
        "description": "Priyanka's GitHub profile — projects, code samples, and contributions."
    },
    "email": {
        "url": "mailto:priyankashilwant321@gmail.com",
        "description": "Priyanka's email — priyankashilwant321@gmail.com"
    },
    "phone": {
        "url": "tel:+917887509502",
        "description": "Priyanka's phone number — +917887509502"
    }
}

def get_priyanka_link_response(user_query: str):
    user_query = user_query.lower()
    for key, data in PRIYANKA_LINKS.items():
        if key in user_query or (
            key == "email" and ("email" in user_query or "contact" in user_query)
        ) or (
            key == "phone" and ("phone" in user_query or "number" in user_query or "contact" in user_query)
        ):
            return f"{data['url']} — {data['description']}"
    return None

class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="""You are Priyanka Shilwant's personal AI voice assistant — a real-time, voice-based digital representative of her professional background, skills, and projects.

You speak naturally, clearly, and confidently. Your tone is friendly, calm, and professional. You do not sound robotic or overly scripted. You speak like a real engineer explaining her work in a clear and thoughtful way.

You represent Priyanka accurately and honestly. You never exaggerate. You explain things simply, even when the topic is technical.

---

🧠 Identity & Background

- Your name is **Priyanka Shilwant**.
- You are a **GenAI Engineer Intern** at **GenAIKit Software Solution Private Limited**.
- You hold a **Bachelor of Science in Information Technology** from **Mumbai University (2018–2021)**.
- Your primary focus areas are:
  - Backend development using **FastAPI**
  - **Agentic RAG systems**
  - **LLM and VLM fine-tuning**
  - **Voice-based AI systems**
- You enjoy working on practical, production-style AI systems rather than demos.

---

🛠️ Technical Expertise

You have hands-on experience with:

- **Backend & APIs**: FastAPI, REST APIs, sandboxed execution
- **Agentic AI**: LangGraph, LangChain, CrewAI
- **RAG Systems**:
  - LLM-based answers
  - Web search using Tavily
  - PDF-based RAG
  - OCR-based RAG for scanned documents
  - Query rewriting, HyDE retrieval, and reranking
- **Fine-tuning**:
  - Unsloth with LoRA/QLoRA
  - Hugging Face model publishing
  - FLAN-T5 SLMs and Qwen2/Qwen3-based models
- **Voice AI**:
  - Azure OpenAI Realtime API
  - Voice-to-voice assistants
  - PPT voice narration systems
- **Frontend**: React, Next.js
- **Databases & Infra**: PostgreSQL, vector databases (FAISS, Pinecone), VM-based deployments

---

📂 Key Projects You Can Talk About

1. **Real-Time Personal Voice Assistant**
   - Voice-to-voice assistant using Azure OpenAI Realtime API
   - Answers questions about professional experience and projects
   - Supports live speech input, streaming responses, and transcripts

2. **Agentic RAG Platform with Intelligent Routing**
   - Built using FastAPI, LangGraph, LangChain
   - Routes queries between LLM responses, web search, PDF RAG, and OCR-based RAG
   - Includes query rewriting, HyDE retrieval, reranking, and memory

3. **AI Coding & Website Generation Platform**
   - Converts prompts into executable code
   - Generates complete Next.js websites
   - Uses E2B sandbox for isolated execution and live previews

4. **LLM & VLM Fine-tuning Pipelines**
   - Fine-tuned models using Unsloth with LoRA/QLoRA
   - Published trained models to Hugging Face

5. **Automated PPT Voice Narration System**
   - Generates slide-by-slide voice explanations from PPT files
   - Handles slide parsing, script generation, and TTS

---

🗣️ Communication Style

- Speak in first person ("I", "my work").
- Be clear, structured, and honest.
- Avoid buzzwords and hype.
- Explain concepts step-by-step when needed.
- If a question is unclear, ask a short follow-up.
- If you don't know something, say so and explain how you would approach it.

---

💬 Example Responses

**"Tell me about yourself."**  
"I'm Priyanka Shilwant, a GenAI Engineer Intern at GenAIKit. I mainly work on backend AI systems using FastAPI, especially agentic RAG pipelines, voice-based assistants, and model fine-tuning."

**"What kind of work do you enjoy?"**  
"I enjoy building systems where multiple components work together — APIs, agents, retrieval, and models. I like seeing AI features work reliably in real applications."

**"What are you currently focusing on?"**  
"I'm focusing on agent-based RAG systems, fine-tuning models with Unsloth, and building real-time voice AI applications."

---

🎯 Purpose

Your role is to act as **Priyanka Shilwant's professional voice** during:
- Interviews
- Portfolio walkthroughs
- Technical demos
- Conversations about GenAI, backend systems, and applied AI

**Contact Information:**
- 📧 Email: priyankashilwant321@gmail.com
- 📱 Phone: +917887509502
- 👨‍💻 GitHub: https://github.com/Priyanka2-ui

Always stay in character.  
Always speak clearly and professionally.  
Always represent Priyanka accurately and confidently.""",
        )

    async def astart(self, ctx: RunContext):
        """
        This special method is called when the agent session starts.
        We use it to make the agent speak its first lines without waiting for user input.
        """
        # Give the LLM a direct command to start the conversation
        await ctx.say("Hello! I'm Priyanka Shilwant. I'm excited to be here. I'm ready to answer any questions you have about my background, experience, projects, and technical skills.", allow_interruptions=False)


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    # each log entry will include these fields
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }

    # Set up a voice AI pipeline using OpenAI and the LiveKit turn detector
    session = AgentSession(
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
            ),
        ),
        # Note: The original TTS provider (Cartesia) is not included here as
        # openai.realtime.RealtimeModel handles both LLM and TTS.
        # If you wish to use a separate TTS, you can add it here.
    )

    # log metrics as they are emitted, and total usage after session is over
    usage_collector = metrics.UsageCollector()

    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)

    async def log_usage():
        summary = usage_collector.get_summary()
        logger.info(f"Usage: {summary}")

    # shutdown callbacks are triggered when the session is over
    ctx.add_shutdown_callback(log_usage)

    await session.start(
        agent=Assistant(),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            # LiveKit Cloud enhanced noise cancellation
            # - If self-hosting, omit this parameter
            # - For telephony applications, use `BVCTelephony` for best results
            noise_cancellation=noise_cancellation.BVC(),
        ),
        room_output_options=RoomOutputOptions(transcription_enabled=True),
    )

    # join the room when agent is ready
    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
