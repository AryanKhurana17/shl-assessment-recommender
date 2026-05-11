"""
FastAPI service for the SHL Assessment Recommendation Agent.

Endpoints (from Assignment PDF, "API specification"):
- GET  /health → {"status": "ok"} with HTTP 200
- POST /chat   → stateless conversation, returns reply + recommendations

"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.schemas import ChatRequest, ChatResponse, Recommendation
from app.agent import initialize, run_agent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize resources at startup, cleanup at shutdown."""
    logger.info("Initializing SHL Assessment Agent...")
    initialize()
    logger.info("Agent ready.")
    yield
    logger.info("Shutting down.")


app = FastAPI(
    title="SHL Assessment Recommendation Agent",
    description="Conversational agent for SHL assessment recommendations",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — allow evaluator to call from any origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Stateless chat endpoint."""
    try:
        # Convert Pydantic models to dicts for the agent
        messages = [{"role": m.role, "content": m.content} for m in request.messages]

        # Run the LangGraph agent
        result = run_agent(messages)

        # Validate and construct response
        recommendations = []
        for rec in result.get("recommendations", []):
            try:
                recommendations.append(Recommendation(**rec))
            except Exception as e:
                # Anti-hallucination Layer 2: Drop invalid recommendations
                logger.warning(f"Dropping invalid recommendation: {rec} — {e}")
                continue

        response = ChatResponse(
            reply=result.get("reply", "I can help you find SHL assessments."),
            recommendations=recommendations,
            end_of_conversation=result.get("end_of_conversation", False),
        )

        return response

    except Exception as e:
        logger.error(f"Chat error: {e}", exc_info=True)
        # CRITICAL: Always return valid schema, even on error.
        # Source: Assignment PDF — "Schema compliance on every response" is a hard eval.
        return ChatResponse(
            reply=f"I encountered an issue. Could you rephrase your request about SHL assessments? Debug: {str(e)[:200]}",
            recommendations=[],
            end_of_conversation=False,
        )


# ---- Test code (run with: uvicorn app.main:app --reload) ----
