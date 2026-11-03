"""Minimal FastAPI wrapper exposing /chat and /health for local and Cloud Run deployment."""

from __future__ import annotations

from functools import lru_cache
from typing import Optional

from fastapi import Depends, FastAPI
from pydantic import BaseModel, Field

from app.agent import AgentTurnResponse, SDPAgentWorkflow
from config.settings import get_settings


class ChatRequest(BaseModel):
    """Incoming request schema for POST /chat."""

    prompt: str = Field(..., min_length=1, description="User prompt for the ADK agent.")
    user_id: str = Field(default="enterprise-user", description="Caller user identifier.")
    session_id: Optional[str] = Field(
        default=None,
        description="Optional session ID for multi-turn conversation continuity.",
    )


class HealthResponse(BaseModel):
    """Health & runtime configuration summary."""

    status: str = "ok"
    project_id: str
    sdp_location: str
    vertex_location: str
    model: str
    inspect_template_path: str
    deidentify_template_path: str
    prompt_filter_template: str
    response_filter_template: str


@lru_cache(maxsize=1)
def get_workflow() -> SDPAgentWorkflow:
    """Singleton ADK workflow provider (overridable in FastAPI tests)."""
    return SDPAgentWorkflow()


def create_app(workflow_override: Optional[SDPAgentWorkflow] = None) -> FastAPI:
    """Create and configure the FastAPI application."""
    fastapi_app = FastAPI(
        title="ADK + Cloud SDP Resilient Redaction Middleware",
        version="0.1.0",
        description=(
            "High-code Google ADK agent applying Model Armor prompt screening and "
            "post-inference Cloud SDP in-place redaction."
        ),
    )

    if workflow_override is not None:
        fastapi_app.dependency_overrides[get_workflow] = lambda: workflow_override

    @fastapi_app.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        settings = (
            workflow_override.settings if workflow_override else get_settings()
        )
        return HealthResponse(
            project_id=settings.project_id,
            sdp_location=settings.location,
            vertex_location=settings.vertex_location,
            model=settings.gemini_model,
            inspect_template_path=settings.inspect_template_path,
            deidentify_template_path=settings.deidentify_template_path,
            prompt_filter_template=settings.prompt_filter_template_id,
            response_filter_template=settings.response_filter_template_id,
        )

    @fastapi_app.post("/chat", response_model=AgentTurnResponse)
    async def chat(
        payload: ChatRequest,
        workflow: SDPAgentWorkflow = Depends(get_workflow),
    ) -> AgentTurnResponse:
        return await workflow.chat(
            prompt=payload.prompt,
            user_id=payload.user_id,
            session_id=payload.session_id,
        )

    return fastapi_app


app = create_app()
