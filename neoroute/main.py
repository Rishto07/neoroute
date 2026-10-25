from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Response
from neoroute.config import get_settings
from neoroute.gateway import attach_trace_headers, forward_chat
from neoroute.schemas import ChatCompletionRequest


@asynccontextmanager
async def lifespan(_: FastAPI):
    get_settings()  # fail early on missing/invalid configuration
    yield


app = FastAPI(title="neoroute", description="A self-healing, OpenAI-compatible AI gateway MVP", version="0.1.0", lifespan=lifespan)


@app.get("/health")
async def health():
    try:
        settings = get_settings()
        return {"status": "ok", "configured_providers": [p.name for p in settings.providers], "streaming": False}
    except ValueError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/v1/chat/completions")
async def chat_completions(payload: ChatCompletionRequest):
    try:
        settings = get_settings()
    except ValueError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    result, elapsed = await forward_chat(payload, settings)
    response = Response(content=result.body, status_code=result.status_code, media_type=result.content_type.split(";")[0])
    attach_trace_headers(response, result, elapsed)
    return response

