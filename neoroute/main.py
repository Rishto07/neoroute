import json
import subprocess
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from neoroute.anthropic import (
    AnthropicRequestError,
    anthropic_response_body,
    error_response,
    gateway_error_message,
    to_chat_request,
)
from neoroute.config import get_settings
from neoroute.gateway import attach_trace_headers, forward_chat
from neoroute.schemas import ChatCompletionRequest


@asynccontextmanager
async def lifespan(_: FastAPI):
    get_settings()  # fail early on missing/invalid configuration
    yield


app = FastAPI(title="neoroute", description="A self-healing, OpenAI-compatible AI gateway MVP", version="0.1.0", lifespan=lifespan)

_REPO_ROOT = Path(__file__).resolve().parents[1]
_FRONTEND_DIR = _REPO_ROOT / "frontend"
_REPORT_PATH = _REPO_ROOT / "reports" / "response_eval.json"
_EVAL_SCRIPT = _REPO_ROOT / "tests" / "corpus_eval.py"


@app.get("/")
async def root():
    return RedirectResponse(url="/ui/")


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


@app.post("/v1/messages")
async def anthropic_messages(request: Request):
    """Anthropic Messages API surface (non-streaming translation).

    Translates to the OpenAI-compatible request, forwards through the same
    gateway (fallback, attempts, response contract), and returns the Anthropic
    response shape. ``stream: true`` is rejected because there is no SSE
    passthrough; real Claude Code requires SSE and is not yet supported.
    """
    try:
        raw = await request.json()
    except (ValueError, UnicodeDecodeError):
        return error_response("Request body must be valid JSON.", 400)
    if not isinstance(raw, dict):
        return error_response("Request body must be a JSON object.", 400)

    try:
        chat_payload = to_chat_request(raw)
    except AnthropicRequestError as exc:
        return error_response(exc.message, exc.status, exc.type)

    try:
        settings = get_settings()
    except ValueError as exc:
        return error_response(str(exc), 503)

    try:
        result, elapsed = await forward_chat(chat_payload, settings)
    except HTTPException as exc:
        return error_response(gateway_error_message(exc.detail), exc.status_code)

    try:
        body = anthropic_response_body(result.body, chat_payload.model)
    except ValueError as exc:
        return error_response(str(exc), 502)

    response = JSONResponse(content=body)
    attach_trace_headers(response, result, elapsed)
    return response


@app.get("/api/eval/report")
def get_eval_report():
    if not _REPORT_PATH.exists():
        raise HTTPException(status_code=404, detail="No saved evaluation report found. Run the evaluation first.")
    try:
        return json.loads(_REPORT_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=500, detail=f"Saved evaluation report could not be read: {exc}") from exc


@app.post("/api/eval/run")
def run_eval():
    _REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    try:
        completed = subprocess.run(
            [sys.executable, str(_EVAL_SCRIPT), "--json-out", str(_REPORT_PATH)],
            cwd=str(_REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise HTTPException(status_code=504, detail="Evaluation timed out.") from exc
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip() or "Evaluation failed."
        raise HTTPException(status_code=500, detail=detail)
    try:
        return json.loads(_REPORT_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=500, detail=f"Evaluation completed but its report could not be read: {exc}") from exc


if _FRONTEND_DIR.exists():
    app.mount("/ui", StaticFiles(directory=str(_FRONTEND_DIR), html=True))

