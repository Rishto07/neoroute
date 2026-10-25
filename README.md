# GhostRoute — self-healing AI gateway MVP

A small, auditable starting point for an OpenAI-compatible gateway. It forwards `/v1/chat/completions` requests to configured OpenAI-compatible upstreams and falls back when an upstream is unavailable, rate-limited, or returns a server error.

## Quick start (Windows PowerShell)

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
Copy-Item .env.example .env
```

Edit `.env` to configure one or more providers. Example:

```dotenv
GHOSTROUTE_PROVIDERS=[{"name":"primary","base_url":"https://api.openai.com/v1","api_key_env":"OPENAI_API_KEY","model":"gpt-4o-mini"},{"name":"fallback","base_url":"https://api.groq.com/openai/v1","api_key_env":"GROQ_API_KEY","model":"llama-3.3-70b-versatile"}]
OPENAI_API_KEY=replace-me
GROQ_API_KEY=replace-me
```

Never commit real API keys. Provider names and model names are examples; verify availability and terms with each provider.

Run the server:

```powershell
uvicorn ghostroute.main:app --reload
```

Health check: `http://127.0.0.1:8000/health`
Interactive API docs: `http://127.0.0.1:8000/docs`

Example request in PowerShell:

```powershell
$body = @{ model = "auto"; messages = @(@{ role = "user"; content = "Say hello in one sentence." }) } | ConvertTo-Json -Depth 8
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/v1/chat/completions -ContentType "application/json" -Body $body
```

Use `model: "auto"` for configured order or a configured provider name to prefer that provider. Response headers include `X-GhostRoute-Provider`, `X-GhostRoute-Attempts`, and `X-GhostRoute-Latency-Ms`.

## Tests

```powershell
pytest -q
```

## Deliberate MVP limitations

- Streaming is explicitly rejected (`stream=true` returns 501); it is better to state this than silently corrupt SSE/tool-call streams.
- No dashboard, automatic quality scoring, quota discovery, or broad protocol translation yet.
- This MVP is for local development; add authentication, rate limiting, hardened SSRF controls, secret management, and operational monitoring before exposing it to a network.

## Next milestones

1. Add structured routing decisions and safe error classification.
2. Add a provider capability registry.
3. Persist traces, latency, cost, and outcomes.
4. Implement proper streaming passthrough and explicit retry boundaries.
5. Add task-aware routing and a reproducible benchmark suite.
