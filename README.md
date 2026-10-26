# Neoroute - self-healing AI gateway MVP

A small, auditable starting point for an OpenAI-compatible gateway. It forwards `/v1/chat/completions` requests to configured OpenAI-compatible upstreams and falls back when an upstream is unavailable, rate-limited, returns a server error, or returns a malformed successful response.

## Quick start (Windows PowerShell)

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
Copy-Item .env.example .env
```

Edit `.env` to configure one or more providers. Example:

```dotenv
NEOROUTE_PROVIDERS=[{"name":"primary","base_url":"https://api.openai.com/v1","api_key_env":"OPENAI_API_KEY","model":"gpt-4o-mini"},{"name":"fallback","base_url":"https://api.groq.com/openai/v1","api_key_env":"GROQ_API_KEY","model":"llama-3.3-70b-versatile"}]
OPENAI_API_KEY=replace-me
GROQ_API_KEY=replace-me
```

Never commit real API keys. `.env` is git-ignored and must never be committed, pushed, or pasted into issues, pull requests, or logs — only `.env.example` (placeholders such as `replace-me`) is tracked. Provider names and model names are examples; verify availability and terms with each provider.

Run the server:

```powershell
uvicorn neoroute.main:app --reload
```

Health check: `http://127.0.0.1:8000/health`
Interactive API docs: `http://127.0.0.1:8000/docs`

Example request in PowerShell:

```powershell
$body = @{ model = "auto"; messages = @(@{ role = "user"; content = "Say hello in one sentence." }) } | ConvertTo-Json -Depth 8
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/v1/chat/completions -ContentType "application/json" -Body $body
```

Use `model: "auto"` for configured order or a configured provider name to prefer that provider. Response headers include `X-Neoroute-Provider`, `X-Neoroute-Attempts`, and `X-Neoroute-Latency-Ms`.

## Capability routing

A provider may declare what its configured model supports. Capabilities are `vision`, `tools`, `json_mode`, `json_schema`, `audio_input`, and `audio_output`; each value is `supported`, `unsupported`, or `unknown`. Omitting a capability is the same as `unknown`. Declarations are operator-configured; Neoroute does not discover model capabilities automatically.

```json
{
  "name": "primary",
  "base_url": "https://api.openai.com/v1",
  "api_key_env": "OPENAI_API_KEY",
  "model": "gpt-4o-mini",
  "capabilities": {
    "vision": "supported",
    "tools": "supported",
    "json_mode": "supported",
    "json_schema": "supported",
    "audio_input": "unsupported",
    "audio_output": "unsupported"
  }
}
```

Neoroute derives requirements from image/audio content parts, non-empty `tools`, legacy `functions` and `function_call`, tool-role or tool-call messages, `response_format` (`json_object` versus `json_schema`), and `modalities` containing `audio`. `tool_choice` is validated but does not require `tools` unless tools are actually supplied. Providers are filtered before named-provider preference and attempt truncation.

Legacy `function_call` follows OpenAI semantics: `"none"` and `""` do not require `tools`, `"auto"` requires `tools` only when non-empty `functions` are supplied, and an assistant `function_call` object requires `tools`.

Malformed or unrecognized request shapes fail closed. Neoroute recognizes `text`/`input_text`, image, and audio content parts; `json_object`, `json_schema`, and `text` response formats; and list-shaped `tools`, `functions`, and `modalities`. An unknown content-part type, a media part with no `type`, an unknown response-format type, malformed `tools`/`functions`/`modalities`, an invalid `function_call`, or another unsupported shape is rejected with `422` and a fixed reason code before any upstream call. It is never routed as text-only.

Unknown metadata is fail-closed: if a request requires a capability and a provider does not explicitly declare it as `supported`, that provider is excluded. If no configured provider is eligible, Neoroute returns `422` with the required capability list before making any upstream call. Text-only requests have no capability requirements, so existing routing behavior is unchanged.

Configuration JSON must not contain duplicate object keys, including inside `capabilities`; duplicate keys are a configuration error rather than last-key-wins.

## Response verification

For non-streaming requests, every successful (`2xx`) upstream response is structurally validated before it is returned. The check requires a top-level JSON object, a non-empty `choices` array, a `message` object for each choice, a non-empty string `role`, and `content` as a string or `null`. Tool calls and legacy function calls are checked for their required structure when present. Optional metadata such as `usage`, `finish_reason`, `id`, and provider-specific fields is preserved and is not type-checked.

When the request uses `response_format: {"type": "json_object"}`, every choice's assistant content must be parseable JSON. JSON is not repaired or coerced. Tool-call and refusal outcomes with `null` content are accepted without a JSON-content check; non-null content is still validated. `json_schema` requests receive structural validation only; JSON Schema conformance is not enforced. Passing these checks establishes structural validity, not semantic correctness or output quality.

A malformed `2xx` response is recorded as a failed attempt with its original status and a fixed failure category (`malformed_response` or `invalid_json_content`), then routing continues to the next eligible provider within the existing plan and `NEOROUTE_MAX_ATTEMPTS` limit. If every eligible provider returns a malformed successful response, the gateway returns its generic `502` error. Diagnostics list provider names and fixed reason codes; they never include upstream response bodies, prompts, completions, or credentials.

When a response passes validation, Neoroute forwards the original upstream body bytes, HTTP status, and `Content-Type` unchanged. Streaming remains unsupported: `stream=true` returns `501`.

## Anthropic-compatible endpoint (Claude Code status)

`POST /v1/messages` accepts the Anthropic Messages API shape (system, messages, tools, tool_choice, stop_sequences), translates it to the internal OpenAI-compatible request, and forwards it through the same gateway - so fallback, bounded retries, attempt records, and response verification apply unchanged. Responses use the Anthropic message format with the `X-Neoroute-*` trace headers.

Honest status - what is verified versus what is not:

- Non-streaming `/v1/messages` requests work end-to-end and are covered by deterministic route tests (mocked providers; no live Anthropic credentials were used).
- `stream: true` is rejected with `400`. The gateway has no SSE passthrough, and **Claude Code requires SSE streaming, so real Claude Code is not yet supported end to end.**
- Image content blocks, extended-thinking parameters, and `POST /v1/messages/count_tokens` are not implemented; unsupported content is rejected explicitly rather than silently dropped.

## Tests

```powershell
python -m pytest -q
```

## Reliability evaluator

The evaluator replays the committed response corpus through the real verifier and gateway in-process. No provider is contacted. It checks structural response behavior and corpus coverage; it is not a guarantee of semantic output quality.

```powershell
python tests/corpus_eval.py
```

Exit codes: `0` (all cases pass; no regressions; baseline present or explicitly allowed), `1` (case failures, regressions, or strict baseline drift), `2` (configuration, I/O, or invalid-baseline error).

## Contributing

See `CONTRIBUTING.md` for setup, tests, lint, evaluator, and pull request workflow.
## Deliberate MVP limitations

- Streaming is explicitly rejected (`stream=true` returns 501); it is better to state this than silently corrupt SSE/tool-call streams.
- No dashboard, automatic quality scoring, quota discovery, or broad protocol translation yet.
- This MVP is for local development; add authentication, rate limiting, hardened SSRF controls, secret management, and operational monitoring before exposing it to a network.

## Next milestones

1. Persist traces, latency, cost, and outcomes.
2. Implement proper streaming passthrough and explicit retry boundaries.
3. Add task-aware routing and a reproducible benchmark suite.
4. Add broader protocol translation and provider discovery.

## Environment variables

All configuration is read from environment variables (or `.env`, loaded at startup). Variable names and safe placeholder usage:

| Name | Purpose | Safe value |
| --- | --- | --- |
| `NEOROUTE_PROVIDERS` | JSON array of provider objects, each with `name`, `base_url`, `api_key_env`, `model`, and optional `capabilities` | the JSON shown in `.env.example` |
| `NEOROUTE_TIMEOUT_SECONDS` | Per-provider request timeout in seconds | `45` |
| `NEOROUTE_MAX_ATTEMPTS` | Maximum provider attempts per request | `3` |
| `OPENAI_API_KEY`, `GROQ_API_KEY` | Key *names* are whatever `api_key_env` references; values are read from the environment at request time | `replace-me` |

Copy `.env.example` to `.env` and replace the placeholders with real values locally. `.env` must never be committed; do not paste real keys into source files, tests, CI configuration, or commits.

## Repository hygiene

Ignored paths: `.env` and `.env.*` (except `.env.example`), `.venv/`, `__pycache__/`, `.pytest_cache/`, `*.py[cod]`, `*.egg-info/`, `build/`, `dist/`, coverage output (`.coverage`, `coverage.xml`, `htmlcov/`), and `_research/` (external repositories cloned for research purposes).

## Continuous integration

`.github/workflows/ci.yml` runs `pip install -e ".[dev]"` followed by `pytest -q` on Python 3.11–3.14 for every push and pull request. The workflow needs no secrets.

