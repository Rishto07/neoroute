# Neoroute Product Validation Report: Response-Contract Regression Attribution

## 1. Executive summary

Neoroute is a small, auditable, OpenAI-compatible AI gateway. In the current repository, it forwards chat requests to configured providers, falls back on transient errors, and returns trace headers. It does not yet implement the proposed response-contract regression-attribution feature as a released user-facing capability in this branch context.

The proposed feature is a deterministic, privacy-preserving report that explains why a provider's successful response became invalid, malformed, empty, or mismatched, without exposing prompts, completions, response bodies, or credentials.

Public evidence shows that malformed or mismatched successful AI responses are a real debugging problem in related systems. However, no Neoroute users have been interviewed, no outreach has been sent, and no fixtures have been collected.

**Recommendation: DEFER.** The product decision should wait for real interviews and sanitized fixtures. This is not a rejection of the idea; it means demand is currently unvalidated.

## 2. What Neoroute currently implements

Verified in the repository:

- FastAPI app with `/health` and `POST /v1/chat/completions`.
- Provider configuration loading and validation.
- Request schema validation.
- Provider ordering and named-provider preference.
- Fallback on 429, 5xx, timeouts, and network errors.
- Immediate failure on other 4xx responses.
- Trace headers for provider, attempts, and latency.
- Unit tests for routing, fallback, capabilities, response contracts, and corpus evaluation.

Remote documentation and newer branches describe additional capabilities, including response verification and corpus evaluation. This report does not present unmerged or branch-specific work as released functionality.

## 3. Product opportunity being evaluated

The opportunity is a deterministic, privacy-preserving response-contract regression-attribution report. Its intended value is to help developers understand what changed between an expected response shape and a current response, using compact failure signatures and category-level deltas.

## 4. Research method

The research used:

- Repository and documentation inspection.
- Public GitHub issue review.
- Competitor documentation review.
- Candidate identification from public sources.
- Preparation of outreach templates, interview guide, tracker, and scorecard.

No interviews, outreach, quotes, or fixtures were completed. Public issue reports are context, not Neoroute customers or user interviews.

## 5. Verified public evidence

### Vexa-ai/vexa #1666

- URL: https://github.com/Vexa-ai/vexa/issues/1666
- Problem: A custom gateway returned an HTTP 200 with an OpenAI-shaped body where an Anthropic Messages response was expected. The harness reported "empty or malformed response." A 401 retry loop also occurred when `Authorization: Bearer` was sent to an endpoint expecting `x-api-key`.
- Impact: Agent chat turns could not run. The UI stayed on "Working" for about 70 seconds and retried about every 5 seconds. A CDN could return an HTML page when transfer encoding and content length were both forwarded.
- Maintainer response: The maintainer verified the diagnosis, created PR #1682, and stated that 2xx bodies are judged by shape, 401 is terminal, and settings tests now ask each call shape its own question at its own path with only its own auth header.
- Relevance: High for malformed successful responses; medium for regression attribution because the fix focused on body-shape validation and dialect separation.

### AI-Adriane/ailu-core #268

- URL: https://github.com/AI-Adriane/ailu-core/issues/268
- Problem: Several parsing paths replace malformed or unexpected model output with a plausible default instead of reporting it. Examples include invalid tool-call arguments becoming `{}` and an HTTP 200 with an unexpected body becoming an empty response.
- Impact: Tools can run with arguments nobody wrote; agent runs can complete successfully with an empty answer; reflection can trigger extra LLM round-trips due to substring matching.
- Maintainer response: No maintainer response was visible in the fetched page.
- Relevance: High for malformed or unexpected successful responses; medium for regression attribution.

### lsdefine/GenericAgent #815

- URL: https://github.com/lsdefine/GenericAgent/issues/815
- Problem: GenericAgent can receive a syntactically valid but semantically empty completion, such as a text block containing only a whitespace character. Third-party relays or gateways may return an HTTP-successful response while dropping, truncating, or normalizing streamed content to whitespace.
- Impact: The TUI shows "LLM returned an empty response. Retrying..." and retries. The failure is recoverable but confusing to diagnose because intermediaries make attribution harder.
- Maintainer or contributor response: A public commenter proposed a compact redacted diagnostic and later stated that PR #819 implemented the diagnostic and TUI severity label, while leaving retry behavior unchanged.
- Relevance: High for empty or whitespace-only successful responses; medium for regression attribution.

### microsoft/agent-framework #6067

- URL: https://github.com/microsoft/agent-framework/issues/6067
- Problem: LLMs occasionally return malformed or slightly corrupted JSON responses, including markdown fences, trailing commas, truncated outputs, and stringified fields.
- Impact: The issue argues that handling this at the parser layer would improve resilience and reduce redundant error handling.
- Maintainer or contributor response: A public commenter suggested separating syntax repair from semantic recovery, capturing raw argument strings and provider stop reasons, applying deterministic syntax-only repairs, parsing with `System.Text.Json`, validating against schema, and retaining an audit list of applied repairs. The commenter also referenced a public 14-case acceptance corpus and threat model.
- Relevance: High for structured outputs and tool calls; low to medium for gateway-level regression attribution.

## 6. Competitive landscape

Inspected documentation for LiteLLM, Portkey, OpenRouter, and promptfoo shows broader platforms covering routing, fallback, observability, budgets, model access, and evaluation. The proposed Neoroute feature may be differentiated as a small, auditable, gateway-path response-contract check, but this differentiation is not yet validated by users.

Sources:

- LiteLLM: https://github.com/BerriAI/litellm and https://docs.litellm.ai
- Portkey: https://docs.portkey.ai/docs/introduction/what-is-portkey
- OpenRouter: https://openrouter.ai/docs/quickstart
- promptfoo: https://www.promptfoo.dev/docs/getting-started/

## 7. Candidate interview participants

The following public candidates were identified. All are marked `NOT CONTACTED`.

| Candidate | Project | Relevant public evidence | Why relevant |
|---|---|---|---|
| timjuedemann | Vexa-ai/vexa | https://github.com/Vexa-ai/vexa/issues/1666 | Reported a detailed malformed HTTP 200 and 401 retry loop. |
| DmitriyG228 | Vexa-ai/vexa | https://github.com/Vexa-ai/vexa/issues/1666 | Maintainer who described body-shape validation and terminal 401 handling. |
| prxmat | AI-Adriane/ailu-core | https://github.com/AI-Adriane/ailu-core/issues/268 | Reported malformed or unexpected model output being treated as valid. |
| ZhulongNT | lsdefine/GenericAgent | https://github.com/lsdefine/GenericAgent/issues/815 | Reported whitespace-only LLM responses from third-party relays. |
| louisss1016 | lsdefine/GenericAgent | https://github.com/lsdefine/GenericAgent/issues/815 | Publicly discussed a compact redacted diagnostic and related PR. |
| thnak | microsoft/agent-framework | https://github.com/microsoft/agent-framework/issues/6067 | Requested handling for malformed JSON responses in tool calling and structured outputs. |
| gilgold | microsoft/agent-framework | https://github.com/microsoft/agent-framework/issues/6067 | Publicly discussed syntax repair, semantic recovery, and audit trails. |

Contact should use only public GitHub profiles or issue threads. No private contact details were collected.

## 8. Outreach readiness and next actions

The research workspace contains supporting documents:

- `research-plan.md`
- `outreach-tracker.md`
- `interview-guide.md`
- `outreach-templates.md`
- `validation-scorecard.md`
- `public-evidence-review.md`

Outreach messages are ready to send, but they have not been sent. The project lead must perform outreach.

## 9. Evidence supporting the opportunity

- Malformed or mismatched successful responses appear in multiple public issue reports.
- Some maintainers have implemented or proposed body-shape validation or diagnostics.
- Neoroute already has fallback and request validation, making response verification a plausible next direction.

## 10. Evidence against the opportunity and alternative explanations

- No Neoroute users or issues request the feature.
- Public fixes focused on body-shape validation and diagnostics, not baseline regression attribution.
- Developers may prefer broader platforms or prompt-level evaluation tools.
- Baseline maintenance and false positives may reduce usefulness.
- Public issue reports are not evidence of demand for Neoroute specifically.

## 11. Privacy, trust, false-positive, and maintenance risks

- The report must avoid prompts, completions, response bodies, and credentials.
- Privacy-safe summaries may be too limited for some debugging needs.
- False positives could make CI noisy and reduce trust.
- Baselines may drift as providers change.
- Maintenance burden may discourage adoption.

## 12. Success criteria and decision threshold

Proceed only if:

- At least two qualified developers independently describe a relevant real-world problem.
- At least two independently indicate that they would use the proposed report in CI or a comparable workflow.
- There is willingness to share sanitized fixtures, or another concrete way to test the proposed report.
- The report can be useful without exposing sensitive data.
- False positives and baseline maintenance appear manageable enough to investigate further.

Current counts:

- People contacted: 0
- Responses received: 0
- Interviews completed: 0
- Fixtures received: 0

Threshold status: NOT MET.

## 13. Recommendation

**DEFER.**

The opportunity is plausible and supported by related public evidence, but user demand is unvalidated. Do not implement the feature yet. Complete outreach and interviews first, then decide whether to proceed, revise, or pivot.

## 14. Immediate next steps

1. Send the two personalized outreach messages.
2. Log all outreach activity in `outreach-tracker.md`.
3. Conduct 15-minute interviews using `interview-guide.md`.
4. Collect sanitized fixtures with permission.
5. Update `validation-scorecard.md` with verified findings.
6. Re-run the decision threshold after at least two interviews.

## 15. Limitations and what remains unvalidated

- No interviews have been completed.
- No users have been contacted.
- No fixtures have been received.
- Public issues are not Neoroute customers.
- Competitor absence claims are based only on inspected documentation.
- Remote Neoroute features and unmerged pull requests are not treated as released functionality.

## Outreach message template, marked as not sent

The following template has not been sent. It is included for the project lead to use.

```text
Hi! I'm a student working on Neoroute, an open-source AI gateway project. I came across your work on [project/issue] and wanted to learn how developers diagnose reliability problems in multi-provider AI systems. Have you encountered malformed, empty, or schema-invalid model responses that were difficult to diagnose? If so, would you be open to a short 15-minute conversation about how you handled the issue? I'm researching developer workflows, not trying to sell anything. Thanks!
```

Personalized versions for timjuedemann and DmitriyG228 are available in `outreach-templates.md`. They have not been sent.