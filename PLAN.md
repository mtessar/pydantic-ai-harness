# Guardrails Capability Plan

## Goal

Provide four reusable `AbstractCapability` subclasses for common safety and cost-control concerns:

| Capability | Hook used | Purpose |
|---|---|---|
| `InputGuardrail` | `before_run` | Validate user input before the agent starts |
| `OutputGuardrail` | `after_run` | Validate model output before returning to the caller |
| `CostGuard` | `before_model_request` | Enforce token budget limits per run |
| `ToolGuard` | `prepare_tools` + `before_tool_execute` | Block tools or require approval |

## Design Decisions

### Guard functions are user-supplied callables

`InputGuardrail` and `OutputGuardrail` accept a `guard: GuardFunc` -- a sync or async `(str) -> bool` function where `True` means "safe".  This keeps the capabilities general-purpose: users bring their own validation logic (regex, moderation API, LLM judge, etc.) and the capability handles the lifecycle plumbing.

Because the guard is a callable, these capabilities are not spec-serializable (`get_serialization_name` returns `None`).

### CostGuard uses token counts, not USD estimates

Unlike the `CostTracking` capability in pydantic-ai-shields (which depends on `genai-prices` for per-model USD pricing), `CostGuard` operates purely on token counts available from `ctx.usage`.  This avoids an external dependency and works reliably across all providers. Users can set `max_input_tokens`, `max_output_tokens`, and/or `max_total_tokens`.

The check runs in `before_model_request` so it fires before each LLM call, catching budget overruns mid-run rather than only at the end.

`CostGuard` is spec-serializable since it only takes simple numeric configuration.

### ToolGuard combines prepare_tools and before_tool_execute

- `blocked` tools are removed from the tool definitions the model sees (`prepare_tools`), so the model cannot even attempt to call them.
- `require_approval` tools are still visible to the model, but `before_tool_execute` checks an `approval_callback` before execution proceeds.  If no callback is configured, the tool call is denied.

This two-layer approach mirrors pydantic-ai-shields' `ToolGuard` and gives users precise control: hidden vs. gated.

### Exception hierarchy

All guardrail violations share a common base (`GuardrailError`) for catch-all handling, with specific subclasses for each violation type:

```
GuardrailError
  InputBlocked
  OutputBlocked
  BudgetExceededError
  ToolBlocked
```

### Sync and async guard/approval functions

Both sync and async functions are accepted everywhere (guard functions, approval callbacks).  At call time, `inspect.isawaitable` is used to detect and `await` coroutines.  This matches the pattern used throughout pydantic-ai's hook system.

## Prior Art

- **pydantic-ai-shields** (`vstorm-co/pydantic-ai-shields`): Direct inspiration.  `InputGuard`, `OutputGuard`, `CostTracking`, `ToolGuard`, and content shields (`PromptInjection`, `PiiDetector`, `SecretRedaction`, `BlockedKeywords`, `NoRefusals`).
- **OpenAI Agents SDK**: `InputGuardrails` and `OutputGuardrails` with a "tripwire" mechanism for parallel guard + LLM execution.
- **pydantic-ai #1197**: 20+ comments requesting guardrail support.

## Future Work (out of scope for this PR)

- **Content shields** (PromptInjection, PiiDetector, SecretRedaction, BlockedKeywords, NoRefusals) -- tracked in harness #47.
- **AsyncGuardrail** -- concurrent guardrail + LLM execution with cancellation, as in OpenAI Agents SDK.
- **USD cost estimation** via `genai-prices` or model profile pricing data.
- **Warning mode** -- log instead of raise when a guard fails.

## References

- Harness issue #28: Input/Output Guardrails capability
- Harness issue #46: Cost/Token Budget capability
- Harness issue #47: Safety guardrail implementations
- pydantic-ai #1197: Guardrails feature request
