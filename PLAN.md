# Plan: ToolOutputManagement capability

Closes #82

## Context

When tools return large outputs (file contents, search results, command output), they can consume most of the context window. A single `grep -r` or verbose test output can crowd out all useful conversation history.

Issue #82 originally proposed this as harness infrastructure baked into the agent loop, but per maintainer feedback, this is better implemented as a configurable capability that uses the existing `after_tool_execute` hook on `AbstractCapability`.

## Design

A `ToolOutputManagement` capability (dataclass extending `AbstractCapability`) that:

1. **Intercepts tool results** via `after_tool_execute` -- the standard hook that fires after every tool execution, before the result enters the model's context
2. **Measures output size** by converting the result to its string representation
3. **Truncates when over limit** using one of three strategies:
   - `head` -- keep first N chars
   - `tail` -- keep last N chars (good for build/test output)
   - `head_tail` (default) -- keep first 60% + last 40% with middle elided
4. **Supports per-tool overrides** via `per_tool_limits` and `per_tool_strategies` dicts
5. **Supports custom summarization** via an optional `summarize_fn(tool_name, output) -> str` (sync or async), with truncation as a safety net if the summary still exceeds the limit

### What it does NOT do (deliberately)

- **Spill-to-file**: The issue proposed writing full output to files and returning paths. This requires filesystem access and assumptions about the execution environment. Better to leave this to a more specialized capability or to the `summarize_fn` hook.
- **Token-based limits**: Character limits are a simple, model-independent proxy. Token counting requires model-specific tokenizers and adds complexity. Can be added later.
- **Model-aware scaling**: Adjusting limits based on `ModelProfile.context_window` is a good idea but depends on #35 (ContextWindowTracker) which doesn't exist yet.

### Key property: original preserved upstream

The `after_tool_execute` hook modifies only what the model sees. The original full result is already captured in telemetry/trajectory before this hook fires, so no data is lost.

## Files

- `src/pydantic_harness/tool_output_management.py` -- the capability
- `src/pydantic_harness/__init__.py` -- re-exports `ToolOutputManagement` and `TruncationStrategy`
- `tests/test_tool_output_management.py` -- unit + integration tests (27 tests)

## Usage

```python
from pydantic_ai import Agent
from pydantic_harness import ToolOutputManagement, TruncationStrategy

agent = Agent(
    'openai:gpt-4o',
    capabilities=[
        ToolOutputManagement(
            max_output_chars=8000,
            strategy=TruncationStrategy.head_tail,
            per_tool_limits={'bash': 4000},
            per_tool_strategies={'bash': TruncationStrategy.tail},
        ),
    ],
)
```
