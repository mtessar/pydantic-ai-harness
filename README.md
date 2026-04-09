# Pydantic Harness

Composable, reusable capabilities for [Pydantic AI](https://ai.pydantic.dev/) agents.

## What is it?

Pydantic Harness provides a library of **capabilities** -- self-contained bundles of system prompts, tools, and lifecycle hooks -- that you can attach to any Pydantic AI agent to give it new powers without writing boilerplate.

Each capability is an [`AbstractCapability`](https://ai.pydantic.dev/capabilities/) subclass that plugs into the agent loop via Pydantic AI's capabilities API.

## Installation

```bash
pip install pydantic-harness
```

Requires Python 3.10+ and `pydantic-ai-slim>=1.78.0`.

## Quick start

```python
from pydantic_ai import Agent
from pydantic_harness import Memory, Skills, Compaction

agent = Agent(
    'openai:gpt-4o',
    capabilities=[Memory(), Skills(), Compaction()],
)

result = agent.run_sync('Remember that my favourite colour is blue.')
```

## Available capabilities

| Capability | Description |
|---|---|
| AdaptiveReasoning | Dynamically adjust reasoning effort based on task complexity |
| Approval | Require human approval before executing sensitive operations |
| Compaction | Compress conversation history to stay within context limits |
| FileSystem | Read, write, and navigate the local filesystem |
| Guardrails | Validate inputs/outputs and enforce cost and tool constraints |
| KnowsCurrentTime | Inject the current date and time into the system prompt |
| Memory | Persistent key-value memory across agent sessions |
| Planning | Break complex tasks into plans before execution |
| RepoContextInjection | Inject repository structure and context into the system prompt |
| SecretMasking | Detect and redact secrets in agent inputs and outputs |
| SessionPersistence | Save and restore full conversation sessions |
| Shell | Execute shell commands with safety controls |
| Skills | Progressive tool loading via search and activate |
| SlidingWindow | Keep conversation history within a sliding token window |
| StuckLoopDetection | Detect and break out of repetitive agent loops |
| SubAgent | Delegate subtasks to specialised child agents |
| SystemReminders | Inject periodic reminders into the conversation |
| ToolErrorRecovery | Automatically retry or recover from tool execution errors |
| ToolOrphanRepair | Repair orphaned tool calls in conversation history |
| ToolOutputManagement | Control and format tool output for the model |

## Code Mode

The `CodeMode` capability replaces individual tool calls with a single `run_code` tool. Instead of calling tools one at a time -- each requiring a separate round-trip to the model -- the model writes Python code that orchestrates multiple tools at once with loops, conditionals, variables, and parallel execution, all inside a sandboxed [Monty](https://github.com/pydantic/monty) runtime.

**Key advantages:**

- **Fewer model round-trips**: fetching 10 items and looking up details for each = 11 model calls with standard tool calling, but just 1 with code mode.
- **Parallelism**: independent tool calls run concurrently via async/await, without waiting for each to complete before starting the next.
- **Smaller context**: fewer round-trips means less conversation history, saving tokens and keeping the model focused.
- **Local processing**: the model can filter, transform, and aggregate results in code without additional model calls.

Further reading: [Tool use via code](https://www.anthropic.com/engineering/code-execution-with-mcp) (Anthropic), [Code mode in production](https://blog.cloudflare.com/code-mode/) (Cloudflare).

### Basic usage

```python
from pydantic_ai import Agent
from pydantic_harness import CodeMode

agent = Agent('anthropic:claude-sonnet-4-6', capabilities=[CodeMode()])

@agent.tool_plain
def get_weather(city: str) -> dict:
    """Get current weather for a city."""
    return {'city': city, 'temp_f': 72, 'condition': 'sunny'}

@agent.tool_plain
def convert_temp(fahrenheit: float) -> float:
    """Convert Fahrenheit to Celsius."""
    return round((fahrenheit - 32) * 5 / 9, 1)

result = agent.run_sync("What's the weather in Paris and Tokyo, in Celsius?")
```

The model sees a single `run_code` tool whose description includes the signatures of all available tools as async Python functions. It writes code like:

```python
# Fire both lookups concurrently
future_paris = get_weather(city="Paris")
future_tokyo = get_weather(city="Tokyo")
paris = await future_paris
tokyo = await future_tokyo

# Convert locally
paris_c = await convert_temp(fahrenheit=paris["temp_f"])
tokyo_c = await convert_temp(fahrenheit=tokyo["temp_f"])
{"paris": paris_c, "tokyo": tokyo_c}  # last expression returned automatically
```

### Selective tool sandboxing

By default, `CodeMode(tools='all')` sandboxes every tool. You can selectively choose which tools to sandbox:

```python
# By name
CodeMode(tools=['search', 'fetch'])

# By predicate
CodeMode(tools=lambda ctx, td: td.name != 'dangerous_tool')

# By metadata -- use with SetToolMetadata or .with_metadata() on toolsets
CodeMode(tools={'code_mode': True})
```

When using the metadata dict selector, mark tools for sandboxing with `SetToolMetadata` or `.with_metadata()`:

```python
from pydantic_ai import Agent
from pydantic_ai.toolsets import FunctionToolset
from pydantic_harness import CodeMode

search_tools = FunctionToolset(tools=[search, fetch]).with_metadata(code_mode=True)

agent = Agent(
    'anthropic:claude-sonnet-4-6',
    toolsets=[search_tools],
    capabilities=[CodeMode(tools={'code_mode': True})],
)
```

Tools that match the selector are wrapped inside `run_code`; non-matching tools remain available as regular tool calls.

### Return values

The last expression in the code snippet is automatically captured as the return value -- the model does **not** need to `print()` it. `print()` output is only useful for supplementary logging.

- **No print output**: the last expression's value is returned directly.
- **With print output**: returns `{"output": "<printed text>", "result": <last expression>}`.
- **Multimodal content** (e.g. binary images from tools): returned natively so the model can process them.

### Nested tool call metadata

The `run_code` tool return includes metadata with all nested tool calls and their results, keyed by tool call ID:

```python
result = await agent.run('...')

# Access the run_code ToolReturnPart from messages
for msg in result.all_messages():
    for part in msg.parts:
        if isinstance(part, ToolReturnPart) and part.tool_name == 'run_code':
            tool_calls = part.metadata['tool_calls']      # dict[str, ToolCallPart]
            tool_returns = part.metadata['tool_returns']   # dict[str, ToolReturnPart]
```

This is useful for observability, audit logging, or building UIs that show what happened inside each `run_code` invocation. When the agent is instrumented with Logfire/OTel, nested tool calls produce their own spans.

## Documentation

- [Pydantic AI docs](https://ai.pydantic.dev/)
- [Capabilities API](https://ai.pydantic.dev/capabilities/)

## Development

```bash
make install   # install dependencies
make lint      # ruff format check + lint
make typecheck # pyright strict
make test      # pytest
make testcov   # pytest with coverage
```

## License

MIT
