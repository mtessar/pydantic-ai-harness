"""Tool output management capability.

Intercepts tool return values and truncates or summarizes large outputs
to prevent context window blowup. Uses the `after_tool_execute` hook so
that the original tool result is preserved in telemetry / trajectory logs,
while only the LLM sees the truncated version.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from pydantic_ai.capabilities.abstract import AbstractCapability, ValidatedToolArgs
from pydantic_ai.messages import ToolCallPart
from pydantic_ai.tools import AgentDepsT, RunContext, ToolDefinition


class TruncationStrategy(str, Enum):
    """Strategy for truncating oversized tool output."""

    head = 'head'
    """Keep only the first characters."""

    tail = 'tail'
    """Keep only the last characters."""

    head_tail = 'head_tail'
    """Keep the first and last characters, eliding the middle."""


SummarizeFn = Callable[[str, str], str | Awaitable[str]]
"""A function `(tool_name, output) -> summarized_output`.

May be sync or async.
"""


def _head_tail_default_split(limit: int) -> tuple[int, int]:
    """Split a character limit into head and tail portions (60/40)."""
    head = int(limit * 0.6)
    tail = limit - head
    return head, tail


def _truncate(text: str, limit: int, strategy: TruncationStrategy) -> str:
    """Apply a truncation strategy to *text* that exceeds *limit* chars."""
    total = len(text)
    if total <= limit:
        return text

    if strategy is TruncationStrategy.head:
        kept = text[:limit]
        return f'{kept}\n\n[Truncated: showing first {limit:,} of {total:,} chars]'

    if strategy is TruncationStrategy.tail:
        kept = text[-limit:]
        return f'[Truncated: showing last {limit:,} of {total:,} chars]\n\n{kept}'

    # head_tail
    head_chars, tail_chars = _head_tail_default_split(limit)
    head_part = text[:head_chars]
    tail_part = text[-tail_chars:]
    omitted = total - head_chars - tail_chars
    return (
        f'{head_part}\n\n'
        f'[Truncated: {omitted:,} chars omitted from middle; showing first {head_chars:,} + last {tail_chars:,} of {total:,} chars]\n\n'
        f'{tail_part}'
    )


def _stringify(value: Any) -> str:
    """Convert an arbitrary tool return value to a string for size measurement."""
    if isinstance(value, str):
        return value
    return str(value)


@dataclass
class ToolOutputManagement(AbstractCapability[AgentDepsT]):
    """Manage large tool outputs to prevent context window blowup.

    Intercepts tool return values via the `after_tool_execute` hook and
    truncates or summarizes them when they exceed a configurable character
    limit.  The original (full) result is preserved upstream (telemetry,
    `FunctionToolResultEvent.content`); only the value forwarded to the
    model is modified.

    Example:
        ```python
        from pydantic_ai import Agent
        from pydantic_harness import ToolOutputManagement

        agent = Agent(
            'openai:gpt-4o',
            capabilities=[
                ToolOutputManagement(max_output_chars=8000),
            ],
        )
        ```
    """

    max_output_chars: int = 10_000
    """Default character limit for tool outputs.  Outputs exceeding this
    are truncated according to `strategy`."""

    strategy: TruncationStrategy = TruncationStrategy.head_tail
    """Default truncation strategy applied when output exceeds
    `max_output_chars`."""

    per_tool_limits: dict[str, int] = field(default_factory=lambda: {})
    """Per-tool character limits.  Keys are tool names; values override
    `max_output_chars` for that tool."""

    per_tool_strategies: dict[str, TruncationStrategy] = field(default_factory=lambda: {})
    """Per-tool truncation strategies.  Keys are tool names; values
    override `strategy` for that tool."""

    summarize_fn: SummarizeFn | None = None
    """Optional summarization function called *instead of* truncation.

    Receives `(tool_name, full_output_str)` and must return a
    (potentially shorter) string.  If the returned string still exceeds
    the limit, it is truncated as a safety net.

    May be sync or async.
    """

    async def after_tool_execute(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: ValidatedToolArgs,
        result: Any,
    ) -> Any:
        """Truncate or summarize the tool result if it exceeds the configured limit."""
        text = _stringify(result)
        limit = self.per_tool_limits.get(call.tool_name, self.max_output_chars)

        if len(text) <= limit:
            return result

        # Summarize path
        if self.summarize_fn is not None:
            summary = self.summarize_fn(call.tool_name, text)
            if isinstance(summary, Awaitable):
                summary = await summary
            assert isinstance(summary, str)
            # Safety net: if the summary itself is too long, truncate it
            if len(summary) > limit:
                strategy = self.per_tool_strategies.get(call.tool_name, self.strategy)
                return _truncate(summary, limit, strategy)
            return summary

        # Truncation path
        strategy = self.per_tool_strategies.get(call.tool_name, self.strategy)
        return _truncate(text, limit, strategy)
