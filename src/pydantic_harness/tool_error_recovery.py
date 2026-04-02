"""Tool error recovery capability for PydanticAI agents.

Catches unhandled tool execution errors and applies a configurable recovery
strategy so the agent run can continue instead of crashing.

Strategies:
    - ``inform`` (default): Return a descriptive error message to the model
      so it can adapt its approach.
    - ``retry``: Retry the failed tool call up to *N* times before falling
      back to ``inform``.
    - ``fallback``: Return a static fallback value on error.

Per-tool strategies can be configured via ``tool_strategies``, with a
``default_strategy`` applied to any tools not listed.

Example:
    ```python
    from pydantic_ai import Agent
    from pydantic_harness import ToolErrorRecovery

    agent = Agent(
        'openai:gpt-4.1',
        capabilities=[
            ToolErrorRecovery(
                default_strategy='inform',
                tool_strategies={
                    'flaky_api': ('retry', 3),
                    'optional_lookup': ('fallback', None),
                },
            ),
        ],
    )
    ```
"""

from __future__ import annotations

import traceback
from dataclasses import dataclass, field
from typing import Any

from pydantic_ai.capabilities.abstract import AbstractCapability, ValidatedToolArgs, WrapToolExecuteHandler
from pydantic_ai.messages import ToolCallPart
from pydantic_ai.tools import RunContext, ToolDefinition

# ---------------------------------------------------------------------------
# Strategy types
# ---------------------------------------------------------------------------

InformStrategy = str  # Literal['inform'], but we use str for 3.10 compat
RetryStrategy = tuple[str, int]  # ('retry', max_retries)
FallbackStrategy = tuple[str, Any]  # ('fallback', value)

Strategy = InformStrategy | RetryStrategy | FallbackStrategy
"""A recovery strategy.

- ``'inform'``: Return the error message to the model.
- ``('retry', N)``: Retry up to *N* times, then fall back to ``inform``.
- ``('fallback', value)``: Return *value* on error.
"""


def _validate_strategy(strategy: Strategy, label: str = 'strategy') -> None:
    """Raise ``ValueError`` if *strategy* is not a well-formed :data:`Strategy`.

    Note: accepts ``Strategy`` at the type level but performs full runtime
    validation (including shape checks) because strategies can come from
    untyped sources like YAML specs.
    """
    if isinstance(strategy, str):
        if strategy != 'inform':
            raise ValueError(f"Invalid {label}: string strategy must be 'inform', got {strategy!r}")
        return
    # strategy is RetryStrategy | FallbackStrategy (a 2-tuple) per the type,
    # but at runtime it could be anything if coming from untyped input.
    try:
        kind, value = strategy  # type: ignore[misc]
    except (TypeError, ValueError):
        raise ValueError(f'Invalid {label}: expected a string or 2-tuple, got {strategy!r}') from None
    if kind == 'retry':
        if not isinstance(value, int) or value < 1:
            raise ValueError(f'Invalid {label}: retry max_retries must be a positive integer, got {value!r}')
    elif kind == 'fallback':
        pass  # any value is acceptable
    else:
        raise ValueError(f"Invalid {label}: tuple strategy kind must be 'retry' or 'fallback', got {kind!r}")


def _format_error(tool_name: str, error: Exception, *, include_traceback: bool) -> str:
    """Build a human-readable error string for the model."""
    parts = [f'Error in tool `{tool_name}` ({type(error).__name__}): {error}']
    if include_traceback:
        tb = traceback.format_exception(type(error), error, error.__traceback__)
        parts.append('Traceback:\n' + ''.join(tb))
    return '\n'.join(parts)


# ---------------------------------------------------------------------------
# Convenience constructors
# ---------------------------------------------------------------------------


def retry(max_retries: int = 3) -> RetryStrategy:
    """Create a retry strategy.

    Args:
        max_retries: Maximum number of retry attempts before falling back to ``inform``.
    """
    if not isinstance(max_retries, int) or max_retries < 1:  # pyright: ignore[reportUnnecessaryIsInstance]
        raise ValueError(f'max_retries must be a positive integer, got {max_retries!r}')
    return ('retry', max_retries)


def fallback(value: Any = None) -> FallbackStrategy:
    """Create a fallback strategy.

    Args:
        value: The value to return when the tool fails.
    """
    return ('fallback', value)


# ---------------------------------------------------------------------------
# Capability
# ---------------------------------------------------------------------------


@dataclass
class ToolErrorRecovery(AbstractCapability[Any]):
    """Catch tool execution errors and recover gracefully.

    Instead of letting unhandled exceptions crash the agent run, this
    capability intercepts failures via the ``on_tool_execute_error`` hook
    and applies a configurable strategy per tool.

    Strategies:
        - ``'inform'`` (default) -- Return a descriptive error message to the
          model so it can adjust its approach.
        - ``('retry', N)`` -- Retry the tool call up to *N* times. If all
          retries fail, fall back to ``inform``.
        - ``('fallback', value)`` -- Return a static value on error.

    Per-tool configuration is available via ``tool_strategies``.  Any tool
    not listed uses ``default_strategy``.

    Example::

        from pydantic_ai import Agent
        from pydantic_harness import ToolErrorRecovery

        agent = Agent(
            'openai:gpt-4.1',
            capabilities=[
                ToolErrorRecovery(
                    tool_strategies={
                        'flaky_api': ('retry', 3),
                        'optional_lookup': ('fallback', None),
                    },
                ),
            ],
        )
    """

    default_strategy: Strategy = 'inform'
    """Strategy applied to tools not listed in ``tool_strategies``."""

    tool_strategies: dict[str, Strategy] = field(default_factory=lambda: dict[str, Strategy]())
    """Per-tool strategy overrides.  Keys are tool names."""

    include_traceback: bool = False
    """Whether to include the Python traceback in error messages sent to the model.

    Useful for debugging but may waste tokens in production.
    """

    # --- Per-run state (populated by ``for_run``) ---

    _retry_counts: dict[str, int] = field(default_factory=lambda: dict[str, int](), repr=False)
    """Tracks per-tool retry counts within a single run.  Keys are ``tool_name``."""

    def __post_init__(self) -> None:
        """Validate strategies at construction time."""
        _validate_strategy(self.default_strategy, 'default_strategy')
        for name, strat in self.tool_strategies.items():
            _validate_strategy(strat, f'tool_strategies[{name!r}]')

    @classmethod
    def get_serialization_name(cls) -> str | None:
        """Return serialization name for spec construction."""
        return 'ToolErrorRecovery'

    async def for_run(self, ctx: RunContext[Any]) -> ToolErrorRecovery:
        """Return a fresh instance with empty retry counts for each agent run."""
        return ToolErrorRecovery(
            default_strategy=self.default_strategy,
            tool_strategies=self.tool_strategies,
            include_traceback=self.include_traceback,
        )

    def _get_strategy(self, tool_name: str) -> Strategy:
        """Look up the strategy for a given tool."""
        return self.tool_strategies.get(tool_name, self.default_strategy)

    async def wrap_tool_execute(
        self,
        ctx: RunContext[Any],
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: ValidatedToolArgs,
        handler: WrapToolExecuteHandler,
    ) -> Any:
        """Wrap tool execution to implement retry logic.

        The ``retry`` strategy requires re-invoking the tool, which can only
        be done from ``wrap_tool_execute`` (``on_tool_execute_error`` fires
        after the tool has already failed and cannot re-invoke it).
        """
        strategy = self._get_strategy(call.tool_name)
        if not (isinstance(strategy, tuple) and strategy[0] == 'retry'):
            # Non-retry strategies are handled by on_tool_execute_error.
            return await handler(args)

        max_retries: int = strategy[1]
        last_error: Exception | None = None

        for attempt in range(1 + max_retries):
            try:
                result = await handler(args)
                # Success -- reset retry count for this tool.
                self._retry_counts.pop(call.tool_name, None)
                return result
            except Exception as exc:
                last_error = exc
                self._retry_counts[call.tool_name] = attempt + 1
                if attempt < max_retries:
                    continue
                # All retries exhausted -- fall through to inform.
                return _format_error(call.tool_name, exc, include_traceback=self.include_traceback)

        # Unreachable, but satisfies the type checker.
        raise last_error  # type: ignore[misc] # pragma: no cover

    async def on_tool_execute_error(
        self,
        ctx: RunContext[Any],
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: ValidatedToolArgs,
        error: Exception,
    ) -> Any:
        """Handle tool execution errors for non-retry strategies.

        For ``retry`` strategies, errors are handled by ``wrap_tool_execute``
        and this hook is not reached (the wrapper catches exceptions before
        they propagate to the hook).
        """
        strategy = self._get_strategy(call.tool_name)

        if isinstance(strategy, tuple) and strategy[0] == 'fallback':
            return strategy[1]

        # Default: 'inform' strategy.
        return _format_error(call.tool_name, error, include_traceback=self.include_traceback)


# ---------------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------------

__all__ = [
    'ToolErrorRecovery',
    'Strategy',
    'retry',
    'fallback',
]
