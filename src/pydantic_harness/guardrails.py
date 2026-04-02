"""Guardrail capabilities for Pydantic AI agents.

Reusable capabilities for input/output validation, cost/token budget enforcement,
and per-tool permission control. Built on Pydantic AI's native capabilities API.

Example:
    ```python
    from pydantic_ai import Agent
    from pydantic_harness import InputGuardrail, OutputGuardrail, CostGuard, ToolGuard

    agent = Agent(
        'openai:gpt-4.1',
        capabilities=[
            InputGuardrail(guard=lambda text: 'DROP TABLE' not in text),
            OutputGuardrail(guard=lambda text: 'password' not in text.lower()),
            CostGuard(max_total_tokens=100_000),
            ToolGuard(blocked=['execute_sql'], require_approval=['delete_file']),
        ],
    )
    ```
"""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, TypeAlias

from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.messages import ToolCallPart
from pydantic_ai.tools import RunContext, ToolDefinition

# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class GuardrailError(Exception):
    """Base exception for guardrail violations."""


class InputBlocked(GuardrailError):
    """Raised when user input fails a guardrail check."""


class OutputBlocked(GuardrailError):
    """Raised when model output fails a guardrail check."""


class BudgetExceededError(GuardrailError):
    """Raised when token or cost budget is exceeded.

    Attributes:
        detail: A human-readable description of which limit was breached.
    """

    def __init__(self, detail: str) -> None:  # noqa: D107
        self.detail = detail
        super().__init__(detail)


class ToolBlocked(GuardrailError):
    """Raised when a tool call is denied by a guardrail.

    Attributes:
        tool_name: The name of the blocked tool.
        reason: Why the tool was blocked.
    """

    def __init__(self, tool_name: str, *, reason: str = '') -> None:  # noqa: D107
        self.tool_name = tool_name
        self.reason = reason
        msg = f"Tool '{tool_name}' blocked"
        if reason:
            msg += f': {reason}'
        super().__init__(msg)


# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

GuardFunc: TypeAlias = Callable[[str], bool] | Callable[[str], Awaitable[bool]]
"""A sync or async function that receives a text string and returns ``True`` if safe."""

ApprovalFunc: TypeAlias = Callable[[str, dict[str, Any]], bool] | Callable[[str, dict[str, Any]], Awaitable[bool]]
"""A sync or async function ``(tool_name, args) -> bool`` that grants or denies tool execution."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _call_guard(func: GuardFunc, text: str) -> bool:
    """Call a sync or async guard function and return its bool result."""
    result = func(text)
    if inspect.isawaitable(result):
        return await result
    return result  # type: ignore[return-value]


async def _call_approval(func: ApprovalFunc, tool_name: str, args: dict[str, Any]) -> bool:
    """Call a sync or async approval function and return its bool result."""
    result = func(tool_name, args)
    if inspect.isawaitable(result):
        return await result
    return result  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# InputGuardrail
# ---------------------------------------------------------------------------


@dataclass
class InputGuardrail(AbstractCapability[Any]):
    """Validate user input before the agent run starts.

    The guard function receives the user prompt as a string and returns ``True``
    if the input is acceptable.  When it returns ``False``, an
    :class:`InputBlocked` exception is raised and the run never starts.

    Both sync and async guard functions are accepted.

    Example:
        ```python
        from pydantic_ai import Agent
        from pydantic_harness import InputGuardrail

        async def check_toxicity(text: str) -> bool:
            # Call a moderation API ...
            return True

        agent = Agent('openai:gpt-4.1', capabilities=[InputGuardrail(guard=check_toxicity)])
        ```
    """

    guard: GuardFunc
    """Function that checks input safety.  Returns ``True`` if safe."""

    @classmethod
    def get_serialization_name(cls) -> str | None:
        """Not spec-serializable (takes a callable)."""
        return None

    async def before_run(self, ctx: RunContext[Any]) -> None:
        """Check user input before the run starts."""
        prompt = ctx.prompt
        if prompt is None:
            return

        prompt_str = str(prompt) if not isinstance(prompt, str) else prompt
        if not await _call_guard(self.guard, prompt_str):
            raise InputBlocked(f'Input blocked by guardrail: {prompt_str[:100]}')


# ---------------------------------------------------------------------------
# OutputGuardrail
# ---------------------------------------------------------------------------


@dataclass
class OutputGuardrail(AbstractCapability[Any]):
    """Validate model output after the agent run completes.

    The guard function receives the stringified output and returns ``True``
    if the output is acceptable.  When it returns ``False``, an
    :class:`OutputBlocked` exception is raised.

    Both sync and async guard functions are accepted.

    Example:
        ```python
        from pydantic_ai import Agent
        from pydantic_harness import OutputGuardrail

        def no_secrets(text: str) -> bool:
            return 'sk-' not in text

        agent = Agent('openai:gpt-4.1', capabilities=[OutputGuardrail(guard=no_secrets)])
        ```
    """

    guard: GuardFunc
    """Function that checks output safety.  Returns ``True`` if safe."""

    @classmethod
    def get_serialization_name(cls) -> str | None:
        """Not spec-serializable (takes a callable)."""
        return None

    async def after_run(self, ctx: RunContext[Any], *, result: Any) -> Any:
        """Check model output after the run completes."""
        output_str = str(result.output)
        if not await _call_guard(self.guard, output_str):
            raise OutputBlocked(f'Output blocked by guardrail: {output_str[:100]}')
        return result


# ---------------------------------------------------------------------------
# CostGuard
# ---------------------------------------------------------------------------


@dataclass
class CostGuard(AbstractCapability[Any]):
    """Enforce token budget limits during an agent run.

    Checks cumulative token usage via ``ctx.usage`` before each model request
    and raises :class:`BudgetExceededError` when a configured threshold is
    exceeded.

    At least one of ``max_input_tokens``, ``max_output_tokens``, or
    ``max_total_tokens`` must be set.

    Example:
        ```python
        from pydantic_ai import Agent
        from pydantic_harness import CostGuard

        agent = Agent(
            'openai:gpt-4.1',
            capabilities=[CostGuard(max_total_tokens=50_000)],
        )
        ```
    """

    max_input_tokens: int | None = None
    """Maximum cumulative input tokens allowed.  ``None`` means unlimited."""

    max_output_tokens: int | None = None
    """Maximum cumulative output tokens allowed.  ``None`` means unlimited."""

    max_total_tokens: int | None = None
    """Maximum cumulative total tokens (input + output) allowed.  ``None`` means unlimited."""

    @classmethod
    def get_serialization_name(cls) -> str | None:
        """Return serialization name for spec construction."""
        return 'CostGuard'

    async def before_model_request(self, ctx: RunContext[Any], request_context: Any) -> Any:
        """Check token budget before each model request."""
        usage = ctx.usage

        if self.max_input_tokens is not None and usage.input_tokens > self.max_input_tokens:
            raise BudgetExceededError(f'Input token budget exceeded: {usage.input_tokens}/{self.max_input_tokens}')

        if self.max_output_tokens is not None and usage.output_tokens > self.max_output_tokens:
            raise BudgetExceededError(f'Output token budget exceeded: {usage.output_tokens}/{self.max_output_tokens}')

        if self.max_total_tokens is not None:
            total = usage.input_tokens + usage.output_tokens
            if total > self.max_total_tokens:
                raise BudgetExceededError(f'Total token budget exceeded: {total}/{self.max_total_tokens}')

        return request_context


# ---------------------------------------------------------------------------
# ToolGuard
# ---------------------------------------------------------------------------


@dataclass
class ToolGuard(AbstractCapability[Any]):
    """Control per-tool access: block tools or require approval before execution.

    Blocked tools are hidden from the model entirely via ``prepare_tools``.
    Tools requiring approval trigger the ``approval_callback`` before execution;
    if no callback is configured or the callback returns ``False``, a
    :class:`ToolBlocked` exception is raised.

    Both sync and async approval callbacks are accepted.

    Example:
        ```python
        from pydantic_ai import Agent
        from pydantic_harness import ToolGuard

        async def ask_user(tool_name: str, args: dict) -> bool:
            return input(f'Allow {tool_name}? (y/n) ').lower() == 'y'

        agent = Agent(
            'openai:gpt-4.1',
            capabilities=[ToolGuard(
                blocked=['execute_sql'],
                require_approval=['delete_file', 'send_email'],
                approval_callback=ask_user,
            )],
        )
        ```
    """

    blocked: list[str] = field(default_factory=list[str])
    """Tool names to hide from the model entirely."""

    require_approval: list[str] = field(default_factory=list[str])
    """Tool names that require approval before execution."""

    approval_callback: ApprovalFunc | None = None
    """Callback ``(tool_name, args) -> bool``.  Required when ``require_approval`` is non-empty."""

    @classmethod
    def get_serialization_name(cls) -> str | None:
        """Not spec-serializable (takes a callable)."""
        return None

    async def prepare_tools(
        self,
        ctx: RunContext[Any],
        tool_defs: list[ToolDefinition],
    ) -> list[ToolDefinition]:
        """Hide blocked tools from the model."""
        if not self.blocked:
            return tool_defs
        blocked_set = frozenset(self.blocked)
        return [td for td in tool_defs if td.name not in blocked_set]

    async def before_tool_execute(
        self,
        ctx: RunContext[Any],
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: dict[str, Any],
    ) -> dict[str, Any]:
        """Enforce approval for tools in ``require_approval``."""
        if call.tool_name not in self.require_approval:
            return args

        if self.approval_callback is None:
            raise ToolBlocked(call.tool_name, reason='approval required but no callback configured')

        if not await _call_approval(self.approval_callback, call.tool_name, args):
            raise ToolBlocked(call.tool_name, reason='approval denied')

        return args


# ---------------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------------

__all__ = [
    # Capabilities
    'InputGuardrail',
    'OutputGuardrail',
    'CostGuard',
    'ToolGuard',
    # Exceptions
    'GuardrailError',
    'InputBlocked',
    'OutputBlocked',
    'BudgetExceededError',
    'ToolBlocked',
]
