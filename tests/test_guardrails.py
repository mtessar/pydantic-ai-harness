"""Tests for guardrail capabilities."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic_ai import Agent
from pydantic_ai.messages import ToolCallPart
from pydantic_ai.models.test import TestModel
from pydantic_ai.tools import RunContext, ToolDefinition
from pydantic_ai.usage import RunUsage

from pydantic_harness.guardrails import (
    BudgetExceededError,
    CostGuard,
    GuardrailError,
    InputBlocked,
    InputGuardrail,
    OutputBlocked,
    OutputGuardrail,
    ToolBlocked,
    ToolGuard,
)

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    """Restrict to asyncio — pydantic-ai internals use asyncio.gather."""
    return 'asyncio'


# ---------------------------------------------------------------------------
# Exception hierarchy
# ---------------------------------------------------------------------------


class TestExceptionHierarchy:
    def test_guardrail_error_is_base(self) -> None:
        assert issubclass(InputBlocked, GuardrailError)
        assert issubclass(OutputBlocked, GuardrailError)
        assert issubclass(BudgetExceededError, GuardrailError)
        assert issubclass(ToolBlocked, GuardrailError)

    def test_tool_blocked_attributes(self) -> None:
        err = ToolBlocked('my_tool', reason='denied')
        assert err.tool_name == 'my_tool'
        assert err.reason == 'denied'
        assert "Tool 'my_tool' blocked: denied" in str(err)

    def test_tool_blocked_no_reason(self) -> None:
        err = ToolBlocked('my_tool')
        assert err.reason == ''
        assert str(err) == "Tool 'my_tool' blocked"

    def test_budget_exceeded_detail(self) -> None:
        err = BudgetExceededError('Token budget exceeded: 200/100')
        assert err.detail == 'Token budget exceeded: 200/100'
        assert 'Token budget exceeded' in str(err)


# ---------------------------------------------------------------------------
# InputGuardrail
# ---------------------------------------------------------------------------


class TestInputGuardrail:
    async def test_sync_guard_allows(self) -> None:
        """Sync guard returning True should allow the run to proceed."""
        agent = Agent(TestModel(), capabilities=[InputGuardrail(guard=lambda text: True)])
        result = await agent.run('Hello')
        assert result.output is not None

    async def test_sync_guard_blocks(self) -> None:
        """Sync guard returning False should raise InputBlocked."""
        agent = Agent(TestModel(), capabilities=[InputGuardrail(guard=lambda text: False)])
        with pytest.raises(InputBlocked, match='Input blocked by guardrail'):
            await agent.run('Hello')

    async def test_async_guard_allows(self) -> None:
        """Async guard returning True should allow the run to proceed."""

        async def safe_check(text: str) -> bool:
            return True

        agent = Agent(TestModel(), capabilities=[InputGuardrail(guard=safe_check)])
        result = await agent.run('Hello')
        assert result.output is not None

    async def test_async_guard_blocks(self) -> None:
        """Async guard returning False should raise InputBlocked."""

        async def unsafe_check(text: str) -> bool:
            return False

        agent = Agent(TestModel(), capabilities=[InputGuardrail(guard=unsafe_check)])
        with pytest.raises(InputBlocked):
            await agent.run('Hello')

    async def test_guard_receives_prompt_text(self) -> None:
        """Guard function should receive the actual prompt text."""
        received: list[str] = []

        def capture(text: str) -> bool:
            received.append(text)
            return True

        agent = Agent(TestModel(), capabilities=[InputGuardrail(guard=capture)])
        await agent.run('test prompt 123')
        assert received == ['test prompt 123']

    async def test_guard_blocks_with_content_in_message(self) -> None:
        """The error message should include a truncated version of the input."""

        def block_sql(text: str) -> bool:
            return 'DROP TABLE' not in text

        agent = Agent(TestModel(), capabilities=[InputGuardrail(guard=block_sql)])
        with pytest.raises(InputBlocked, match='Input blocked by guardrail'):
            await agent.run('DROP TABLE users')

    async def test_none_prompt_skips_guard(self) -> None:
        """When prompt is None, the guard function should not be called."""
        called = False

        def guard(text: str) -> bool:  # pragma: no cover
            nonlocal called
            called = True
            return False

        guardrail = InputGuardrail(guard=guard)
        ctx = _make_run_context()
        await guardrail.before_run(ctx)
        assert not called

    def test_not_serializable(self) -> None:
        """InputGuardrail should not be spec-serializable (takes a callable)."""
        assert InputGuardrail.get_serialization_name() is None


# ---------------------------------------------------------------------------
# OutputGuardrail
# ---------------------------------------------------------------------------


class TestOutputGuardrail:
    async def test_sync_guard_allows(self) -> None:
        """Sync guard returning True should pass the result through."""
        agent = Agent(
            TestModel(custom_output_text='safe output'),
            capabilities=[OutputGuardrail(guard=lambda text: True)],
        )
        result = await agent.run('Hello')
        assert result.output == 'safe output'

    async def test_sync_guard_blocks(self) -> None:
        """Sync guard returning False should raise OutputBlocked."""
        agent = Agent(
            TestModel(custom_output_text='bad output'),
            capabilities=[OutputGuardrail(guard=lambda text: False)],
        )
        with pytest.raises(OutputBlocked, match='Output blocked by guardrail'):
            await agent.run('Hello')

    async def test_async_guard_allows(self) -> None:
        """Async guard returning True should pass the result through."""

        async def safe_check(text: str) -> bool:
            return True

        agent = Agent(
            TestModel(custom_output_text='good output'),
            capabilities=[OutputGuardrail(guard=safe_check)],
        )
        result = await agent.run('Hello')
        assert result.output == 'good output'

    async def test_async_guard_blocks(self) -> None:
        """Async guard returning False should raise OutputBlocked."""

        async def unsafe_check(text: str) -> bool:
            return False

        agent = Agent(
            TestModel(custom_output_text='bad output'),
            capabilities=[OutputGuardrail(guard=unsafe_check)],
        )
        with pytest.raises(OutputBlocked):
            await agent.run('Hello')

    async def test_guard_receives_output_text(self) -> None:
        """Guard function should receive the stringified output."""
        received: list[str] = []

        def capture(text: str) -> bool:
            received.append(text)
            return True

        agent = Agent(
            TestModel(custom_output_text='hello world'),
            capabilities=[OutputGuardrail(guard=capture)],
        )
        await agent.run('test')
        assert received == ['hello world']

    async def test_guard_content_check(self) -> None:
        """Guard should be able to check output content."""

        def no_secrets(text: str) -> bool:
            return 'sk-' not in text

        agent = Agent(
            TestModel(custom_output_text='Your key is sk-abc123'),
            capabilities=[OutputGuardrail(guard=no_secrets)],
        )
        with pytest.raises(OutputBlocked):
            await agent.run('What is my API key?')

    def test_not_serializable(self) -> None:
        """OutputGuardrail should not be spec-serializable (takes a callable)."""
        assert OutputGuardrail.get_serialization_name() is None


# ---------------------------------------------------------------------------
# CostGuard
# ---------------------------------------------------------------------------


class TestCostGuard:
    async def test_no_limits_set(self) -> None:
        """With all limits None, runs should proceed normally."""
        agent = Agent(TestModel(), capabilities=[CostGuard()])
        result = await agent.run('Hello')
        assert result.output is not None

    async def test_high_limits_allow(self) -> None:
        """Limits well above usage should not trigger."""
        agent = Agent(
            TestModel(),
            capabilities=[CostGuard(max_total_tokens=1_000_000)],
        )
        result = await agent.run('Hello')
        assert result.output is not None

    async def test_input_token_limit_exceeded(self) -> None:
        """Exceeding input token limit should raise BudgetExceededError."""
        guard = CostGuard(max_input_tokens=10)
        ctx = _make_run_context(input_tokens=100)
        with pytest.raises(BudgetExceededError, match='Input token budget exceeded'):
            await guard.before_model_request(ctx, _mock_request_context())

    async def test_output_token_limit_exceeded(self) -> None:
        """Exceeding output token limit should raise BudgetExceededError."""
        guard = CostGuard(max_output_tokens=10)
        ctx = _make_run_context(output_tokens=100)
        with pytest.raises(BudgetExceededError, match='Output token budget exceeded'):
            await guard.before_model_request(ctx, _mock_request_context())

    async def test_total_token_limit_exceeded(self) -> None:
        """Exceeding total token limit should raise BudgetExceededError."""
        guard = CostGuard(max_total_tokens=50)
        ctx = _make_run_context(input_tokens=30, output_tokens=30)
        with pytest.raises(BudgetExceededError, match='Total token budget exceeded'):
            await guard.before_model_request(ctx, _mock_request_context())

    async def test_within_limits_passes(self) -> None:
        """Usage within all limits should pass through."""
        guard = CostGuard(max_input_tokens=100, max_output_tokens=100, max_total_tokens=200)
        ctx = _make_run_context(input_tokens=10, output_tokens=10)
        result = await guard.before_model_request(ctx, _mock_request_context())
        assert result is not None

    def test_serialization_name(self) -> None:
        """CostGuard should be spec-serializable."""
        assert CostGuard.get_serialization_name() == 'CostGuard'


# ---------------------------------------------------------------------------
# ToolGuard
# ---------------------------------------------------------------------------


class TestToolGuard:
    async def test_blocked_tools_hidden(self) -> None:
        """Blocked tools should be removed from the tool definitions list."""
        guard = ToolGuard(blocked=['dangerous_tool'])
        ctx = _make_run_context()

        tool_defs = [
            _make_tool_def('safe_tool'),
            _make_tool_def('dangerous_tool'),
            _make_tool_def('another_tool'),
        ]

        result = await guard.prepare_tools(ctx, tool_defs)
        names = [td.name for td in result]
        assert 'dangerous_tool' not in names
        assert 'safe_tool' in names
        assert 'another_tool' in names

    async def test_no_blocked_tools_passes_through(self) -> None:
        """With empty blocked list, all tools should pass through."""
        guard = ToolGuard()
        ctx = _make_run_context()

        tool_defs = [_make_tool_def('tool_a'), _make_tool_def('tool_b')]

        result = await guard.prepare_tools(ctx, tool_defs)
        assert len(result) == 2

    async def test_approval_denied_raises(self) -> None:
        """When approval callback returns False, ToolBlocked should be raised."""
        guard = ToolGuard(
            require_approval=['send_email'],
            approval_callback=lambda name, args: False,
        )
        ctx = _make_run_context()
        call = ToolCallPart(tool_name='send_email', args='{}')
        tool_def = _make_tool_def('send_email')

        with pytest.raises(ToolBlocked, match='approval denied'):
            await guard.before_tool_execute(ctx, call=call, tool_def=tool_def, args={'to': 'user@example.com'})

    async def test_approval_granted_passes(self) -> None:
        """When approval callback returns True, args should pass through."""
        guard = ToolGuard(
            require_approval=['send_email'],
            approval_callback=lambda name, args: True,
        )
        ctx = _make_run_context()
        call = ToolCallPart(tool_name='send_email', args='{}')
        tool_def = _make_tool_def('send_email')

        result = await guard.before_tool_execute(ctx, call=call, tool_def=tool_def, args={'to': 'user@example.com'})
        assert result == {'to': 'user@example.com'}

    async def test_async_approval_callback(self) -> None:
        """Async approval callbacks should work."""

        async def approve(name: str, args: dict[str, Any]) -> bool:
            return True

        guard = ToolGuard(require_approval=['my_tool'], approval_callback=approve)
        ctx = _make_run_context()
        call = ToolCallPart(tool_name='my_tool', args='{}')
        tool_def = _make_tool_def('my_tool')

        result = await guard.before_tool_execute(ctx, call=call, tool_def=tool_def, args={'x': 1})
        assert result == {'x': 1}

    async def test_no_callback_raises(self) -> None:
        """When require_approval is set but no callback provided, ToolBlocked should be raised."""
        guard = ToolGuard(require_approval=['my_tool'])
        ctx = _make_run_context()
        call = ToolCallPart(tool_name='my_tool', args='{}')
        tool_def = _make_tool_def('my_tool')

        with pytest.raises(ToolBlocked, match='no callback configured'):
            await guard.before_tool_execute(ctx, call=call, tool_def=tool_def, args={})

    async def test_unrestricted_tool_passes(self) -> None:
        """Tools not in require_approval should pass through without checking."""
        guard = ToolGuard(require_approval=['restricted'])
        ctx = _make_run_context()
        call = ToolCallPart(tool_name='unrestricted', args='{}')
        tool_def = _make_tool_def('unrestricted')

        result = await guard.before_tool_execute(ctx, call=call, tool_def=tool_def, args={'a': 'b'})
        assert result == {'a': 'b'}

    def test_not_serializable(self) -> None:
        """ToolGuard should not be spec-serializable (takes a callable)."""
        assert ToolGuard.get_serialization_name() is None


# ---------------------------------------------------------------------------
# Integration: multiple guardrails on one agent
# ---------------------------------------------------------------------------


class TestComposition:
    async def test_input_and_output_guardrails_together(self) -> None:
        """Both input and output guardrails should work when combined."""
        agent = Agent(
            TestModel(custom_output_text='safe'),
            capabilities=[
                InputGuardrail(guard=lambda text: True),
                OutputGuardrail(guard=lambda text: True),
            ],
        )
        result = await agent.run('Hello')
        assert result.output == 'safe'

    async def test_input_guardrail_blocks_before_output(self) -> None:
        """If input guardrail blocks, output guardrail should never run."""
        output_called = False

        def output_guard(text: str) -> bool:  # pragma: no cover
            nonlocal output_called
            output_called = True
            return True

        agent = Agent(
            TestModel(custom_output_text='safe'),
            capabilities=[
                InputGuardrail(guard=lambda text: False),
                OutputGuardrail(guard=output_guard),
            ],
        )
        with pytest.raises(InputBlocked):
            await agent.run('Hello')

        assert not output_called


# ---------------------------------------------------------------------------
# Import tests
# ---------------------------------------------------------------------------


class TestImports:
    def test_import_from_package(self) -> None:
        """All public symbols should be importable from the package root."""
        from pydantic_harness import (
            BudgetExceededError,
            CostGuard,
            GuardrailError,
            InputBlocked,
            InputGuardrail,
            OutputBlocked,
            OutputGuardrail,
            ToolBlocked,
            ToolGuard,
        )

        assert InputGuardrail is not None
        assert OutputGuardrail is not None
        assert CostGuard is not None
        assert ToolGuard is not None
        assert GuardrailError is not None
        assert InputBlocked is not None
        assert OutputBlocked is not None
        assert BudgetExceededError is not None
        assert ToolBlocked is not None

    def test_import_from_guardrails_module(self) -> None:
        """All public symbols should be importable from the guardrails module."""
        from pydantic_harness.guardrails import (
            BudgetExceededError,
            CostGuard,
            GuardrailError,
            InputBlocked,
            InputGuardrail,
            OutputBlocked,
            OutputGuardrail,
            ToolBlocked,
            ToolGuard,
        )

        assert InputGuardrail is not None
        assert OutputGuardrail is not None
        assert CostGuard is not None
        assert ToolGuard is not None
        assert GuardrailError is not None
        assert InputBlocked is not None
        assert OutputBlocked is not None
        assert BudgetExceededError is not None
        assert ToolBlocked is not None


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def _make_run_context(
    *,
    input_tokens: int = 0,
    output_tokens: int = 0,
) -> RunContext[None]:
    """Create a minimal RunContext for unit testing hooks directly."""
    return RunContext[None](
        deps=None,
        model=TestModel(),
        usage=RunUsage(input_tokens=input_tokens, output_tokens=output_tokens),
    )


def _make_tool_def(name: str) -> ToolDefinition:
    """Create a minimal ToolDefinition for testing."""
    return ToolDefinition(name=name, description=f'Tool {name}')


class _MockRequestContext:
    """Minimal stand-in for ModelRequestContext in unit tests."""


def _mock_request_context() -> Any:
    """Create a mock request context for CostGuard tests."""
    return _MockRequestContext()
