"""Tests for ToolOutputManagement capability."""

from __future__ import annotations

import pytest
from pydantic_ai.messages import ToolCallPart
from pydantic_ai.tools import ToolDefinition

from pydantic_harness.tool_output_management import (
    ToolOutputManagement,
    TruncationStrategy,
)

# Re-export private helpers for unit testing; pyright: ignore for private usage.
from pydantic_harness.tool_output_management import (
    _head_tail_default_split as head_tail_default_split,  # pyright: ignore[reportPrivateUsage]
)
from pydantic_harness.tool_output_management import (
    _stringify as stringify,  # pyright: ignore[reportPrivateUsage]
)
from pydantic_harness.tool_output_management import (
    _truncate as truncate,  # pyright: ignore[reportPrivateUsage]
)

CALL = ToolCallPart(tool_name='my_tool', args={})
TOOL_DEF = ToolDefinition(name='my_tool', description='test tool', parameters_json_schema={})


# ---------------------------------------------------------------------------
# Unit tests: stringify
# ---------------------------------------------------------------------------


class TestStringify:
    def test_string_passthrough(self) -> None:
        assert stringify('hello') == 'hello'

    def test_int(self) -> None:
        assert stringify(42) == '42'

    def test_dict(self) -> None:
        result = stringify({'key': 'value'})
        assert 'key' in result
        assert 'value' in result

    def test_list(self) -> None:
        assert stringify([1, 2, 3]) == '[1, 2, 3]'

    def test_none(self) -> None:
        assert stringify(None) == 'None'


# ---------------------------------------------------------------------------
# Unit tests: head_tail_default_split
# ---------------------------------------------------------------------------


class TestHeadTailSplit:
    def test_split_100(self) -> None:
        head, tail = head_tail_default_split(100)
        assert head == 60
        assert tail == 40

    def test_split_sums_to_limit(self) -> None:
        for limit in (1, 10, 77, 1000, 9999):
            head, tail = head_tail_default_split(limit)
            assert head + tail == limit


# ---------------------------------------------------------------------------
# Unit tests: truncate
# ---------------------------------------------------------------------------


class TestTruncate:
    def test_no_truncation_needed(self) -> None:
        text = 'short'
        assert truncate(text, 100, TruncationStrategy.head) == text

    def test_head_strategy(self) -> None:
        text = 'a' * 200
        result = truncate(text, 50, TruncationStrategy.head)
        assert result.startswith('a' * 50)
        assert '[Truncated: showing first 50 of 200 chars]' in result

    def test_tail_strategy(self) -> None:
        text = 'a' * 200
        result = truncate(text, 50, TruncationStrategy.tail)
        assert result.endswith('a' * 50)
        assert '[Truncated: showing last 50 of 200 chars]' in result

    def test_head_tail_strategy(self) -> None:
        text = 'H' * 100 + 'M' * 800 + 'T' * 100
        result = truncate(text, 100, TruncationStrategy.head_tail)
        # head=60 chars, tail=40 chars
        assert result.startswith('H' * 60)
        assert result.endswith('T' * 40)
        assert 'omitted from middle' in result
        assert '900' in result  # 1000 - 60 - 40 = 900 omitted

    def test_head_tail_exact_boundary(self) -> None:
        text = 'x' * 100
        # Exactly at limit -> no truncation
        assert truncate(text, 100, TruncationStrategy.head_tail) == text

    def test_comma_formatting(self) -> None:
        text = 'a' * 100_000
        result = truncate(text, 1000, TruncationStrategy.head)
        assert '100,000' in result


# ---------------------------------------------------------------------------
# Integration tests: ToolOutputManagement.after_tool_execute
# ---------------------------------------------------------------------------


class TestAfterToolExecute:
    @pytest.mark.anyio
    async def test_short_output_unchanged(self) -> None:
        cap: ToolOutputManagement[None] = ToolOutputManagement(max_output_chars=100)
        result = await cap.after_tool_execute(
            None,  # type: ignore[arg-type]
            call=CALL,
            tool_def=TOOL_DEF,
            args={},
            result='short output',
        )
        assert result == 'short output'

    @pytest.mark.anyio
    async def test_long_string_truncated(self) -> None:
        cap: ToolOutputManagement[None] = ToolOutputManagement(max_output_chars=50)
        long_text = 'x' * 200
        result = await cap.after_tool_execute(
            None,  # type: ignore[arg-type]
            call=CALL,
            tool_def=TOOL_DEF,
            args={},
            result=long_text,
        )
        assert isinstance(result, str)
        assert len(result) < len(long_text)
        assert 'Truncated' in result

    @pytest.mark.anyio
    async def test_non_string_result_truncated(self) -> None:
        cap: ToolOutputManagement[None] = ToolOutputManagement(max_output_chars=20)
        big_dict = {'key': 'v' * 200}
        result = await cap.after_tool_execute(
            None,  # type: ignore[arg-type]
            call=CALL,
            tool_def=TOOL_DEF,
            args={},
            result=big_dict,
        )
        assert isinstance(result, str)
        assert 'Truncated' in result

    @pytest.mark.anyio
    async def test_per_tool_limit(self) -> None:
        cap: ToolOutputManagement[None] = ToolOutputManagement(
            max_output_chars=10_000,
            per_tool_limits={'special_tool': 20},
        )
        call = ToolCallPart(tool_name='special_tool', args={})
        result = await cap.after_tool_execute(
            None,  # type: ignore[arg-type]
            call=call,
            tool_def=TOOL_DEF,
            args={},
            result='a' * 100,
        )
        assert 'Truncated' in result

    @pytest.mark.anyio
    async def test_per_tool_limit_does_not_affect_others(self) -> None:
        cap: ToolOutputManagement[None] = ToolOutputManagement(
            max_output_chars=10_000,
            per_tool_limits={'other_tool': 5},
        )
        result = await cap.after_tool_execute(
            None,  # type: ignore[arg-type]
            call=CALL,
            tool_def=TOOL_DEF,
            args={},
            result='a' * 100,  # under 10_000
        )
        assert result == 'a' * 100

    @pytest.mark.anyio
    async def test_per_tool_strategy(self) -> None:
        cap: ToolOutputManagement[None] = ToolOutputManagement(
            max_output_chars=50,
            per_tool_strategies={'tail_tool': TruncationStrategy.tail},
        )
        call = ToolCallPart(tool_name='tail_tool', args={})
        result = await cap.after_tool_execute(
            None,  # type: ignore[arg-type]
            call=call,
            tool_def=TOOL_DEF,
            args={},
            result='a' * 200,
        )
        assert 'showing last' in result

    @pytest.mark.anyio
    async def test_head_strategy_via_config(self) -> None:
        cap: ToolOutputManagement[None] = ToolOutputManagement(
            max_output_chars=50,
            strategy=TruncationStrategy.head,
        )
        result = await cap.after_tool_execute(
            None,  # type: ignore[arg-type]
            call=CALL,
            tool_def=TOOL_DEF,
            args={},
            result='a' * 200,
        )
        assert 'showing first' in result

    @pytest.mark.anyio
    async def test_sync_summarize_fn(self) -> None:
        def summarize(tool_name: str, output: str) -> str:
            return f'Summary of {tool_name}: {len(output)} chars'

        cap: ToolOutputManagement[None] = ToolOutputManagement(
            max_output_chars=100,
            summarize_fn=summarize,
        )
        result = await cap.after_tool_execute(
            None,  # type: ignore[arg-type]
            call=CALL,
            tool_def=TOOL_DEF,
            args={},
            result='x' * 200,
        )
        assert result == 'Summary of my_tool: 200 chars'

    @pytest.mark.anyio
    async def test_async_summarize_fn(self) -> None:
        async def summarize(tool_name: str, output: str) -> str:
            return f'Async summary: {len(output)} chars'

        cap: ToolOutputManagement[None] = ToolOutputManagement(
            max_output_chars=100,
            summarize_fn=summarize,
        )
        result = await cap.after_tool_execute(
            None,  # type: ignore[arg-type]
            call=CALL,
            tool_def=TOOL_DEF,
            args={},
            result='x' * 200,
        )
        assert result == 'Async summary: 200 chars'

    @pytest.mark.anyio
    async def test_summarize_fn_safety_net(self) -> None:
        """If summarize_fn returns something still too long, truncation kicks in."""

        def bad_summarize(tool_name: str, output: str) -> str:
            return output  # returns full output, still too long

        cap: ToolOutputManagement[None] = ToolOutputManagement(
            max_output_chars=50,
            summarize_fn=bad_summarize,
        )
        result = await cap.after_tool_execute(
            None,  # type: ignore[arg-type]
            call=CALL,
            tool_def=TOOL_DEF,
            args={},
            result='x' * 200,
        )
        assert 'Truncated' in result

    @pytest.mark.anyio
    async def test_summarize_fn_not_called_under_limit(self) -> None:
        calls: list[str] = []

        def summarize(tool_name: str, output: str) -> str:
            calls.append(tool_name)
            return 'summarized'

        cap: ToolOutputManagement[None] = ToolOutputManagement(
            max_output_chars=100,
            summarize_fn=summarize,
        )
        result = await cap.after_tool_execute(
            None,  # type: ignore[arg-type]
            call=CALL,
            tool_def=TOOL_DEF,
            args={},
            result='short',
        )
        assert result == 'short'
        assert calls == []

    @pytest.mark.anyio
    async def test_original_returned_when_exactly_at_limit(self) -> None:
        cap: ToolOutputManagement[None] = ToolOutputManagement(max_output_chars=10)
        result = await cap.after_tool_execute(
            None,  # type: ignore[arg-type]
            call=CALL,
            tool_def=TOOL_DEF,
            args={},
            result='a' * 10,
        )
        assert result == 'a' * 10


# ---------------------------------------------------------------------------
# Test public exports
# ---------------------------------------------------------------------------


class TestExports:
    def test_import_from_package(self) -> None:
        from pydantic_harness import ToolOutputManagement, TruncationStrategy

        assert ToolOutputManagement is not None
        assert TruncationStrategy is not None
