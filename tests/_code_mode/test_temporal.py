"""Temporal integration tests for CodeMode.

Verifies that the snapshot-based execution loop (`feed_start`/`resume`)
works inside a Temporal workflow sandbox, which forbids threads and
`call_soon_threadsafe`.

These tests start a local Temporal dev server via
`WorkflowEnvironment.start_local()` — the Temporal SDK downloads and
runs `temporalite` automatically.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import timedelta

import pytest

try:
    from pydantic_ai.durable_exec.temporal import (
        AgentPlugin,
        PydanticAIPlugin,
        TemporalAgent,
    )
    from temporalio import workflow
    from temporalio.client import Client
    from temporalio.common import RetryPolicy
    from temporalio.testing import WorkflowEnvironment
    from temporalio.worker import Worker
    from temporalio.workflow import ActivityConfig
except ImportError:  # pragma: lax no cover
    pytest.skip('temporalio not installed', allow_module_level=True)

from pydantic_ai import Agent
from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.toolsets.function import FunctionToolset

from pydantic_harness import CodeMode

pytestmark = pytest.mark.anyio

TEMPORAL_PORT = 7244  # avoid conflict with other test suites
TASK_QUEUE = 'pydantic-harness-code-mode-queue'
BASE_ACTIVITY_CONFIG = ActivityConfig(
    start_to_close_timeout=timedelta(seconds=60),
    retry_policy=RetryPolicy(maximum_attempts=1),
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope='module')
async def temporal_env() -> AsyncIterator[WorkflowEnvironment]:
    async with await WorkflowEnvironment.start_local(  # pyright: ignore[reportUnknownMemberType]
        port=TEMPORAL_PORT,
        dev_server_extra_args=[
            '--dynamic-config-value',
            'frontend.enableServerVersionCheck=false',
        ],
    ) as env:
        yield env


@pytest.fixture
async def client(temporal_env: WorkflowEnvironment) -> Client:
    return await Client.connect(
        f'localhost:{TEMPORAL_PORT}',
        plugins=[PydanticAIPlugin()],
    )


# ---------------------------------------------------------------------------
# Tools and agents (module-level — Temporal requirement)
# ---------------------------------------------------------------------------


def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


# FunctionModel that emits a run_code tool call for the given code snippet.
def _code_mode_model(messages: list[ModelRequest | ModelResponse], info: AgentInfo) -> ModelResponse:
    """Model that generates a run_code call on the first request, then returns the result as text."""
    # Check if we already got a tool result back.
    for msg in messages:
        if isinstance(msg, ModelResponse):
            continue
        for part in msg.parts:
            if isinstance(part, ToolReturnPart) and part.tool_name == 'run_code':
                return ModelResponse(parts=[TextPart(content=f'done: {part.content}')])

    # First call — emit run_code.
    return ModelResponse(
        parts=[
            ToolCallPart(
                tool_name='run_code',
                args={'code': 'result = await add(a=3, b=4)\nresult'},
                tool_call_id='test_tc_1',
            )
        ]
    )


code_mode_agent = Agent(
    FunctionModel(_code_mode_model),
    name='code_mode_temporal_agent',
    toolsets=[FunctionToolset(tools=[add], id='math')],
    capabilities=[CodeMode()],
)

temporal_code_mode_agent = TemporalAgent(
    code_mode_agent,
    activity_config=BASE_ACTIVITY_CONFIG,
)


@workflow.defn
class CodeModeWorkflow:
    @workflow.run
    async def run(self, prompt: str) -> str:
        result = await temporal_code_mode_agent.run(prompt)
        return str(result.output)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


async def test_code_mode_runs_in_temporal_workflow(client: Client) -> None:
    """CodeMode's snapshot-based execution loop works inside a Temporal workflow.

    This is the core regression test for the `call_soon_threadsafe` issue:
    the old `feed_run_async` approach hung because Temporal's sandboxed
    event loop doesn't implement `call_soon_threadsafe`. The snapshot
    approach (`feed_start`/`resume`) avoids threads entirely.
    """
    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[CodeModeWorkflow],
        plugins=[AgentPlugin(temporal_code_mode_agent)],
    ):
        output = await client.execute_workflow(
            CodeModeWorkflow.run,
            args=['Calculate 3 + 4'],
            id='test_code_mode_temporal_1',
            task_queue=TASK_QUEUE,
        )
        assert '7' in output
