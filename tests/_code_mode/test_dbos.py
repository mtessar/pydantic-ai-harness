"""DBOS integration tests for CodeMode.

Verifies that the snapshot-based execution loop works inside a DBOS
durable workflow. DBOS uses SQLite locally — no external services needed.

DBOS defaults to `parallel_ordered_events` execution mode, which triggers
the sequential FutureSnapshot resolution path in the execution loop.
"""

from __future__ import annotations

from collections.abc import Generator
from typing import Any

import pytest

try:
    from dbos import DBOS, DBOSConfig
    from pydantic_ai.durable_exec.dbos import DBOSAgent
except ImportError:  # pragma: lax no cover
    pytest.skip('dbos not installed', allow_module_level=True)

from pydantic_ai import Agent
from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.toolsets.function import FunctionToolset

from pydantic_harness import CodeMode

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope='module')
def dbos_instance(tmp_path_factory: pytest.TempPathFactory) -> Generator[DBOS, Any, None]:
    dbos_sqlite_file = tmp_path_factory.mktemp('dbos') / 'dbostest.sqlite'
    dbos_config: DBOSConfig = {
        'name': 'pydantic_harness_dbos_tests',
        'system_database_url': f'sqlite:///{dbos_sqlite_file}',
        'run_admin_server': False,
        'enable_otlp': False,
    }
    dbos = DBOS(config=dbos_config)
    DBOS.launch()
    try:
        yield dbos
    finally:
        DBOS.destroy()


# ---------------------------------------------------------------------------
# Tools and agents (module-level)
# ---------------------------------------------------------------------------


def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


def _code_mode_model(messages: list[ModelRequest | ModelResponse], info: AgentInfo) -> ModelResponse:
    for msg in messages:
        if isinstance(msg, ModelResponse):
            continue
        for part in msg.parts:
            if isinstance(part, ToolReturnPart) and part.tool_name == 'run_code':
                return ModelResponse(parts=[TextPart(content=f'done: {part.content}')])

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
    name='code_mode_dbos_agent',
    toolsets=[FunctionToolset(tools=[add], id='math')],
    capabilities=[CodeMode()],
)

dbos_code_mode_agent = DBOSAgent(code_mode_agent)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_code_mode_runs_in_dbos_workflow(dbos_instance: DBOS) -> None:
    """CodeMode's snapshot-based execution loop works inside a DBOS durable
    workflow. DBOS defaults to `parallel_ordered_events` mode, which triggers
    the sequential FutureSnapshot resolution path."""
    result = dbos_code_mode_agent.run_sync('Calculate 3 + 4')
    assert '7' in str(result.output)
