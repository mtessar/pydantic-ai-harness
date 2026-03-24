"""CodeMode capability — wraps CodeExecutionToolset as an AbstractCapability."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.tools import AgentDepsT
from pydantic_ai.toolsets import AbstractToolset

from pydantic_harness.environments._base import ExecutionEnvironment
from pydantic_harness.toolsets.code_execution import (
    CodeExecutionToolset,
    DescriptionFunc,
    EnvironmentName,
    build_default_description,
)


@dataclass
class CodeMode(AbstractCapability[AgentDepsT]):
    """Capability that provides code execution via CodeExecutionToolset.

    Wraps an ExecutionEnvironment (or environment name like 'monty') and an optional
    toolset of tools to expose as callable Python functions within the code sandbox.

    Usage:
        ```python {test="skip" lint="skip"}
        from pydantic_ai import Agent
        from pydantic_harness import CodeMode

        agent = Agent('anthropic:claude-sonnet-4-5', capabilities=[CodeMode()])
        ```
    """

    environment: ExecutionEnvironment | EnvironmentName = 'monty'
    """The code execution environment. Can be an instance or a string shorthand ('monty')."""

    toolset: AbstractToolset[AgentDepsT] | None = None
    """Optional toolset to wrap. Its tools become callable Python functions in the sandbox."""

    description: str | DescriptionFunc = field(default=build_default_description)
    """Custom tool description. String or callback for full control."""

    max_retries: int = 3
    """Maximum retries for code execution errors."""

    @classmethod
    def get_serialization_name(cls) -> str | None:
        return 'CodeMode'

    def get_toolset(self) -> AbstractToolset[Any] | None:
        return CodeExecutionToolset(
            environment=self.environment,
            toolset=self.toolset,
            description=self.description,
            max_retries=self.max_retries,
        )
