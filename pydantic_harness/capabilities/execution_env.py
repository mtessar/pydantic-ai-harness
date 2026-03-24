"""ExecutionEnv capability — wraps ExecutionEnvironmentToolset as an AbstractCapability."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.tools import AgentDepsT
from pydantic_ai.toolsets import AbstractToolset

from pydantic_harness.environments._base import ExecutionEnvironment
from pydantic_harness.toolsets.execution_environment import (
    Capability,
    CodeLanguage,
    EditStrategy,
    ExecutionEnvironmentToolset,
)


@dataclass
class ExecutionEnv(AbstractCapability[AgentDepsT]):
    """Capability that provides coding-agent-style tools backed by an ExecutionEnvironment.

    Exposes ls, shell, read_file, write_file, edit, glob, grep tools.

    Usage:
        ```python {test="skip" lint="skip"}
        from pydantic_ai import Agent
        from pydantic_harness import ExecutionEnv
        from pydantic_harness.environments.local import LocalEnvironment

        agent = Agent(
            'anthropic:claude-sonnet-4-5',
            capabilities=[ExecutionEnv(environment=LocalEnvironment(root_dir='/tmp/workspace'))],
        )
        ```
    """

    environment: ExecutionEnvironment
    """The execution environment backing the tools."""

    include: frozenset[Capability] | None = None
    """Capabilities to include. None = all (minus default excludes)."""

    exclude: frozenset[Capability] | None = None
    """Capabilities to exclude. None = default excludes (run_code)."""

    edit_strategy: EditStrategy | None = None
    """Edit tool strategy. None = auto-select."""

    code_language: CodeLanguage | None = None
    """Code execution language. None = auto-detect."""

    require_shell_approval: bool = False
    """Whether shell tool requires human approval."""

    require_write_approval: bool = False
    """Whether write/edit tools require human approval."""

    image_support: bool = True
    """Whether read_file returns images as BinaryContent."""

    max_image_bytes: int = 50 * 1024 * 1024
    """Maximum image file size in bytes to return as BinaryContent."""

    max_retries: int = 1
    """Maximum retries per tool call."""

    @classmethod
    def get_serialization_name(cls) -> str | None:
        return 'ExecutionEnv'

    def get_toolset(self) -> AbstractToolset[Any] | None:
        return ExecutionEnvironmentToolset(
            environment=self.environment,
            include=self.include,
            exclude=self.exclude,
            edit_strategy=self.edit_strategy,
            code_language=self.code_language,
            require_shell_approval=self.require_shell_approval,
            require_write_approval=self.require_write_approval,
            image_support=self.image_support,
            max_image_bytes=self.max_image_bytes,
            max_retries=self.max_retries,
        )
