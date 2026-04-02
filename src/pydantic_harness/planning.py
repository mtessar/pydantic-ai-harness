"""Planning capability for structured task planning and tracking.

Provides tools for creating and managing a step-by-step plan during agent runs,
with dynamic system prompt injection of current plan state.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from pydantic_ai._instructions import AgentInstructions
from pydantic_ai.capabilities.abstract import AbstractCapability
from pydantic_ai.tools import AgentDepsT, RunContext
from pydantic_ai.toolsets import AgentToolset
from pydantic_ai.toolsets.function import FunctionToolset


class TaskStatus(str, Enum):
    """Status of a task in the plan."""

    pending = 'pending'
    in_progress = 'in_progress'
    completed = 'completed'
    skipped = 'skipped'
    blocked = 'blocked'


@dataclass
class Task:
    """A single task in the plan."""

    description: str
    status: TaskStatus = TaskStatus.pending
    parent_index: int | None = None


def format_plan(tasks: list[Task]) -> str:
    """Format the current plan as a readable string.

    Subtasks (those with a ``parent_index``) are indented under their parent.

    Args:
        tasks: The list of tasks to format.

    Returns:
        A human-readable string representation of the plan.
    """
    if not tasks:
        return 'No plan created yet.'

    status_icons = {
        TaskStatus.pending: '[ ]',
        TaskStatus.in_progress: '[~]',
        TaskStatus.completed: '[x]',
        TaskStatus.skipped: '[-]',
        TaskStatus.blocked: '[!]',
    }

    lines: list[str] = []
    for i, task in enumerate(tasks):
        icon = status_icons[task.status]
        indent = '  ' if task.parent_index is not None else ''
        lines.append(f'{indent}{i}. {icon} {task.description}')

    total = len(tasks)
    completed = sum(1 for t in tasks if t.status == TaskStatus.completed)
    skipped = sum(1 for t in tasks if t.status == TaskStatus.skipped)
    in_progress = sum(1 for t in tasks if t.status == TaskStatus.in_progress)
    pending = sum(1 for t in tasks if t.status == TaskStatus.pending)
    blocked = sum(1 for t in tasks if t.status == TaskStatus.blocked)

    lines.append('')
    lines.append(
        f'Progress: {completed}/{total} completed, {in_progress} in progress, '
        f'{pending} pending, {skipped} skipped, {blocked} blocked'
    )

    return '\n'.join(lines)


def create_plan_impl(tasks: list[Task], steps: list[str]) -> str:
    """Create a new plan, replacing any existing one.

    Args:
        tasks: The shared task list to modify.
        steps: Step descriptions for the new plan.

    Returns:
        A confirmation message with the formatted plan.
    """
    tasks.clear()
    tasks.extend(Task(description=step) for step in steps)
    return f'Plan created with {len(tasks)} steps.\n\n{format_plan(tasks)}'


def update_task_impl(tasks: list[Task], index: int, status: TaskStatus) -> str:
    """Update the status of a task.

    Args:
        tasks: The shared task list to modify.
        index: Zero-based index of the task to update.
        status: The new status.

    Returns:
        A confirmation message or an error description.
    """
    if not tasks:
        return 'No plan exists. Use create_plan first.'
    if index < 0 or index >= len(tasks):
        return f'Invalid task index {index}. Valid range: 0-{len(tasks) - 1}.'
    tasks[index].status = status
    return f'Task {index} updated to {status.value}.\n\n{format_plan(tasks)}'


def add_subtask_impl(tasks: list[Task], parent_index: int, description: str) -> str:
    """Add a subtask under an existing parent task.

    The subtask is inserted immediately after the parent and any existing
    subtasks of that parent.

    Args:
        tasks: The shared task list to modify.
        parent_index: Zero-based index of the parent task.
        description: Description of the subtask.

    Returns:
        A confirmation message or an error description.
    """
    if not tasks:
        return 'No plan exists. Use create_plan first.'
    if parent_index < 0 or parent_index >= len(tasks):
        return f'Invalid parent index {parent_index}. Valid range: 0-{len(tasks) - 1}.'
    if tasks[parent_index].parent_index is not None:
        return f'Task {parent_index} is itself a subtask. Nested subtasks are not supported.'

    # Find insertion point: after parent and its existing subtasks.
    insert_at = parent_index + 1
    while insert_at < len(tasks) and tasks[insert_at].parent_index == parent_index:
        insert_at += 1

    tasks.insert(insert_at, Task(description=description, parent_index=parent_index))

    # Adjust parent_index references for tasks shifted by the insertion.
    for task in tasks[insert_at + 1 :]:
        if task.parent_index is not None and task.parent_index >= insert_at:
            task.parent_index += 1

    return f'Subtask added under task {parent_index} at index {insert_at}.\n\n{format_plan(tasks)}'


def insert_task_impl(tasks: list[Task], index: int, description: str) -> str:
    """Insert a new top-level task at a given position.

    Args:
        tasks: The shared task list to modify.
        index: Zero-based position to insert at (clamped to list bounds).
        description: Description of the new task.

    Returns:
        A confirmation message with the formatted plan.
    """
    clamped = max(0, min(index, len(tasks)))
    tasks.insert(clamped, Task(description=description))

    # Adjust parent_index references for tasks shifted by the insertion.
    for task in tasks[clamped + 1 :]:
        if task.parent_index is not None and task.parent_index >= clamped:
            task.parent_index += 1

    return f'Task inserted at index {clamped}.\n\n{format_plan(tasks)}'


def remove_task_impl(tasks: list[Task], index: int) -> str:
    """Remove a task by index.

    If the removed task is a parent, its subtasks are also removed.

    Args:
        tasks: The shared task list to modify.
        index: Zero-based index of the task to remove.

    Returns:
        A confirmation message or an error description.
    """
    if not tasks:
        return 'No plan exists. Use create_plan first.'
    if index < 0 or index >= len(tasks):
        return f'Invalid task index {index}. Valid range: 0-{len(tasks) - 1}.'

    removed = tasks[index]
    # Collect indices to remove: the task itself, plus any subtasks if it's a parent.
    indices_to_remove = {index}
    if removed.parent_index is None:
        # It's a top-level task; remove its subtasks too.
        for i, task in enumerate(tasks):
            if task.parent_index == index:
                indices_to_remove.add(i)

    # Remove in reverse order to keep indices stable.
    for i in sorted(indices_to_remove, reverse=True):
        tasks.pop(i)

    # Adjust parent_index references: for each removed index (ascending),
    # decrement references that pointed past it.
    for removed_idx in sorted(indices_to_remove):
        for task in tasks:
            if task.parent_index is not None and task.parent_index > removed_idx:
                task.parent_index -= 1

    count = len(indices_to_remove)
    suffix = f' (and {count - 1} subtask{"s" if count > 2 else ""})' if count > 1 else ''
    return f'Task {index} removed{suffix}.\n\n{format_plan(tasks)}'


def get_plan_impl(tasks: list[Task]) -> str:
    """Get the current plan.

    Args:
        tasks: The task list to format.

    Returns:
        The formatted plan.
    """
    return format_plan(tasks)


@dataclass
class Planning(AbstractCapability[AgentDepsT]):
    """Structured task planning and tracking capability.

    Provides tools for the agent to create a step-by-step plan, update task
    statuses as work progresses, and review the current plan. The current plan
    state is dynamically injected into the system prompt so the model always
    has context on what has been done and what remains.

    Example:
        ```python
        from pydantic_ai import Agent
        from pydantic_harness.planning import Planning

        agent = Agent('openai:gpt-4o', capabilities=[Planning()])
        ```
    """

    plan_tasks: list[Task] = field(default_factory=lambda: list[Task]())
    """Per-run task list. Populated via ``for_run()``."""

    async def for_run(self, ctx: RunContext[AgentDepsT]) -> Planning[AgentDepsT]:
        """Return a fresh instance with isolated per-run state."""
        return Planning[AgentDepsT]()

    def get_instructions(self) -> AgentInstructions[AgentDepsT]:
        """Return a dynamic instruction that injects the current plan state."""
        tasks = self.plan_tasks

        def _instructions(ctx: RunContext[AgentDepsT]) -> str:
            plan_text = format_plan(tasks)
            return (
                'You have a planning capability. Use the planning tools to break complex tasks '
                'into steps and track your progress. Before starting work, create a plan. '
                'Update task statuses as you make progress.\n\n'
                f'Current plan:\n{plan_text}'
            )

        return _instructions

    def get_toolset(self) -> AgentToolset[AgentDepsT] | None:
        """Return a toolset with plan management tools."""
        tasks = self.plan_tasks
        toolset: FunctionToolset[AgentDepsT] = FunctionToolset()

        @toolset.tool
        def create_plan(ctx: RunContext[AgentDepsT], steps: list[str]) -> str:  # pyright: ignore[reportUnusedFunction]
            """Create a new plan with the given steps. Replaces any existing plan.

            Args:
                ctx: The run context.
                steps: A list of step descriptions for the plan.
            """
            return create_plan_impl(tasks, steps)

        @toolset.tool
        def update_task(ctx: RunContext[AgentDepsT], index: int, status: TaskStatus) -> str:  # pyright: ignore[reportUnusedFunction]
            """Update the status of a task in the plan.

            Args:
                ctx: The run context.
                index: The zero-based index of the task to update.
                status: The new status for the task.
            """
            return update_task_impl(tasks, index, status)

        @toolset.tool
        def add_subtask(ctx: RunContext[AgentDepsT], parent_index: int, description: str) -> str:  # pyright: ignore[reportUnusedFunction]
            """Add a subtask under an existing parent task.

            Args:
                ctx: The run context.
                parent_index: The zero-based index of the parent task.
                description: Description of the subtask.
            """
            return add_subtask_impl(tasks, parent_index, description)

        @toolset.tool
        def insert_task(ctx: RunContext[AgentDepsT], index: int, description: str) -> str:  # pyright: ignore[reportUnusedFunction]
            """Insert a new task at a given position in the plan.

            Args:
                ctx: The run context.
                index: The zero-based position to insert at.
                description: Description of the new task.
            """
            return insert_task_impl(tasks, index, description)

        @toolset.tool
        def remove_task(ctx: RunContext[AgentDepsT], index: int) -> str:  # pyright: ignore[reportUnusedFunction]
            """Remove a task from the plan by index.

            If the task has subtasks, they are also removed.

            Args:
                ctx: The run context.
                index: The zero-based index of the task to remove.
            """
            return remove_task_impl(tasks, index)

        @toolset.tool
        def get_plan(ctx: RunContext[AgentDepsT]) -> str:  # pyright: ignore[reportUnusedFunction]
            """Get the current plan with all task statuses.

            Args:
                ctx: The run context.
            """
            return get_plan_impl(tasks)

        return toolset

    @classmethod
    def get_serialization_name(cls) -> str | None:
        """Return the serialization name for spec support."""
        return 'Planning'

    @property
    def tasks(self) -> list[Task]:
        """Read-only access to the current task list."""
        return list(self.plan_tasks)
