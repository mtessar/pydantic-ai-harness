"""Tests for the Planning capability."""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

from pydantic_harness.planning import (
    Planning,
    Task,
    TaskStatus,
    add_subtask_impl,
    create_plan_impl,
    format_plan,
    get_plan_impl,
    insert_task_impl,
    remove_task_impl,
    update_task_impl,
)


class TestTaskStatus:
    def test_values(self):
        assert TaskStatus.pending == 'pending'
        assert TaskStatus.in_progress == 'in_progress'
        assert TaskStatus.completed == 'completed'
        assert TaskStatus.skipped == 'skipped'
        assert TaskStatus.blocked == 'blocked'

    def test_all_statuses(self):
        assert set(TaskStatus) == {
            TaskStatus.pending,
            TaskStatus.in_progress,
            TaskStatus.completed,
            TaskStatus.skipped,
            TaskStatus.blocked,
        }


class TestTask:
    def test_default_status(self):
        task = Task(description='Do something')
        assert task.description == 'Do something'
        assert task.status == TaskStatus.pending
        assert task.parent_index is None

    def test_explicit_status(self):
        task = Task(description='Done thing', status=TaskStatus.completed)
        assert task.status == TaskStatus.completed

    def test_explicit_parent_index(self):
        task = Task(description='Sub', parent_index=0)
        assert task.parent_index == 0


class TestFormatPlan:
    def test_empty(self):
        assert format_plan([]) == 'No plan created yet.'

    def test_single_pending(self):
        result = format_plan([Task(description='Step one')])
        assert '0. [ ] Step one' in result
        assert 'Progress: 0/1 completed' in result

    def test_mixed_statuses(self):
        tasks = [
            Task(description='First', status=TaskStatus.completed),
            Task(description='Second', status=TaskStatus.in_progress),
            Task(description='Third', status=TaskStatus.pending),
            Task(description='Fourth', status=TaskStatus.skipped),
        ]
        result = format_plan(tasks)
        assert '0. [x] First' in result
        assert '1. [~] Second' in result
        assert '2. [ ] Third' in result
        assert '3. [-] Fourth' in result
        assert '1 in progress, 1 pending, 1 skipped, 0 blocked' in result

    def test_all_completed(self):
        tasks = [
            Task(description='A', status=TaskStatus.completed),
            Task(description='B', status=TaskStatus.completed),
        ]
        result = format_plan(tasks)
        assert 'Progress: 2/2 completed, 0 in progress, 0 pending, 0 skipped, 0 blocked' in result

    def test_blocked_status(self):
        tasks = [
            Task(description='Setup', status=TaskStatus.completed),
            Task(description='Blocked step', status=TaskStatus.blocked),
        ]
        result = format_plan(tasks)
        assert '1. [!] Blocked step' in result
        assert '1 blocked' in result

    def test_subtask_indented(self):
        tasks = [
            Task(description='Parent'),
            Task(description='Child', parent_index=0),
        ]
        result = format_plan(tasks)
        assert '0. [ ] Parent' in result
        assert '  1. [ ] Child' in result


class TestCreatePlanImpl:
    def test_creates_tasks(self):
        tasks: list[Task] = []
        result = create_plan_impl(tasks, ['Step A', 'Step B', 'Step C'])
        assert len(tasks) == 3
        assert tasks[0].description == 'Step A'
        assert tasks[1].description == 'Step B'
        assert tasks[2].description == 'Step C'
        assert all(t.status == TaskStatus.pending for t in tasks)
        assert 'Plan created with 3 steps' in result

    def test_replaces_existing(self):
        tasks = [Task(description='Old', status=TaskStatus.completed)]
        result = create_plan_impl(tasks, ['New'])
        assert len(tasks) == 1
        assert tasks[0].description == 'New'
        assert tasks[0].status == TaskStatus.pending
        assert 'Plan created with 1 steps' in result

    def test_empty_steps(self):
        tasks: list[Task] = []
        result = create_plan_impl(tasks, [])
        assert len(tasks) == 0
        assert 'Plan created with 0 steps' in result


class TestUpdateTaskImpl:
    def test_update_status(self):
        tasks = [Task(description='Do X'), Task(description='Do Y')]
        result = update_task_impl(tasks, 0, TaskStatus.in_progress)
        assert tasks[0].status == TaskStatus.in_progress
        assert 'Task 0 updated to in_progress' in result

    def test_update_to_completed(self):
        tasks = [Task(description='Do X')]
        result = update_task_impl(tasks, 0, TaskStatus.completed)
        assert tasks[0].status == TaskStatus.completed
        assert 'Task 0 updated to completed' in result

    def test_update_to_skipped(self):
        tasks = [Task(description='Do X')]
        result = update_task_impl(tasks, 0, TaskStatus.skipped)
        assert tasks[0].status == TaskStatus.skipped
        assert 'Task 0 updated to skipped' in result

    def test_no_plan_exists(self):
        tasks: list[Task] = []
        result = update_task_impl(tasks, 0, TaskStatus.completed)
        assert result == 'No plan exists. Use create_plan first.'

    def test_index_too_high(self):
        tasks = [Task(description='Only one')]
        result = update_task_impl(tasks, 5, TaskStatus.completed)
        assert 'Invalid task index 5' in result
        assert 'Valid range: 0-0' in result

    def test_negative_index(self):
        tasks = [Task(description='Only one')]
        result = update_task_impl(tasks, -1, TaskStatus.completed)
        assert 'Invalid task index -1' in result


class TestGetPlanImpl:
    def test_empty(self):
        assert get_plan_impl([]) == 'No plan created yet.'

    def test_with_tasks(self):
        tasks = [Task(description='A', status=TaskStatus.completed)]
        result = get_plan_impl(tasks)
        assert '0. [x] A' in result


class TestAddSubtaskImpl:
    def test_add_subtask(self):
        tasks = [Task(description='Parent'), Task(description='Other')]
        result = add_subtask_impl(tasks, 0, 'Child')
        assert len(tasks) == 3
        assert tasks[1].description == 'Child'
        assert tasks[1].parent_index == 0
        assert 'Subtask added under task 0 at index 1' in result

    def test_add_multiple_subtasks(self):
        tasks = [Task(description='Parent'), Task(description='Other')]
        add_subtask_impl(tasks, 0, 'Child A')
        add_subtask_impl(tasks, 0, 'Child B')
        assert len(tasks) == 4
        assert tasks[1].description == 'Child A'
        assert tasks[1].parent_index == 0
        assert tasks[2].description == 'Child B'
        assert tasks[2].parent_index == 0
        assert tasks[3].description == 'Other'

    def test_no_plan(self):
        result = add_subtask_impl([], 0, 'Child')
        assert result == 'No plan exists. Use create_plan first.'

    def test_invalid_parent_index(self):
        tasks = [Task(description='Only')]
        result = add_subtask_impl(tasks, 5, 'Child')
        assert 'Invalid parent index 5' in result

    def test_negative_parent_index(self):
        tasks = [Task(description='Only')]
        result = add_subtask_impl(tasks, -1, 'Child')
        assert 'Invalid parent index -1' in result

    def test_nested_subtask_rejected(self):
        tasks = [Task(description='Parent'), Task(description='Child', parent_index=0)]
        result = add_subtask_impl(tasks, 1, 'Grandchild')
        assert 'is itself a subtask' in result
        assert len(tasks) == 2

    def test_parent_index_adjustment(self):
        """Subtasks of later parents get their parent_index adjusted on insertion."""
        tasks = [
            Task(description='A'),
            Task(description='B'),
            Task(description='B-child', parent_index=1),
        ]
        # Add subtask under A; B-child's parent_index should shift from 1 to 2.
        add_subtask_impl(tasks, 0, 'A-child')
        assert tasks[3].description == 'B-child'
        assert tasks[3].parent_index == 2


class TestInsertTaskImpl:
    def test_insert_at_beginning(self):
        tasks = [Task(description='Existing')]
        result = insert_task_impl(tasks, 0, 'New first')
        assert len(tasks) == 2
        assert tasks[0].description == 'New first'
        assert tasks[1].description == 'Existing'
        assert 'Task inserted at index 0' in result

    def test_insert_at_end(self):
        tasks = [Task(description='Existing')]
        result = insert_task_impl(tasks, 1, 'New last')
        assert len(tasks) == 2
        assert tasks[1].description == 'New last'
        assert 'Task inserted at index 1' in result

    def test_insert_in_middle(self):
        tasks = [Task(description='A'), Task(description='C')]
        insert_task_impl(tasks, 1, 'B')
        assert [t.description for t in tasks] == ['A', 'B', 'C']

    def test_index_clamped_high(self):
        tasks = [Task(description='Only')]
        result = insert_task_impl(tasks, 100, 'Clamped')
        assert len(tasks) == 2
        assert tasks[1].description == 'Clamped'
        assert 'Task inserted at index 1' in result

    def test_index_clamped_negative(self):
        tasks = [Task(description='Only')]
        result = insert_task_impl(tasks, -5, 'Clamped')
        assert len(tasks) == 2
        assert tasks[0].description == 'Clamped'
        assert 'Task inserted at index 0' in result

    def test_insert_into_empty(self):
        tasks: list[Task] = []
        result = insert_task_impl(tasks, 0, 'First')
        assert len(tasks) == 1
        assert 'Task inserted at index 0' in result

    def test_parent_index_adjustment(self):
        """Inserting before a parent adjusts subtask parent_index."""
        tasks = [
            Task(description='Parent'),
            Task(description='Child', parent_index=0),
        ]
        insert_task_impl(tasks, 0, 'New first')
        assert tasks[2].description == 'Child'
        assert tasks[2].parent_index == 1


class TestRemoveTaskImpl:
    def test_remove_single(self):
        tasks = [Task(description='A'), Task(description='B'), Task(description='C')]
        result = remove_task_impl(tasks, 1)
        assert len(tasks) == 2
        assert [t.description for t in tasks] == ['A', 'C']
        assert 'Task 1 removed' in result

    def test_no_plan(self):
        result = remove_task_impl([], 0)
        assert result == 'No plan exists. Use create_plan first.'

    def test_invalid_index(self):
        tasks = [Task(description='Only')]
        result = remove_task_impl(tasks, 5)
        assert 'Invalid task index 5' in result

    def test_negative_index(self):
        tasks = [Task(description='Only')]
        result = remove_task_impl(tasks, -1)
        assert 'Invalid task index -1' in result

    def test_remove_parent_cascades_subtasks(self):
        tasks = [
            Task(description='Parent'),
            Task(description='Child A', parent_index=0),
            Task(description='Child B', parent_index=0),
            Task(description='Other'),
        ]
        result = remove_task_impl(tasks, 0)
        assert len(tasks) == 1
        assert tasks[0].description == 'Other'
        assert 'and 2 subtasks' in result

    def test_remove_parent_with_one_subtask(self):
        tasks = [
            Task(description='Parent'),
            Task(description='Child', parent_index=0),
            Task(description='Other'),
        ]
        result = remove_task_impl(tasks, 0)
        assert len(tasks) == 1
        assert tasks[0].description == 'Other'
        assert 'and 1 subtask)' in result

    def test_remove_subtask_only(self):
        tasks = [
            Task(description='Parent'),
            Task(description='Child', parent_index=0),
        ]
        result = remove_task_impl(tasks, 1)
        assert len(tasks) == 1
        assert tasks[0].description == 'Parent'
        assert 'Task 1 removed' in result

    def test_parent_index_adjustment_after_remove(self):
        """Removing a task adjusts parent_index of later subtasks."""
        tasks = [
            Task(description='A'),
            Task(description='B'),
            Task(description='B-child', parent_index=1),
        ]
        remove_task_impl(tasks, 0)
        assert tasks[0].description == 'B'
        assert tasks[1].description == 'B-child'
        assert tasks[1].parent_index == 0


class TestPlanning:
    def test_serialization_name(self):
        assert Planning.get_serialization_name() == 'Planning'

    def test_tasks_property_empty(self):
        cap: Planning[None] = Planning()
        assert cap.tasks == []

    def test_tasks_property_returns_copy(self):
        cap: Planning[None] = Planning()
        cap.plan_tasks.append(Task(description='test'))
        tasks = cap.tasks
        assert len(tasks) == 1
        # Modifying the returned list doesn't affect the internal state.
        tasks.clear()
        assert len(cap.tasks) == 1

    def test_get_toolset_returns_toolset(self):
        cap: Planning[None] = Planning()
        toolset = cap.get_toolset()
        assert toolset is not None

    def test_get_instructions_returns_callable(self):
        cap: Planning[None] = Planning()
        instructions = cap.get_instructions()
        assert callable(instructions)

    def test_instructions_content_no_plan(self):
        cap: Planning[None] = Planning()
        instructions = cap.get_instructions()
        assert callable(instructions)
        ctx = MagicMock()
        result = instructions(ctx)  # pyright: ignore[reportCallIssue,reportUnknownVariableType]
        assert isinstance(result, str)
        assert 'No plan created yet.' in result
        assert 'planning capability' in result

    def test_instructions_reflect_plan_state(self):
        cap: Planning[None] = Planning()
        cap.plan_tasks.append(Task(description='Do X', status=TaskStatus.in_progress))
        instructions = cap.get_instructions()
        assert callable(instructions)
        ctx = MagicMock()
        result = instructions(ctx)  # pyright: ignore[reportCallIssue,reportUnknownVariableType]
        assert isinstance(result, str)
        assert '0. [~] Do X' in result


class TestPlanningForRun:
    def test_for_run_returns_fresh_instance(self):
        cap: Planning[None] = Planning()
        cap.plan_tasks.append(Task(description='leftover'))

        ctx = MagicMock()
        fresh = asyncio.run(cap.for_run(ctx))

        assert fresh is not cap
        assert fresh.plan_tasks == []
        # Original is unchanged.
        assert len(cap.plan_tasks) == 1


class TestPlanningToolsIntegration:
    """Test the tool functions through shared state with the capability."""

    def test_toolset_has_expected_tools(self):
        cap: Planning[None] = Planning()
        toolset = cap.get_toolset()
        assert toolset is not None
        from pydantic_ai.toolsets.function import FunctionToolset

        assert isinstance(toolset, FunctionToolset)
        assert 'create_plan' in toolset.tools
        assert 'update_task' in toolset.tools
        assert 'add_subtask' in toolset.tools
        assert 'insert_task' in toolset.tools
        assert 'remove_task' in toolset.tools
        assert 'get_plan' in toolset.tools

    def test_tools_share_state_with_capability(self):
        """The tool closures operate on the same list as plan_tasks."""
        cap: Planning[None] = Planning()
        # Use the impl functions with the capability's task list.
        create_plan_impl(cap.plan_tasks, ['Alpha', 'Beta'])
        assert len(cap.plan_tasks) == 2
        assert cap.tasks[0].description == 'Alpha'

        update_task_impl(cap.plan_tasks, 0, TaskStatus.completed)
        assert cap.plan_tasks[0].status == TaskStatus.completed

        result = get_plan_impl(cap.plan_tasks)
        assert '0. [x] Alpha' in result

    def test_tool_closures_delegate_correctly(self):
        """Calling the registered tool functions exercises the closures."""
        from pydantic_ai.toolsets.function import FunctionToolset

        cap: Planning[None] = Planning()
        toolset = cap.get_toolset()
        assert isinstance(toolset, FunctionToolset)
        ctx = MagicMock()

        # Exercise create_plan closure.
        result = toolset.tools['create_plan'].function(ctx, steps=['X', 'Y'])  # pyright: ignore[reportUnknownMemberType,reportUnknownVariableType]
        assert isinstance(result, str)
        assert 'Plan created with 2 steps' in result
        assert len(cap.plan_tasks) == 2

        # Exercise update_task closure.
        result = toolset.tools['update_task'].function(ctx, index=0, status=TaskStatus.completed)  # pyright: ignore[reportUnknownMemberType,reportUnknownVariableType]
        assert isinstance(result, str)
        assert 'Task 0 updated to completed' in result

        # Exercise add_subtask closure.
        result = toolset.tools['add_subtask'].function(ctx, parent_index=1, description='Sub')  # pyright: ignore[reportUnknownMemberType,reportUnknownVariableType]
        assert isinstance(result, str)
        assert 'Subtask added' in result

        # Exercise insert_task closure.
        result = toolset.tools['insert_task'].function(ctx, index=0, description='New first')  # pyright: ignore[reportUnknownMemberType,reportUnknownVariableType]
        assert isinstance(result, str)
        assert 'Task inserted' in result

        # Exercise remove_task closure.
        result = toolset.tools['remove_task'].function(ctx, index=0)  # pyright: ignore[reportUnknownMemberType,reportUnknownVariableType]
        assert isinstance(result, str)
        assert 'removed' in result

        # Exercise get_plan closure.
        result = toolset.tools['get_plan'].function(ctx)  # pyright: ignore[reportUnknownMemberType,reportUnknownVariableType]
        assert isinstance(result, str)
        assert '[x] X' in result

    def test_plan_isolation_between_runs(self):
        """Each for_run produces independent state."""
        cap: Planning[None] = Planning()
        ctx = MagicMock()

        run1: Planning[None] = asyncio.run(cap.for_run(ctx))
        run1.plan_tasks.append(Task(description='Run 1 task'))

        run2: Planning[None] = asyncio.run(cap.for_run(ctx))
        assert run2.plan_tasks == []
        assert len(run1.plan_tasks) == 1
