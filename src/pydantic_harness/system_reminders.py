"""System reminders capability for periodic behavioral steering.

Provides the [`SystemReminders`][pydantic_harness.SystemReminders] capability,
which injects periodic system messages into model conversations to counteract
instruction fade-out in long-running agent sessions.

Example usage::

    from pydantic_ai import Agent
    from pydantic_harness import SystemReminders, Reminder

    reminders = SystemReminders(
        reminders=[
            Reminder('Remember to use the provided tools.', interval=3),
            Reminder('Always verify your work before responding.', interval=5),
        ],
    )
    agent = Agent('openai:gpt-5', capabilities=[reminders])
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from pydantic_ai.capabilities.abstract import AbstractCapability
from pydantic_ai.messages import ModelRequest, SystemPromptPart
from pydantic_ai.tools import AgentDepsT, RunContext

if TYPE_CHECKING:
    from pydantic_ai.models import ModelRequestContext


@dataclass
class Reminder:
    """A static reminder to inject periodically during an agent run.

    Args:
        content: The reminder text to inject as a system prompt part.
        interval: Inject this reminder every N model requests. For example,
            ``interval=3`` means the reminder fires on the 3rd, 6th, 9th, etc.
            model request within a single run.
        trigger: An optional predicate receiving the current
            [`RunContext`][pydantic_ai.tools.RunContext]. When provided, the
            reminder only fires when the trigger returns ``True`` *and* the
            interval condition is met.
        max_fires: Maximum number of times this reminder may fire within a
            single run. ``None`` means no limit.
        tag: When set, wrap the content in XML tags: ``<tag>content</tag>``.
            For example, ``tag='system-reminder'`` produces
            ``<system-reminder>content</system-reminder>``.
    """

    content: str
    interval: int = 1
    trigger: Callable[[RunContext[Any]], bool] | None = None
    max_fires: int | None = None
    tag: str | None = None

    def __post_init__(self) -> None:  # noqa: D105
        if self.interval < 1:
            raise ValueError(f'interval must be >= 1, got {self.interval}')
        if self.max_fires is not None and self.max_fires < 1:
            raise ValueError(f'max_fires must be >= 1, got {self.max_fires}')

    def render_content(self) -> str:
        """Return the content, wrapped in XML tags if ``tag`` is set."""
        if self.tag is not None:
            return f'<{self.tag}>{self.content}</{self.tag}>'
        return self.content


DynamicReminder = Callable[[RunContext[Any]], str | None]
"""A callable that returns reminder text (or None to skip) based on the current run context.

Dynamic reminders are called on every model request, giving full control
over when and what to inject.
"""

AsyncDynamicReminder = Callable[[RunContext[Any]], Awaitable[str | None]]
"""An async callable variant of [`DynamicReminder`][pydantic_harness.system_reminders.DynamicReminder]."""


@dataclass
class SystemReminders(AbstractCapability[AgentDepsT]):
    r"""Capability that injects periodic system reminders into model conversations.

    System reminders counteract *instruction fade-out* -- the phenomenon where
    agents progressively ignore system prompt guidelines after many turns of
    tool use. Reminders are injected as [`SystemPromptPart`][pydantic_ai.messages.SystemPromptPart]
    entries appended to the last [`ModelRequest`][pydantic_ai.messages.ModelRequest]
    in the message history before each model call.

    Supports two kinds of reminders:

    - **Static** ([`Reminder`][pydantic_harness.Reminder]): a fixed message
      injected every N model requests within a run.
    - **Dynamic** (callable): a function receiving
      [`RunContext`][pydantic_ai.tools.RunContext] and returning a string to inject
      (or ``None`` to skip). Called on every model request.

    Per-run state (the model request counter) is isolated via
    [`for_run`][pydantic_ai.capabilities.AbstractCapability.for_run], so
    concurrent runs on the same agent don't interfere with each other.

    Example::

        reminders = SystemReminders(
            reminders=[
                Reminder('Stay focused on the user\'s original request.', interval=5),
            ],
            dynamic_reminders=[
                lambda ctx: 'Wrap up soon.' if ctx.run_step > 20 else None,
            ],
        )
    """

    reminders: list[Reminder] = field(default_factory=list[Reminder])
    """Static reminders to inject at fixed intervals."""

    dynamic_reminders: list[DynamicReminder | AsyncDynamicReminder] = field(
        default_factory=list[DynamicReminder | AsyncDynamicReminder]
    )
    """Dynamic reminders evaluated on every model request."""

    _request_count: int = field(default=0, init=False, repr=False)
    _fire_counts: list[int] = field(default_factory=list[int], init=False, repr=False)

    def __post_init__(self) -> None:  # noqa: D105
        if not self.reminders and not self.dynamic_reminders:
            raise ValueError('At least one static or dynamic reminder must be provided.')
        self._fire_counts = [0] * len(self.reminders)

    async def for_run(self, ctx: RunContext[AgentDepsT]) -> SystemReminders[AgentDepsT]:
        """Return a fresh instance with a reset request counter for per-run isolation."""
        return SystemReminders(
            reminders=self.reminders,
            dynamic_reminders=self.dynamic_reminders,
        )

    async def before_model_request(
        self,
        ctx: RunContext[AgentDepsT],
        request_context: ModelRequestContext,
    ) -> ModelRequestContext:
        """Inject applicable reminders into the message history before the model call."""
        self._request_count += 1

        parts_to_inject: list[SystemPromptPart] = []

        # Evaluate static reminders based on interval, trigger, and max_fires.
        for idx, reminder in enumerate(self.reminders):
            if self._request_count % reminder.interval != 0:
                continue
            if reminder.trigger is not None and not reminder.trigger(ctx):
                continue
            if reminder.max_fires is not None and self._fire_counts[idx] >= reminder.max_fires:
                continue
            self._fire_counts[idx] += 1
            parts_to_inject.append(SystemPromptPart(content=reminder.render_content()))

        # Evaluate dynamic reminders.
        for dynamic in self.dynamic_reminders:
            result = dynamic(ctx)
            if isinstance(result, Awaitable):
                result = await result
            if result is not None:
                parts_to_inject.append(SystemPromptPart(content=result))

        if parts_to_inject:
            _inject_into_last_request(request_context.messages, parts_to_inject)

        return request_context

    @classmethod
    def get_serialization_name(cls) -> str | None:  # noqa: D102
        return None  # Not spec-serializable (dynamic reminders take callables)


def _inject_into_last_request(
    messages: list[Any],
    parts: list[SystemPromptPart],
) -> None:
    """Append system prompt parts to the last ModelRequest in the message list.

    If no ModelRequest exists yet, prepend one containing just the reminder parts.
    """
    for i in range(len(messages) - 1, -1, -1):
        msg = messages[i]
        if isinstance(msg, ModelRequest):
            # ModelRequest.parts is a Sequence; we need to produce a new list
            # with the reminder parts appended.
            messages[i] = ModelRequest(
                parts=[*msg.parts, *parts],
                timestamp=msg.timestamp,
                instructions=msg.instructions,
                kind=msg.kind,
                run_id=msg.run_id,
                metadata=msg.metadata,
            )
            return
    # No existing request -- create one with just the reminder parts.
    messages.append(ModelRequest(parts=parts))
