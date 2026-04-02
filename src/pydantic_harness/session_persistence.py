"""Session persistence capability for saving and loading agent conversation history.

Provides automatic save/restore of conversation messages across agent runs,
with pluggable storage backends (``InMemorySessionStore`` for testing,
``FileSessionStore`` for on-disk persistence via JSON files).
"""

from __future__ import annotations

import json as _json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable
from uuid import uuid4

from pydantic_ai.capabilities.abstract import AbstractCapability
from pydantic_ai.messages import ModelMessage, ModelMessagesTypeAdapter
from pydantic_ai.run import AgentRunResult
from pydantic_ai.tools import AgentDepsT, RunContext


@runtime_checkable
class SessionStore(Protocol):
    """Protocol for pluggable session storage backends."""

    def save(
        self,
        session_id: str,
        messages: list[ModelMessage],
        *,
        metadata: dict[str, Any] | None = None,
    ) -> None:  # pragma: no cover
        """Persist conversation messages (and optional metadata) for the given session."""
        ...

    def load(self, session_id: str) -> list[ModelMessage] | None:  # pragma: no cover
        """Load conversation messages for the given session, or None if not found."""
        ...

    def load_metadata(self, session_id: str) -> dict[str, Any] | None:  # pragma: no cover
        """Load metadata for the given session, or None if not found."""
        ...

    def list_sessions(self) -> list[str]:  # pragma: no cover
        """Return all stored session IDs."""
        ...

    def delete(self, session_id: str) -> bool:  # pragma: no cover
        """Delete a session by ID. Returns True if it existed."""
        ...


class InMemorySessionStore:
    """Dict-based in-memory session store, suitable for testing.

    All data lives in a plain ``dict`` and is lost when the process exits.
    """

    def __init__(self) -> None:
        """Initialize an empty in-memory session store."""
        self._sessions: dict[str, list[ModelMessage]] = {}
        self._metadata: dict[str, dict[str, Any]] = {}

    def save(
        self,
        session_id: str,
        messages: list[ModelMessage],
        *,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Persist conversation messages (and optional metadata) for the given session."""
        self._sessions[session_id] = list(messages)
        if metadata is not None:
            self._metadata[session_id] = dict(metadata)
        else:
            self._metadata.pop(session_id, None)

    def load(self, session_id: str) -> list[ModelMessage] | None:
        """Load conversation messages for the given session."""
        messages = self._sessions.get(session_id)
        if messages is None:
            return None
        return list(messages)

    def load_metadata(self, session_id: str) -> dict[str, Any] | None:
        """Load metadata for the given session."""
        meta = self._metadata.get(session_id)
        if meta is None:
            return None
        return dict(meta)

    def list_sessions(self) -> list[str]:
        """Return all stored session IDs."""
        return list(self._sessions)

    def delete(self, session_id: str) -> bool:
        """Delete a session by ID."""
        self._metadata.pop(session_id, None)
        return self._sessions.pop(session_id, None) is not None


class FileSessionStore:
    """JSON-file-based session store for on-disk persistence.

    Each session is stored as a separate JSON file in the configured directory,
    using ``ModelMessagesTypeAdapter`` for serialization.
    """

    def __init__(self, directory: str | Path) -> None:
        """Initialize a file-backed session store at the given directory.

        Args:
            directory: Path to the directory where session files are stored.
                Created automatically if it does not exist.
        """
        self._directory = Path(directory)

    def _path_for(self, session_id: str) -> Path:
        return self._directory / f'{session_id}.json'

    def _meta_path_for(self, session_id: str) -> Path:
        return self._directory / f'{session_id}.meta.json'

    def save(
        self,
        session_id: str,
        messages: list[ModelMessage],
        *,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Persist conversation messages (and optional metadata) as JSON files."""
        self._directory.mkdir(parents=True, exist_ok=True)
        data = ModelMessagesTypeAdapter.dump_json(messages)
        self._path_for(session_id).write_bytes(data)

        meta_path = self._meta_path_for(session_id)
        if metadata is not None:
            meta_path.write_text(_json.dumps(metadata), encoding='utf-8')
        elif meta_path.exists():
            meta_path.unlink()

    def load(self, session_id: str) -> list[ModelMessage] | None:
        """Load conversation messages from a JSON file."""
        path = self._path_for(session_id)
        if not path.exists():
            return None
        data = path.read_bytes()
        return ModelMessagesTypeAdapter.validate_json(data)

    def load_metadata(self, session_id: str) -> dict[str, Any] | None:
        """Load metadata from a JSON file."""
        meta_path = self._meta_path_for(session_id)
        if not meta_path.exists():
            return None
        raw = meta_path.read_text(encoding='utf-8')
        result: dict[str, Any] = _json.loads(raw)
        return result

    def list_sessions(self) -> list[str]:
        """Return all session IDs found in the directory."""
        if not self._directory.exists():
            return []
        return sorted(p.stem for p in self._directory.glob('*.json') if not p.name.endswith('.meta.json'))

    def delete(self, session_id: str) -> bool:
        """Delete a session file and its metadata. Returns True if it existed."""
        path = self._path_for(session_id)
        existed = path.exists()
        if existed:
            path.unlink()
        meta_path = self._meta_path_for(session_id)
        if meta_path.exists():
            meta_path.unlink()
        return existed


@dataclass
class SessionPersistence(AbstractCapability[AgentDepsT]):
    """Capability for saving and restoring conversation state across agent runs.

    On run start, loads any previously saved messages for the session and
    prepends them to the conversation. On run end, saves the full message
    history back to the store.

    Example:
        ```python
        from pydantic_ai import Agent
        from pydantic_harness.session_persistence import (
            SessionPersistence,
            InMemorySessionStore,
        )

        store = InMemorySessionStore()
        agent = Agent(
            'openai:gpt-4o',
            capabilities=[SessionPersistence(store=store, session_id='my-session')],
        )
        ```
    """

    store: SessionStore = field(default_factory=InMemorySessionStore)
    """The storage backend. Defaults to ``InMemorySessionStore`` (ephemeral)."""

    session_id: str = field(default_factory=lambda: str(uuid4()))
    """Unique identifier for this session. Auto-generated (UUID4) if not provided."""

    auto_save: bool = True
    """Whether to automatically save messages after each run."""

    metadata: dict[str, Any] | None = None
    """Optional metadata to store alongside the session messages.

    When set, this dict is persisted on each save and can be retrieved
    via ``store.load_metadata(session_id)``.
    """

    @classmethod
    def get_serialization_name(cls) -> str | None:
        """Return the name used for spec serialization."""
        return 'SessionPersistence'

    @classmethod
    def from_spec(cls, *args: Any, **kwargs: Any) -> SessionPersistence[Any]:
        """Create from spec arguments.

        Supports ``backend`` kwarg: ``"memory"`` (default) or ``"file"`` (requires ``directory``).
        """
        backend = kwargs.pop('backend', 'memory')
        if backend == 'file':
            directory = kwargs.pop('directory', '.sessions')
            return cls(store=FileSessionStore(directory), **kwargs)
        return cls(store=InMemorySessionStore(), **kwargs)

    async def before_run(
        self,
        ctx: RunContext[AgentDepsT],
    ) -> None:
        """Load saved messages and prepend them to the conversation."""
        existing = self.store.load(self.session_id)
        if existing:
            ctx.messages[:0] = existing

    async def after_run(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        result: AgentRunResult[Any],
    ) -> AgentRunResult[Any]:
        """Save the full message history after a successful run."""
        if self.auto_save:
            self.store.save(self.session_id, result.all_messages(), metadata=self.metadata)
        return result


__all__ = [
    'FileSessionStore',
    'InMemorySessionStore',
    'SessionPersistence',
    'SessionStore',
]
