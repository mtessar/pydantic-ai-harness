"""Memory capability for persistent agent memory across sessions.

Provides tools for saving, recalling, searching, listing, and deleting
key-value memories, with pluggable storage backends (`InMemoryStore` for
testing, `FileStore` for on-disk persistence).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from pydantic_ai._instructions import AgentInstructions
from pydantic_ai.capabilities.abstract import AbstractCapability
from pydantic_ai.tools import AgentDepsT, RunContext, Tool
from pydantic_ai.toolsets import AgentToolset
from pydantic_ai.toolsets.function import FunctionToolset


@dataclass
class MemoryEntry:
    """A single memory entry with content, tags, and timestamps."""

    key: str
    """Unique identifier for this memory."""

    content: str
    """The content of the memory."""

    tags: list[str] = field(default_factory=list[str])
    """Optional tags for categorization and search."""

    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    """ISO 8601 timestamp of when the memory was first created."""

    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    """ISO 8601 timestamp of the last update."""

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict for JSON storage."""
        return {
            'key': self.key,
            'content': self.content,
            'tags': self.tags,
            'created_at': self.created_at,
            'updated_at': self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MemoryEntry:
        """Deserialize from a plain dict."""
        return cls(
            key=data['key'],
            content=data['content'],
            tags=data.get('tags', []),
            created_at=data.get('created_at', ''),
            updated_at=data.get('updated_at', ''),
        )


@runtime_checkable
class MemoryStore(Protocol):
    """Protocol for pluggable memory storage backends."""

    def get(self, key: str) -> MemoryEntry | None:
        """Retrieve a memory entry by key, or None if not found."""
        ...

    def put(self, entry: MemoryEntry) -> None:
        """Store or update a memory entry."""
        ...

    def delete(self, key: str) -> bool:
        """Delete a memory entry by key. Returns True if it existed."""
        ...

    def list_all(self) -> list[MemoryEntry]:
        """Return all stored memory entries."""
        ...

    def search(self, query: str) -> list[MemoryEntry]:
        """Search entries by substring match on key, content, or tags."""
        ...


class InMemoryStore:
    """Dict-based in-memory store, suitable for testing.

    All data lives in a plain ``dict`` and is lost when the process exits.
    """

    def __init__(self) -> None:
        """Initialize an empty in-memory store."""
        self._entries: dict[str, MemoryEntry] = {}

    def get(self, key: str) -> MemoryEntry | None:
        """Retrieve a memory entry by key."""
        return self._entries.get(key)

    def put(self, entry: MemoryEntry) -> None:
        """Store or update a memory entry."""
        self._entries[entry.key] = entry

    def delete(self, key: str) -> bool:
        """Delete a memory entry by key."""
        return self._entries.pop(key, None) is not None

    def list_all(self) -> list[MemoryEntry]:
        """Return all stored memory entries."""
        return list(self._entries.values())

    def search(self, query: str) -> list[MemoryEntry]:
        """Search entries by substring match on key, content, or tags."""
        q = query.lower()
        return [
            entry
            for entry in self._entries.values()
            if q in entry.key.lower() or q in entry.content.lower() or any(q in tag.lower() for tag in entry.tags)
        ]


class FileStore:
    """JSON-file-based store for simple on-disk persistence.

    Reads the file on initialization and writes back on every mutation.
    """

    def __init__(self, path: str | Path) -> None:
        """Initialize a file-backed store at the given path."""
        self._path = Path(path)
        self._entries: dict[str, MemoryEntry] = {}
        self._load()

    def _load(self) -> None:
        if self._path.exists():
            raw: dict[str, Any] = json.loads(self._path.read_text(encoding='utf-8'))
            self._entries = {key: MemoryEntry.from_dict(val) for key, val in raw.items()}

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        data = {key: entry.to_dict() for key, entry in self._entries.items()}
        self._path.write_text(json.dumps(data, indent=2), encoding='utf-8')

    def get(self, key: str) -> MemoryEntry | None:
        """Retrieve a memory entry by key."""
        return self._entries.get(key)

    def put(self, entry: MemoryEntry) -> None:
        """Store or update a memory entry."""
        self._entries[entry.key] = entry
        self._save()

    def delete(self, key: str) -> bool:
        """Delete a memory entry by key."""
        existed = self._entries.pop(key, None) is not None
        if existed:
            self._save()
        return existed

    def list_all(self) -> list[MemoryEntry]:
        """Return all stored memory entries."""
        return list(self._entries.values())

    def search(self, query: str) -> list[MemoryEntry]:
        """Search entries by substring match on key, content, or tags."""
        q = query.lower()
        return [
            entry
            for entry in self._entries.values()
            if q in entry.key.lower() or q in entry.content.lower() or any(q in tag.lower() for tag in entry.tags)
        ]


def format_entry(entry: MemoryEntry) -> str:
    """Format a memory entry as a human-readable string."""
    line = f'[{entry.key}] {entry.content}'
    if entry.tags:
        line += f' (tags: {", ".join(entry.tags)})'
    return line


@dataclass
class Memory(AbstractCapability[AgentDepsT]):
    """Capability for persistent memory across agent sessions.

    Provides tools for saving, recalling, searching, listing, and deleting
    key-value memories. Uses a pluggable `MemoryStore` backend for storage.

    Example:
        ```python {test="skip" lint="skip"}
        from pydantic_ai import Agent
        from pydantic_harness.memory import Memory, InMemoryStore

        agent = Agent('openai:gpt-4o', capabilities=[Memory(store=InMemoryStore())])
        ```
    """

    store: MemoryStore = field(default_factory=InMemoryStore)
    """The storage backend. Defaults to `InMemoryStore` (ephemeral, dict-based)."""

    inject_memories_in_instructions: bool = True
    """Whether to inject existing memories into the system prompt at run start."""

    max_instructions_memories: int = 20
    """Maximum number of memories to include in the system prompt."""

    @classmethod
    def get_serialization_name(cls) -> str | None:
        """Return the name used for spec serialization."""
        return 'Memory'

    @classmethod
    def from_spec(cls, *args: Any, **kwargs: Any) -> Memory[Any]:
        """Create from spec arguments.

        Supports `backend` kwarg: ``"memory"`` (default) or ``"file"`` (requires `path`).
        """
        backend = kwargs.pop('backend', 'memory')
        if backend == 'file':
            path = kwargs.pop('path', '.memories.json')
            return cls(store=FileStore(path), **kwargs)
        return cls(store=InMemoryStore(), **kwargs)

    def build_instructions(self, ctx: RunContext[AgentDepsT]) -> str:
        """Build dynamic instructions that include currently stored memories."""
        parts: list[str] = [
            'You have access to a persistent memory system. '
            'Use it to save important information that should be remembered across conversations.',
        ]
        if self.inject_memories_in_instructions:
            entries = self.store.list_all()
            if entries:
                parts.append('\nCurrently stored memories:')
                for entry in entries[: self.max_instructions_memories]:
                    parts.append(f'- {format_entry(entry)}')
                overflow = len(entries) - self.max_instructions_memories
                if overflow > 0:
                    parts.append(f'... and {overflow} more (use list_memories or search_memories to see all).')
        return '\n'.join(parts)

    def get_instructions(self) -> AgentInstructions[AgentDepsT] | None:
        """Return dynamic instructions that include stored memories."""
        return self.build_instructions

    def get_toolset(self) -> AgentToolset[AgentDepsT] | None:
        """Return a toolset with memory management tools.

        Tool functions close over ``self`` to access the store without
        requiring anything from the agent's ``deps``.
        """
        store = self.store

        def save_memory(key: str, content: str, tags: list[str] | None = None) -> str:
            """Save or update a memory entry.

            Args:
                key: Unique key for this memory.
                content: The content to remember.
                tags: Optional tags for categorization and search.
            """
            now = datetime.now(timezone.utc).isoformat()
            existing = store.get(key)
            entry = MemoryEntry(
                key=key,
                content=content,
                tags=tags or [],
                created_at=existing.created_at if existing else now,
                updated_at=now,
            )
            store.put(entry)
            return f'Memory saved: {key}'

        def recall_memory(key: str) -> str:
            """Recall a specific memory by its key.

            Args:
                key: The key of the memory to recall.
            """
            entry = store.get(key)
            if entry is None:
                return f'No memory found for key: {key}'
            return format_entry(entry)

        def search_memories(query: str) -> str:
            """Search memories by substring match on keys, content, or tags.

            Args:
                query: The search query string.
            """
            results = store.search(query)
            if not results:
                return f'No memories found matching: {query}'
            return '\n'.join(format_entry(entry) for entry in results)

        def list_memories() -> str:
            """List all stored memories."""
            entries = store.list_all()
            if not entries:
                return 'No memories stored.'
            return '\n'.join(format_entry(entry) for entry in entries)

        def delete_memory(key: str) -> str:
            """Delete a memory by its key.

            Args:
                key: The key of the memory to delete.
            """
            if store.delete(key):
                return f'Memory deleted: {key}'
            return f'No memory found for key: {key}'

        return FunctionToolset(
            [
                Tool(save_memory, takes_ctx=False),
                Tool(recall_memory, takes_ctx=False),
                Tool(search_memories, takes_ctx=False),
                Tool(list_memories, takes_ctx=False),
                Tool(delete_memory, takes_ctx=False),
            ],
        )
