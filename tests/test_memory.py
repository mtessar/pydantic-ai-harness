"""Tests for the Memory capability."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic_ai._run_context import RunContext
from pydantic_ai.toolsets.function import FunctionToolset
from pydantic_ai.usage import RunUsage

from pydantic_harness.memory import (
    FileStore,
    InMemoryStore,
    Memory,
    MemoryEntry,
    MemoryStore,
    format_entry,
)

# --- MemoryEntry ---


class TestMemoryEntry:
    def test_round_trip(self) -> None:
        entry = MemoryEntry(key='k', content='v', tags=['a', 'b'], created_at='t1', updated_at='t2')
        assert MemoryEntry.from_dict(entry.to_dict()) == entry

    def test_from_dict_defaults(self) -> None:
        entry = MemoryEntry.from_dict({'key': 'k', 'content': 'v'})
        assert entry.tags == []
        assert entry.created_at == ''
        assert entry.updated_at == ''

    def test_default_timestamps(self) -> None:
        entry = MemoryEntry(key='k', content='v')
        assert entry.created_at  # non-empty ISO string
        assert entry.updated_at


# --- InMemoryStore ---


class TestInMemoryStore:
    def test_put_and_get(self) -> None:
        store = InMemoryStore()
        entry = MemoryEntry(key='greeting', content='hello')
        store.put(entry)
        assert store.get('greeting') is entry

    def test_get_missing(self) -> None:
        store = InMemoryStore()
        assert store.get('nope') is None

    def test_put_overwrites(self) -> None:
        store = InMemoryStore()
        store.put(MemoryEntry(key='k', content='v1'))
        store.put(MemoryEntry(key='k', content='v2'))
        result = store.get('k')
        assert result is not None
        assert result.content == 'v2'

    def test_delete_existing(self) -> None:
        store = InMemoryStore()
        store.put(MemoryEntry(key='k', content='v'))
        assert store.delete('k') is True
        assert store.get('k') is None

    def test_delete_missing(self) -> None:
        store = InMemoryStore()
        assert store.delete('nope') is False

    def test_list_all_empty(self) -> None:
        store = InMemoryStore()
        assert store.list_all() == []

    def test_list_all(self) -> None:
        store = InMemoryStore()
        store.put(MemoryEntry(key='a', content='alpha'))
        store.put(MemoryEntry(key='b', content='beta'))
        entries = store.list_all()
        assert len(entries) == 2
        assert {e.key for e in entries} == {'a', 'b'}

    def test_search_by_key(self) -> None:
        store = InMemoryStore()
        store.put(MemoryEntry(key='user_name', content='Alice'))
        store.put(MemoryEntry(key='color', content='blue'))
        results = store.search('user')
        assert len(results) == 1
        assert results[0].key == 'user_name'

    def test_search_by_content(self) -> None:
        store = InMemoryStore()
        store.put(MemoryEntry(key='k1', content='the quick brown fox'))
        store.put(MemoryEntry(key='k2', content='lazy dog'))
        results = store.search('fox')
        assert len(results) == 1
        assert results[0].key == 'k1'

    def test_search_by_tag(self) -> None:
        store = InMemoryStore()
        store.put(MemoryEntry(key='k1', content='x', tags=['important']))
        store.put(MemoryEntry(key='k2', content='y', tags=['trivial']))
        results = store.search('important')
        assert len(results) == 1
        assert results[0].key == 'k1'

    def test_search_case_insensitive(self) -> None:
        store = InMemoryStore()
        store.put(MemoryEntry(key='K1', content='Hello World'))
        results = store.search('hello')
        assert len(results) == 1

    def test_search_no_results(self) -> None:
        store = InMemoryStore()
        store.put(MemoryEntry(key='k', content='v'))
        assert store.search('zzz') == []


# --- FileStore ---


class TestFileStore:
    def test_put_and_get(self, tmp_path: Path) -> None:
        path = tmp_path / 'mem.json'
        store = FileStore(path)
        store.put(MemoryEntry(key='k', content='v'))
        assert store.get('k') is not None
        assert store.get('k').content == 'v'  # type: ignore[union-attr]

    def test_persistence(self, tmp_path: Path) -> None:
        path = tmp_path / 'mem.json'
        store1 = FileStore(path)
        store1.put(MemoryEntry(key='k', content='persisted'))

        # New store instance should load from disk
        store2 = FileStore(path)
        result = store2.get('k')
        assert result is not None
        assert result.content == 'persisted'

    def test_delete_saves(self, tmp_path: Path) -> None:
        path = tmp_path / 'mem.json'
        store = FileStore(path)
        store.put(MemoryEntry(key='k', content='v'))
        store.delete('k')

        # Reload and verify deletion persisted
        store2 = FileStore(path)
        assert store2.get('k') is None

    def test_list_all(self, tmp_path: Path) -> None:
        path = tmp_path / 'mem.json'
        store = FileStore(path)
        store.put(MemoryEntry(key='a', content='alpha'))
        store.put(MemoryEntry(key='b', content='beta'))
        assert len(store.list_all()) == 2

    def test_search(self, tmp_path: Path) -> None:
        path = tmp_path / 'mem.json'
        store = FileStore(path)
        store.put(MemoryEntry(key='k1', content='hello', tags=['greeting']))
        store.put(MemoryEntry(key='k2', content='world'))
        assert len(store.search('greeting')) == 1
        assert len(store.search('hello')) == 1

    def test_empty_file(self, tmp_path: Path) -> None:
        path = tmp_path / 'mem.json'
        # File does not exist yet
        store = FileStore(path)
        assert store.list_all() == []

    def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        path = tmp_path / 'sub' / 'dir' / 'mem.json'
        store = FileStore(path)
        store.put(MemoryEntry(key='k', content='v'))
        assert path.exists()

    def test_file_format(self, tmp_path: Path) -> None:
        path = tmp_path / 'mem.json'
        store = FileStore(path)
        store.put(MemoryEntry(key='k', content='v', tags=['t'], created_at='c', updated_at='u'))
        raw = json.loads(path.read_text())
        assert raw == {
            'k': {
                'key': 'k',
                'content': 'v',
                'tags': ['t'],
                'created_at': 'c',
                'updated_at': 'u',
            }
        }


# --- format_entry ---


class TestFormatEntry:
    def test_no_tags(self) -> None:
        entry = MemoryEntry(key='k', content='hello')
        assert format_entry(entry) == '[k] hello'

    def test_with_tags(self) -> None:
        entry = MemoryEntry(key='k', content='hello', tags=['a', 'b'])
        assert format_entry(entry) == '[k] hello (tags: a, b)'


# --- Memory capability ---


class TestMemoryCapability:
    def test_serialization_name(self) -> None:
        assert Memory.get_serialization_name() == 'Memory'

    def test_from_spec_default(self) -> None:
        cap = Memory.from_spec()
        assert isinstance(cap.store, InMemoryStore)

    def test_from_spec_file(self, tmp_path: Path) -> None:
        path = tmp_path / 'mem.json'
        cap = Memory.from_spec(backend='file', path=str(path))
        assert isinstance(cap.store, FileStore)

    def test_default_store(self) -> None:
        cap: Memory[None] = Memory()
        assert isinstance(cap.store, InMemoryStore)

    def test_get_toolset_returns_function_toolset(self) -> None:
        cap: Memory[None] = Memory()
        toolset = cap.get_toolset()
        assert isinstance(toolset, FunctionToolset)

    def test_toolset_has_expected_tools(self) -> None:
        cap: Memory[None] = Memory()
        toolset = cap.get_toolset()
        assert isinstance(toolset, FunctionToolset)
        tool_names = set(toolset.tools.keys())
        assert tool_names == {'save_memory', 'recall_memory', 'search_memories', 'list_memories', 'delete_memory'}


# --- Tool functions (via closure) ---


class TestMemoryTools:
    """Test the tool functions exposed by the Memory capability."""

    @staticmethod
    def _get_tools(store: InMemoryStore | None = None) -> dict[str, Any]:
        cap: Memory[None] = Memory(store=store or InMemoryStore())
        toolset = cap.get_toolset()
        assert isinstance(toolset, FunctionToolset)
        return {name: tool.function for name, tool in toolset.tools.items()}

    def test_save_and_recall(self) -> None:
        store = InMemoryStore()
        tools = self._get_tools(store)
        result = tools['save_memory']('greeting', 'hello world')
        assert result == 'Memory saved: greeting'

        recalled = tools['recall_memory']('greeting')
        assert '[greeting] hello world' in recalled

    def test_recall_missing(self) -> None:
        tools = self._get_tools()
        assert 'No memory found' in tools['recall_memory']('nope')

    def test_save_updates_existing(self) -> None:
        store = InMemoryStore()
        tools = self._get_tools(store)
        tools['save_memory']('k', 'v1')
        original = store.get('k')
        assert original is not None
        original_created = original.created_at

        tools['save_memory']('k', 'v2')
        updated = store.get('k')
        assert updated is not None
        assert updated.content == 'v2'
        # created_at should be preserved
        assert updated.created_at == original_created

    def test_save_with_tags(self) -> None:
        store = InMemoryStore()
        tools = self._get_tools(store)
        tools['save_memory']('k', 'v', ['tag1', 'tag2'])
        entry = store.get('k')
        assert entry is not None
        assert entry.tags == ['tag1', 'tag2']

    def test_search(self) -> None:
        store = InMemoryStore()
        tools = self._get_tools(store)
        tools['save_memory']('user_name', 'Alice')
        tools['save_memory']('color', 'blue')

        result = tools['search_memories']('alice')
        assert 'Alice' in result
        assert 'blue' not in result

    def test_search_no_results(self) -> None:
        tools = self._get_tools()
        assert 'No memories found' in tools['search_memories']('zzz')

    def test_list_empty(self) -> None:
        tools = self._get_tools()
        assert tools['list_memories']() == 'No memories stored.'

    def test_list_with_entries(self) -> None:
        store = InMemoryStore()
        tools = self._get_tools(store)
        tools['save_memory']('a', 'alpha')
        tools['save_memory']('b', 'beta')
        result = tools['list_memories']()
        assert '[a] alpha' in result
        assert '[b] beta' in result

    def test_delete_existing(self) -> None:
        store = InMemoryStore()
        tools = self._get_tools(store)
        tools['save_memory']('k', 'v')
        assert tools['delete_memory']('k') == 'Memory deleted: k'
        assert store.get('k') is None

    def test_delete_missing(self) -> None:
        tools = self._get_tools()
        assert 'No memory found' in tools['delete_memory']('nope')


# --- Instructions ---


class TestMemoryInstructions:
    @staticmethod
    def _make_ctx() -> RunContext[None]:
        from unittest.mock import MagicMock

        return RunContext(
            deps=None,
            model=MagicMock(),
            usage=RunUsage(),
        )

    def test_get_instructions_is_callable(self) -> None:
        cap: Memory[None] = Memory()
        assert callable(cap.get_instructions())

    def test_instructions_with_no_memories(self) -> None:
        cap: Memory[None] = Memory()
        text = cap.build_instructions(self._make_ctx())
        assert 'persistent memory system' in text
        assert 'Currently stored memories' not in text

    def test_instructions_with_memories(self) -> None:
        store = InMemoryStore()
        store.put(MemoryEntry(key='user', content='Alice'))
        cap: Memory[None] = Memory(store=store)
        text = cap.build_instructions(self._make_ctx())
        assert 'Currently stored memories' in text
        assert '[user] Alice' in text

    def test_instructions_respects_max(self) -> None:
        store = InMemoryStore()
        for i in range(25):
            store.put(MemoryEntry(key=f'k{i}', content=f'v{i}'))
        cap: Memory[None] = Memory(store=store, max_instructions_memories=5)
        text = cap.build_instructions(self._make_ctx())
        assert '... and 20 more' in text

    def test_instructions_disabled(self) -> None:
        store = InMemoryStore()
        store.put(MemoryEntry(key='k', content='v'))
        cap: Memory[None] = Memory(store=store, inject_memories_in_instructions=False)
        text = cap.build_instructions(self._make_ctx())
        assert 'Currently stored memories' not in text


# --- MemoryStore protocol ---


class TestMemoryStoreProtocol:
    def test_in_memory_store_satisfies_protocol(self) -> None:
        assert isinstance(InMemoryStore(), MemoryStore)

    def test_file_store_satisfies_protocol(self, tmp_path: Path) -> None:
        assert isinstance(FileStore(tmp_path / 'mem.json'), MemoryStore)
