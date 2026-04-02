"""Tests for pydantic_harness.session_persistence."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

import pytest
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    UserPromptPart,
)
from pydantic_ai.run import AgentRunResult
from pydantic_ai.usage import RunUsage

from pydantic_harness.session_persistence import (
    FileSessionStore,
    InMemorySessionStore,
    SessionPersistence,
    SessionStore,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _user(text: str) -> ModelRequest:
    return ModelRequest(parts=[UserPromptPart(content=text)])


def _assistant(text: str) -> ModelResponse:
    return ModelResponse(parts=[TextPart(content=text)])


def _make_ctx(
    *,
    messages: list[ModelMessage] | None = None,
) -> Any:
    """Build a minimal RunContext-like object for testing hooks."""

    @dataclasses.dataclass
    class _FakeModel:
        model_id: str = 'test-model'

    @dataclasses.dataclass
    class _FakeCtx:
        usage: RunUsage
        model: Any = dataclasses.field(default_factory=_FakeModel)
        deps: None = None
        messages: list[ModelMessage] = dataclasses.field(default_factory=list[ModelMessage])

    ctx = _FakeCtx(usage=RunUsage())
    if messages:
        ctx.messages = list(messages)
    return ctx


def _make_result(messages: list[ModelMessage], output: str = 'done') -> AgentRunResult[str]:
    """Build a minimal AgentRunResult wrapping the given messages."""
    from pydantic_ai._agent_graph import GraphAgentState

    state = GraphAgentState(message_history=list(messages))
    return AgentRunResult(output=output, _state=state)


# ---------------------------------------------------------------------------
# InMemorySessionStore
# ---------------------------------------------------------------------------


class TestInMemorySessionStore:
    def test_protocol_conformance(self) -> None:
        assert isinstance(InMemorySessionStore(), SessionStore)

    def test_save_and_load(self) -> None:
        store = InMemorySessionStore()
        messages: list[ModelMessage] = [_user('hello'), _assistant('hi')]
        store.save('s1', messages)
        loaded = store.load('s1')
        assert loaded is not None
        assert len(loaded) == 2

    def test_load_nonexistent_returns_none(self) -> None:
        store = InMemorySessionStore()
        assert store.load('missing') is None

    def test_save_overwrites(self) -> None:
        store = InMemorySessionStore()
        store.save('s1', [_user('first')])
        store.save('s1', [_user('second')])
        loaded = store.load('s1')
        assert loaded is not None
        assert len(loaded) == 1
        part = loaded[0].parts[0]
        assert isinstance(part, UserPromptPart)
        assert part.content == 'second'

    def test_list_sessions_empty(self) -> None:
        store = InMemorySessionStore()
        assert store.list_sessions() == []

    def test_list_sessions(self) -> None:
        store = InMemorySessionStore()
        store.save('a', [_user('x')])
        store.save('b', [_user('y')])
        assert set(store.list_sessions()) == {'a', 'b'}

    def test_delete_existing(self) -> None:
        store = InMemorySessionStore()
        store.save('s1', [_user('x')])
        assert store.delete('s1') is True
        assert store.load('s1') is None

    def test_delete_nonexistent(self) -> None:
        store = InMemorySessionStore()
        assert store.delete('missing') is False

    def test_save_returns_copy(self) -> None:
        """Mutating the saved list should not affect stored data."""
        store = InMemorySessionStore()
        messages: list[ModelMessage] = [_user('hello')]
        store.save('s1', messages)
        messages.append(_assistant('bye'))
        loaded = store.load('s1')
        assert loaded is not None
        assert len(loaded) == 1

    def test_load_returns_copy(self) -> None:
        """Mutating loaded list should not affect stored data."""
        store = InMemorySessionStore()
        store.save('s1', [_user('hello')])
        loaded = store.load('s1')
        assert loaded is not None
        loaded.append(_assistant('extra'))
        reloaded = store.load('s1')
        assert reloaded is not None
        assert len(reloaded) == 1


# ---------------------------------------------------------------------------
# FileSessionStore
# ---------------------------------------------------------------------------


class TestFileSessionStore:
    def test_protocol_conformance(self, tmp_path: Path) -> None:
        assert isinstance(FileSessionStore(tmp_path), SessionStore)

    def test_save_and_load(self, tmp_path: Path) -> None:
        store = FileSessionStore(tmp_path / 'sessions')
        messages: list[ModelMessage] = [_user('hello'), _assistant('hi')]
        store.save('s1', messages)
        loaded = store.load('s1')
        assert loaded is not None
        assert len(loaded) == 2

    def test_creates_directory(self, tmp_path: Path) -> None:
        d = tmp_path / 'nested' / 'dir'
        store = FileSessionStore(d)
        store.save('s1', [_user('hi')])
        assert d.exists()

    def test_load_nonexistent_returns_none(self, tmp_path: Path) -> None:
        store = FileSessionStore(tmp_path)
        assert store.load('missing') is None

    def test_file_is_valid_json(self, tmp_path: Path) -> None:
        store = FileSessionStore(tmp_path)
        store.save('s1', [_user('hello')])
        raw = (tmp_path / 's1.json').read_text(encoding='utf-8')
        parsed = json.loads(raw)
        assert isinstance(parsed, list)

    def test_list_sessions_empty(self, tmp_path: Path) -> None:
        store = FileSessionStore(tmp_path)
        assert store.list_sessions() == []

    def test_list_sessions_nonexistent_directory(self, tmp_path: Path) -> None:
        store = FileSessionStore(tmp_path / 'nonexistent')
        assert store.list_sessions() == []

    def test_list_sessions(self, tmp_path: Path) -> None:
        store = FileSessionStore(tmp_path)
        store.save('alpha', [_user('x')])
        store.save('beta', [_user('y')])
        assert store.list_sessions() == ['alpha', 'beta']

    def test_delete_existing(self, tmp_path: Path) -> None:
        store = FileSessionStore(tmp_path)
        store.save('s1', [_user('x')])
        assert store.delete('s1') is True
        assert store.load('s1') is None
        assert not (tmp_path / 's1.json').exists()

    def test_delete_nonexistent(self, tmp_path: Path) -> None:
        store = FileSessionStore(tmp_path)
        assert store.delete('missing') is False

    def test_roundtrip_preserves_content(self, tmp_path: Path) -> None:
        store = FileSessionStore(tmp_path)
        original: list[ModelMessage] = [_user('hello world'), _assistant('greetings')]
        store.save('s1', original)
        loaded = store.load('s1')
        assert loaded is not None
        assert len(loaded) == 2
        user_part = loaded[0].parts[0]
        assert isinstance(user_part, UserPromptPart)
        assert user_part.content == 'hello world'
        assistant_part = loaded[1].parts[0]
        assert isinstance(assistant_part, TextPart)
        assert assistant_part.content == 'greetings'


# ---------------------------------------------------------------------------
# SessionPersistence capability
# ---------------------------------------------------------------------------


class TestSessionPersistence:
    def test_auto_generates_session_id(self) -> None:
        cap = SessionPersistence()
        assert cap.session_id
        cap2 = SessionPersistence()
        assert cap.session_id != cap2.session_id

    def test_explicit_session_id(self) -> None:
        cap = SessionPersistence(session_id='my-session')
        assert cap.session_id == 'my-session'

    def test_default_store_is_in_memory(self) -> None:
        cap = SessionPersistence()
        assert isinstance(cap.store, InMemorySessionStore)

    @pytest.mark.anyio
    async def test_before_run_no_history(self) -> None:
        store = InMemorySessionStore()
        cap = SessionPersistence(store=store, session_id='s1')
        ctx = _make_ctx(messages=[_user('new prompt')])
        await cap.before_run(ctx)
        assert len(ctx.messages) == 1

    @pytest.mark.anyio
    async def test_before_run_loads_history(self) -> None:
        store = InMemorySessionStore()
        store.save('s1', [_user('old'), _assistant('response')])
        cap = SessionPersistence(store=store, session_id='s1')
        ctx = _make_ctx(messages=[_user('new prompt')])
        await cap.before_run(ctx)
        assert len(ctx.messages) == 3
        # History is prepended
        first_part = ctx.messages[0].parts[0]
        assert isinstance(first_part, UserPromptPart)
        assert first_part.content == 'old'

    @pytest.mark.anyio
    async def test_after_run_saves_messages(self) -> None:
        store = InMemorySessionStore()
        cap = SessionPersistence(store=store, session_id='s1')
        messages: list[ModelMessage] = [_user('hello'), _assistant('hi')]
        result = _make_result(messages, output='hi')
        ctx = _make_ctx()
        returned = await cap.after_run(ctx, result=result)
        assert returned is result
        loaded = store.load('s1')
        assert loaded is not None
        assert len(loaded) == 2

    @pytest.mark.anyio
    async def test_after_run_auto_save_disabled(self) -> None:
        store = InMemorySessionStore()
        cap = SessionPersistence(store=store, session_id='s1', auto_save=False)
        messages: list[ModelMessage] = [_user('hello'), _assistant('hi')]
        result = _make_result(messages, output='hi')
        ctx = _make_ctx()
        await cap.after_run(ctx, result=result)
        assert store.load('s1') is None

    @pytest.mark.anyio
    async def test_multi_turn_accumulation(self) -> None:
        """Simulate two agent runs that accumulate messages."""
        store = InMemorySessionStore()

        # First run
        cap = SessionPersistence(store=store, session_id='s1')
        run1_messages: list[ModelMessage] = [_user('turn 1'), _assistant('reply 1')]
        result1 = _make_result(run1_messages, output='reply 1')
        ctx1 = _make_ctx()
        await cap.after_run(ctx1, result=result1)

        # Second run: before_run prepends first run's messages
        ctx2 = _make_ctx(messages=[_user('turn 2')])
        await cap.before_run(ctx2)
        assert len(ctx2.messages) == 3  # 2 from history + 1 new

        # Simulate full run result with all messages
        run2_messages: list[ModelMessage] = [
            _user('turn 1'),
            _assistant('reply 1'),
            _user('turn 2'),
            _assistant('reply 2'),
        ]
        result2 = _make_result(run2_messages, output='reply 2')
        await cap.after_run(ctx2, result=result2)

        saved = store.load('s1')
        assert saved is not None
        assert len(saved) == 4

    def test_get_serialization_name(self) -> None:
        assert SessionPersistence.get_serialization_name() == 'SessionPersistence'

    def test_from_spec_default(self) -> None:
        cap = SessionPersistence.from_spec(session_id='s1')
        assert isinstance(cap.store, InMemorySessionStore)
        assert cap.session_id == 's1'

    def test_from_spec_file_backend(self, tmp_path: Path) -> None:
        cap = SessionPersistence.from_spec(backend='file', directory=str(tmp_path))
        assert isinstance(cap.store, FileSessionStore)

    @pytest.mark.anyio
    async def test_with_file_store_roundtrip(self, tmp_path: Path) -> None:
        """Full roundtrip: save via after_run, restore via before_run, using FileSessionStore."""
        store = FileSessionStore(tmp_path / 'sessions')
        cap = SessionPersistence(store=store, session_id='file-session')

        # Simulate first run
        run1_messages: list[ModelMessage] = [_user('hello'), _assistant('hi there')]
        result = _make_result(run1_messages, output='hi there')
        ctx1 = _make_ctx()
        await cap.after_run(ctx1, result=result)

        # Simulate new process: create fresh store instance pointing at same dir
        store2 = FileSessionStore(tmp_path / 'sessions')
        cap2 = SessionPersistence(store=store2, session_id='file-session')
        ctx2 = _make_ctx(messages=[_user('new prompt')])
        await cap2.before_run(ctx2)
        assert len(ctx2.messages) == 3
        first_part = ctx2.messages[0].parts[0]
        assert isinstance(first_part, UserPromptPart)
        assert first_part.content == 'hello'
