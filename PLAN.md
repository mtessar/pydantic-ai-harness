# Session Persistence Capability

## Summary

This PR implements the `SessionPersistence` capability for saving and loading agent conversation sessions across process restarts.

## Design

### Storage Protocol

`SessionStore` is a `Protocol` with four methods:
- `save(session_id, messages)` — persist a list of `ModelMessage`
- `load(session_id)` — retrieve messages or `None`
- `list_sessions()` — enumerate stored session IDs
- `delete(session_id)` — remove a session

### Backends

- **`InMemorySessionStore`** — dict-based, for testing (data lost on process exit)
- **`FileSessionStore`** — one JSON file per session in a directory, using `ModelMessagesTypeAdapter` for serialization/deserialization

### Capability

`SessionPersistence(AbstractCapability)`:
- **`before_run`**: loads saved messages and prepends them to `ctx.messages`
- **`after_run`**: saves `result.all_messages()` to the store (when `auto_save=True`)
- **`session_id`**: auto-generated UUID4 if not provided
- **`from_spec`**: supports `backend="memory"` (default) and `backend="file"` (with configurable `directory`)

### Key decisions

- Uses `before_run`/`after_run` hooks (not `before_model_request`) since session restore/save is a per-run concern, not per-request
- Prepends history via `ctx.messages[:0] = existing` for clean integration with the agent's message handling
- `InMemorySessionStore` returns copies to prevent aliasing bugs
- `FileSessionStore` uses `ModelMessagesTypeAdapter.dump_json`/`validate_json` for full-fidelity message serialization

## Files

- `src/pydantic_harness/session_persistence.py` — stores, capability
- `src/pydantic_harness/__init__.py` — re-exports
- `tests/test_session_persistence.py` — 33 tests, 100% coverage
