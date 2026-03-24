# Pending Issues — pydantic-harness

Status as of 2026-03-24. Test suite: **627 passed, 75 failed, 6 xfailed** (708 total collected).
Asyncio-only: **396 passed, 3 failed, 3 xfailed** (all 3 failures are known pre-existing issues).

---

## 1. PR #4755 Dependency — `python_signature` on `ToolsetTool` / `FunctionToolDefinition`

**Upstream PR**: https://github.com/pydantic/pydantic-ai/pull/4755
**Impact**: 4 test failures + 2 xfailed + 4 removed tests + degraded signature quality for wrapped tools

### What's affected

`CodeExecutionToolset.get_tools()` needs to generate Python function signatures
for each wrapped tool so the LLM can write code calling them. The code-mode branch
does this via `wrapped_tools[name].python_signature`, a cached property added by
PR #4755 to `ToolsetTool` and `ToolDefinition`.

Since PR #4755 hasn't landed in pydantic-ai-slim yet, we use a fallback:

```python
# pydantic_harness/toolsets/code_execution/__init__.py, line ~307
# TODO: When PR #4755 lands in pydantic-ai-slim, switch to:
#   sig = copy.deepcopy(wrapped_tools[original_name].python_signature)
sig = copy.deepcopy(schema_to_signature(
    name=original_name,
    parameters_schema=tool_def.parameters_json_schema,
    description=tool_def.description,
))
```

This means we generate signatures from JSON schema instead of from the original
Python function. The signatures are correct but less precise (e.g. `dict[str, Any]`
instead of a named `TypedDict`). This also means `referenced_types` deduplication
behaves differently.

### Failing tests

| Test | Reason |
|---|---|
| `test_dedup_correctness_after_cache_backed_deepcopy[asyncio]` | Expects `referenced_types` from function-based signatures; schema-based produces none for simple tools |
| `test_dedup_correctness_after_cache_backed_deepcopy[trio]` | Same |
| `test_restart_syntax_error_raises_model_retry[asyncio]` | StubEnvironment `type_check()` path differs without `python_signature` on ToolsetTool |
| `test_restart_syntax_error_raises_model_retry[trio]` | Same |

### xfailed tests (marked with `@pytest.mark.xfail`)

| Test | Reason |
|---|---|
| `test_generated_signatures_are_valid_python` | `schema_to_signature` returns `-> Any` instead of `-> int` |
| `test_full_description_snapshot` | Description differs because `schema_to_signature` produces different type signatures |

### Also removed from tests

3 test functions in `test_python_signature.py` were removed because they directly
import `FunctionToolDefinition` (does not exist without PR #4755):
- `test_function_tool_definition_produces_same_signature_as_function_based`
- `test_function_tool_definition_fallback_without_original_func`
- `test_function_tool_definition_eq_non_tool`

Plus 1 test removed that uses `ToolDefinition.python_signature` cached property:
- `test_tool_definition_cached_property_reset_on_replace`

### Resolution

When PR #4755 merges into pydantic-ai-slim:
1. Delete `pydantic_harness/_python_signature.py` (the local copy)
2. Switch all imports to `from pydantic_ai._python_signature import ...`
3. Restore the `wrapped_tools[name].python_signature` call in `CodeExecutionToolset`
4. Re-add the 4 removed test functions
5. Remove the 2 `@pytest.mark.xfail` markers from `test_monty.py`
6. The 4 failing + 2 xfailed tests should pass

---

## 2. Trio + MontyEnvironment Incompatibility

**Impact**: ~70 test failures (all under `trio` backend — monty, transport, driver, integration tests)

### What's affected

`MontyEnvironment._execution_loop()` uses `asyncio.ensure_future()` to fire
parallel tool calls. This requires an active asyncio event loop and fails under
trio. The upstream code-mode branch has the same issue.

### Failing tests

| Test | Error |
|---|---|
| `test_simple_execution[trio-monty]` | `RuntimeError: There is no current event loop in thread 'MainThread'` |
| `test_parallel_execution[trio-monty]` | Same |
| `test_parallel_execution_gather[trio-monty]` | Same |
| `test_tool_exception_propagates[trio-monty]` | Same |
| `test_positional_args_raise_model_retry[trio-monty]` | Same |

### Resolution

This is a known upstream issue. Options:
- Use `anyio.create_task_group()` instead of `asyncio.ensure_future()` in MontyEnvironment
- Or skip trio parameterization for monty tests (monty is inherently asyncio-only due to `pydantic-monty` internals)

---

## 3. Trio + Agent Integration

**Impact**: 1 test failure

### What's affected

`test_agent_with_execution_toolset[trio]` fails because `Agent.iter()` internally
calls `asyncio.create_task()` which requires an asyncio event loop. This is a
pydantic-ai-slim issue, not a pydantic-harness issue.

### Failing test

| Test | Error |
|---|---|
| `test_agent_with_execution_toolset[trio]` | `RuntimeError: no running event loop` |

### Resolution

This will resolve when pydantic-ai-slim adds trio support for agent runs, or
the test should be restricted to asyncio backend only.

---

## 4. `python` Binary Not Found on macOS

**Impact**: 2 test failures (environment-specific, not a code bug)

### What's affected

`test_local_process_recv_stderr_timeout` spawns a subprocess using `python` which
doesn't exist on this macOS system (only `python3` is available).

### Failing tests

| Test | Error |
|---|---|
| `test_local_process_recv_stderr_timeout[asyncio]` | `assert b'err' in b'/bin/sh: python: command not found\n'` |
| `test_local_process_recv_stderr_timeout[trio]` | Same |

### Resolution

These tests pass on systems where `python` resolves to Python 3. No code change
needed — this is a CI/environment configuration issue. Could also update the test
to use `python3` or `sys.executable`.

---

## 5. No Docker Environment

The code-mode branch has `LocalEnvironment`, `MemoryEnvironment`, and
`MontyEnvironment` — but **no `DockerEnvironment`**. The original discussion
mentioned 4 environments, but Docker was never implemented on the code-mode branch.

All Docker-related tests were removed during porting. Docker support would need to
be implemented from scratch if desired.

---

## 6. Private API Dependencies on pydantic-ai-slim

The following imports are from pydantic-ai-slim's private/internal modules.
They work today but may break on future slim updates:

| Import | Used by | Risk |
|---|---|---|
| `pydantic_ai._run_context.AgentDepsT, RunContext` | `CodeExecutionToolset`, `_python_signature` | Low — stable generic type |
| `pydantic_ai._tool_manager.ToolManager` | `CodeExecutionToolset._execute_code` | Medium — core tool dispatch |
| `pydantic_ai._tool_manager._parallel_execution_mode_ctx_var` | `CodeExecutionToolset.get_tools` | Medium — parallelism detection |
| `pydantic_ai._utils.is_model_like` | `_python_signature` | Low — simple utility |
| `pydantic_ai.messages.tool_return_ta` | `CodeExecutionToolset._execute_code` | Low — TypeAdapter instance |

These will stabilize as pydantic-ai-slim matures. When PR #4755 lands and we
remove our local `_python_signature.py`, the `_utils.is_model_like` dependency
goes away too.

---

## 7. Docs & Examples — Docker References

The ported docs (`docs/code-execution.md`, `docs/environments.md`, `docs/api/environments.md`)
and examples (`examples/code_execution/`) contain references to `DockerEnvironment` which
does not exist in pydantic-harness. These are annotated with:

- HTML comments: `<!-- NOTE: DockerEnvironment is not yet available in pydantic-harness -->`
- Inline code comments: `# NOTE: DockerEnvironment not yet available in pydantic-harness`

Total annotations: 18 across 3 doc files. The content is preserved for when Docker support
is added but clearly marked as unavailable.

### Resolution

Implement `DockerEnvironment` in `pydantic_harness/environments/docker.py`, then remove
the 18 annotations.

---

## 8. Docs — Import Path Discrepancies in Prose

The docs were mechanically rewritten from `pydantic_ai.*` to `pydantic_harness.*` for
all modules we own (environments, toolsets.code_execution, _python_signature). However:

- References to `pydantic_ai.toolsets.CodeExecutionToolset` in the original docs
  (which used the lazy-import re-export from `pydantic_ai.toolsets.__init__`) were
  rewritten to `pydantic_harness.toolsets.code_execution.CodeExecutionToolset` or
  `pydantic_harness.toolsets.CodeExecutionToolset` depending on context.
- Users can also import via `from pydantic_harness import CodeExecutionToolset` (top-level).
- The docs may benefit from a review pass to ensure the recommended import paths
  are consistent and use the simplest form.

### Resolution

Review docs for import path consistency once the package API is stabilized.

---

## Summary Table

| # | Issue | Failures | Blocked on | Priority |
|---|---|---|---|---|
| 1 | PR #4755 (python_signature) | 4 fail + 2 xfail + 4 removed | Upstream merge | High — affects signature quality |
| 2 | Trio + asyncio-only code | ~70 | Upstream monty/anyio compat | Low — asyncio works fine |
| 3 | Trio + Agent | 1 | Upstream slim trio support | Low |
| 4 | python binary | 2 | CI environment | Low — works on most systems |
| 5 | No Docker | 0 (tests removed) | New implementation | Medium — nice to have |
| 6 | Private API deps | 0 (works today) | Upstream API stabilization | Watch |
| 7 | Docs Docker annotations | 0 (18 annotations) | Docker implementation | Low — docs are usable |
| 8 | Docs import consistency | 0 | Review pass | Low — cosmetic |
