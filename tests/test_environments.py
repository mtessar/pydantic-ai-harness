"""Tests for pydantic_harness.environments -- ExecutionEnvironment, ExecutionEnvironmentToolset, LocalEnvironment, and MemoryEnvironment."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from inline_snapshot import snapshot

from pydantic_ai import ToolCallPart
from pydantic_ai._run_context import RunContext
from pydantic_ai._tool_manager import ToolManager
from pydantic_harness.environments import ExecutionEnvironmentToolset, ExecutionResult, FileInfo
from pydantic_harness.environments._base import (
    apply_edit,
    build_glob_cmd,
    build_grep_cmd,
    build_read_file_cmd,
    filter_grep_count_output,
    format_lines,
    glob_match,
    parse_glob_output,
    shell_escape,
)
from pydantic_harness.environments.local import LocalEnvironment
from pydantic_harness.environments.memory import MemoryEnvironment
from pydantic_ai.exceptions import UnexpectedModelBehavior
from pydantic_ai.models.test import TestModel
from pydantic_ai.usage import RunUsage

pytestmark = pytest.mark.anyio


def build_run_context(deps: Any = None, run_step: int = 0) -> RunContext[Any]:
    return RunContext(
        deps=deps,
        model=TestModel(),
        usage=RunUsage(),
        prompt=None,
        messages=[],
        run_step=run_step,
    )


# --- Data types ---


def test_execute_result():
    result = ExecutionResult(output='hello\n', exit_code=0)
    assert result.output == 'hello\n'
    assert result.exit_code == 0
    assert result.truncated is False


def test_execute_result_truncated():
    result = ExecutionResult(output='data', exit_code=1, truncated=True)
    assert result.truncated is True


def test_file_info():
    info = FileInfo(name='test.py', path='src/test.py', is_dir=False, size=42)
    assert info.name == 'test.py'
    assert info.is_dir is False
    assert info.size == 42


def test_file_info_directory():
    info = FileInfo(name='src', path='src', is_dir=True)
    assert info.is_dir is True
    assert info.size is None


# --- LocalEnvironment: execute ---


async def test_local_execute_basic(tmp_path: Path):
    async with LocalEnvironment(tmp_path) as env:
        result = await env.shell('echo hello')
        assert result.exit_code == 0
        assert 'hello' in result.output


async def test_local_execute_exit_code(tmp_path: Path):
    async with LocalEnvironment(tmp_path) as env:
        result = await env.shell('exit 42')
        assert result.exit_code == 42


async def test_local_execute_timeout(tmp_path: Path):
    async with LocalEnvironment(tmp_path) as env:
        result = await env.shell('sleep 10', timeout=0.5)
        assert result.exit_code == -1
        assert 'timed out' in result.output.lower()


async def test_local_execute_stderr(tmp_path: Path):
    async with LocalEnvironment(tmp_path) as env:
        result = await env.shell('echo error >&2')
        assert 'error' in result.output


# --- LocalEnvironment: environment variables ---


async def test_local_env_vars_baseline(tmp_path: Path):
    async with LocalEnvironment(tmp_path, env_vars={'MY_VAR': 'baseline'}) as env:
        result = await env.shell('echo $MY_VAR')
        assert 'baseline' in result.output


async def test_local_env_vars_per_call(tmp_path: Path):
    async with LocalEnvironment(tmp_path) as env:
        result = await env.shell('echo $CALL_VAR', env={'CALL_VAR': 'per_call'})
        assert 'per_call' in result.output


async def test_local_env_vars_merged(tmp_path: Path):
    async with LocalEnvironment(tmp_path, env_vars={'BASE': 'one'}) as env:
        result = await env.shell('echo $BASE $EXTRA', env={'EXTRA': 'two'})
        assert 'one' in result.output
        assert 'two' in result.output


async def test_local_env_vars_per_call_overrides_baseline(tmp_path: Path):
    async with LocalEnvironment(tmp_path, env_vars={'VAR': 'old'}) as env:
        result = await env.shell('echo $VAR', env={'VAR': 'new'})
        assert 'new' in result.output
        assert 'old' not in result.output


async def test_local_inherit_env_true(tmp_path: Path):
    os.environ['_TEST_INHERIT_CHECK'] = 'inherited'
    try:
        async with LocalEnvironment(tmp_path, inherit_env=True) as env:
            result = await env.shell('echo $_TEST_INHERIT_CHECK')
            assert 'inherited' in result.output
    finally:
        del os.environ['_TEST_INHERIT_CHECK']


async def test_local_inherit_env_false(tmp_path: Path):
    os.environ['_TEST_INHERIT_CHECK'] = 'should_not_see'
    try:
        async with LocalEnvironment(tmp_path, inherit_env=False) as env:
            result = await env.shell('echo x${_TEST_INHERIT_CHECK}x')
            assert result.output.strip() == 'xx'
    finally:
        del os.environ['_TEST_INHERIT_CHECK']


async def test_local_inherit_env_false_with_explicit_vars(tmp_path: Path):
    async with LocalEnvironment(tmp_path, env_vars={'ONLY_THIS': 'yes'}, inherit_env=False) as env:
        result = await env.shell('/bin/echo $ONLY_THIS')
        assert 'yes' in result.output


# --- LocalEnvironment: file operations ---


async def test_local_write_and_read(tmp_path: Path):
    async with LocalEnvironment(tmp_path) as env:
        await env.write_file('test.txt', 'line one\nline two\n')
        content = await env.read_file('test.txt')
        assert isinstance(content, str)
        assert 'line one' in content
        assert 'line two' in content


async def test_local_read_line_numbers(tmp_path: Path):
    async with LocalEnvironment(tmp_path) as env:
        await env.write_file('numbered.txt', 'alpha\nbeta\ngamma\n')
        content = await env.read_file('numbered.txt')
        assert content == snapshot("""\
     1\talpha
     2\tbeta
     3\tgamma
""")


async def test_local_read_with_offset_limit(tmp_path: Path):
    async with LocalEnvironment(tmp_path) as env:
        lines = '\n'.join(f'line {i}' for i in range(20))
        await env.write_file('long.txt', lines)

        content = await env.read_file('long.txt', offset=5, limit=3)
        assert content == snapshot("""\
     6\tline 5
     7\tline 6
     8\tline 7
... (12 more lines. Use offset=8 to continue reading.)
""")


async def test_local_read_continuation_hint(tmp_path: Path):
    async with LocalEnvironment(tmp_path) as env:
        lines = '\n'.join(f'line {i}' for i in range(20))
        await env.write_file('long.txt', lines)

        content = await env.read_file('long.txt', offset=0, limit=5)
        assert content == snapshot("""\
     1\tline 0
     2\tline 1
     3\tline 2
     4\tline 3
     5\tline 4
... (15 more lines. Use offset=5 to continue reading.)
""")


async def test_local_read_offset_out_of_bounds(tmp_path: Path):
    async with LocalEnvironment(tmp_path) as env:
        await env.write_file('short.txt', 'one\ntwo\n')
        with pytest.raises(ValueError, match='Offset 100 exceeds file length'):
            await env.read_file('short.txt', offset=100)


async def test_local_read_directory_error(tmp_path: Path):
    async with LocalEnvironment(tmp_path) as env:
        (tmp_path / 'subdir').mkdir()
        with pytest.raises(FileNotFoundError, match='is a directory'):
            await env.read_file('subdir')


async def test_local_read_nonexistent(tmp_path: Path):
    async with LocalEnvironment(tmp_path) as env:
        with pytest.raises(FileNotFoundError):
            await env.read_file('nonexistent.txt')


async def test_local_write_creates_parent_dirs(tmp_path: Path):
    async with LocalEnvironment(tmp_path) as env:
        await env.write_file('deep/nested/dir/file.txt', 'content')
        content = await env.read_file('deep/nested/dir/file.txt')
        assert isinstance(content, str)
        assert 'content' in content


async def test_local_write_binary(tmp_path: Path):
    async with LocalEnvironment(tmp_path) as env:
        await env.write_file('binary.bin', b'\x00\x01\x02\x03')
        assert (tmp_path / 'binary.bin').read_bytes() == b'\x00\x01\x02\x03'


async def test_local_read_file_bytes(tmp_path: Path):
    # Create a minimal PNG (1x1 transparent pixel)
    png_data = (
        b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01'
        b'\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89'
        b'\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01'
        b'\r\n\xb4\x00\x00\x00\x00IEND\xaeB`\x82'
    )
    async with LocalEnvironment(tmp_path) as env:
        await env.write_file('image.png', png_data)
        result = await env.read_file('image.png')
        assert isinstance(result, bytes)
        assert result == png_data


# --- LocalEnvironment: edit_file ---


async def test_local_edit_single_replacement(tmp_path: Path):
    async with LocalEnvironment(tmp_path) as env:
        await env.write_file('edit.txt', 'foo bar baz')
        count = await env.replace_str('edit.txt', 'bar', 'BAR')
        assert count == 1
        content = (tmp_path / 'edit.txt').read_text()
        assert content == 'foo BAR baz'


async def test_local_edit_replace_all(tmp_path: Path):
    async with LocalEnvironment(tmp_path) as env:
        await env.write_file('edit.txt', 'aaa bbb aaa')
        count = await env.replace_str('edit.txt', 'aaa', 'xxx', replace_all=True)
        assert count == 2
        content = (tmp_path / 'edit.txt').read_text()
        assert content == 'xxx bbb xxx'


async def test_local_edit_not_found(tmp_path: Path):
    async with LocalEnvironment(tmp_path) as env:
        await env.write_file('edit.txt', 'hello world')
        with pytest.raises(ValueError, match='not found'):
            await env.replace_str('edit.txt', 'missing', 'replacement')


async def test_local_edit_ambiguous_without_replace_all(tmp_path: Path):
    async with LocalEnvironment(tmp_path) as env:
        await env.write_file('edit.txt', 'dup dup dup')
        with pytest.raises(ValueError, match='3 times'):
            await env.replace_str('edit.txt', 'dup', 'unique')


async def test_local_edit_nonexistent_file(tmp_path: Path):
    async with LocalEnvironment(tmp_path) as env:
        with pytest.raises(FileNotFoundError):
            await env.replace_str('missing.txt', 'old', 'new')


async def test_local_edit_multiline(tmp_path: Path):
    async with LocalEnvironment(tmp_path) as env:
        await env.write_file('code.py', 'def foo():\n    return "old"\n\nprint("test")\n')
        count = await env.replace_str('code.py', 'def foo():\n    return "old"', 'def foo():\n    return "new"')
        assert count == 1
        content = (tmp_path / 'code.py').read_text()
        assert 'return "new"' in content
        assert 'return "old"' not in content
        assert 'print("test")' in content


# --- LocalEnvironment: ls ---


async def test_local_ls(tmp_path: Path):
    async with LocalEnvironment(tmp_path) as env:
        await env.write_file('a.txt', 'a')
        await env.write_file('b.txt', 'b')
        (tmp_path / 'subdir').mkdir()

        entries = await env.ls('.')
        names = {e.name for e in entries}
        assert 'a.txt' in names
        assert 'b.txt' in names
        assert 'subdir' in names

        dirs = [e for e in entries if e.is_dir]
        files = [e for e in entries if not e.is_dir]
        assert any(d.name == 'subdir' for d in dirs)
        assert all(f.size is not None and f.size > 0 for f in files)


async def test_local_ls_not_a_directory(tmp_path: Path):
    async with LocalEnvironment(tmp_path) as env:
        await env.write_file('file.txt', 'content')
        with pytest.raises(NotADirectoryError):
            await env.ls('file.txt')


# --- LocalEnvironment: glob ---


async def test_local_glob(tmp_path: Path):
    async with LocalEnvironment(tmp_path) as env:
        await env.write_file('src/main.py', '# main')
        await env.write_file('src/utils.py', '# utils')
        await env.write_file('src/data.json', '{}')

        matches = await env.glob('**/*.py')
        assert len(matches) == 2
        assert any('main.py' in m for m in matches)
        assert any('utils.py' in m for m in matches)
        assert not any('data.json' in m for m in matches)


async def test_local_glob_no_matches(tmp_path: Path):
    async with LocalEnvironment(tmp_path) as env:
        matches = await env.glob('**/*.rs')
        assert matches == []


# --- LocalEnvironment: grep ---


async def test_local_grep(tmp_path: Path):
    async with LocalEnvironment(tmp_path) as env:
        await env.write_file('a.py', 'def hello():\n    pass\n')
        await env.write_file('b.py', 'x = 1\n')

        result = await env.grep('hello')
        assert 'a.py' in result
        assert 'hello' in result
        assert 'b.py' not in result


async def test_local_grep_with_glob_pattern(tmp_path: Path):
    async with LocalEnvironment(tmp_path) as env:
        await env.write_file('code.py', 'target = 1\n')
        await env.write_file('code.js', 'target = 2\n')

        result = await env.grep('target', glob_pattern='*.py')
        assert 'code.py' in result
        assert 'code.js' not in result


async def test_local_grep_line_numbers(tmp_path: Path):
    async with LocalEnvironment(tmp_path) as env:
        await env.write_file('test.txt', 'alpha\nbeta\ngamma\nbeta\n')

        result = await env.grep('beta')
        assert result == snapshot('test.txt:2:beta\ntest.txt:4:beta')


async def test_local_grep_no_matches(tmp_path: Path):
    async with LocalEnvironment(tmp_path) as env:
        await env.write_file('test.txt', 'nothing interesting')
        result = await env.grep('nonexistent_pattern')
        assert result == ''


async def test_local_grep_skips_hidden_files(tmp_path: Path):
    async with LocalEnvironment(tmp_path) as env:
        await env.write_file('visible.py', 'target_string\n')
        (tmp_path / '.hidden').mkdir()
        (tmp_path / '.hidden' / 'secret.py').write_text('target_string\n')
        (tmp_path / '.dotfile').write_text('target_string\n')

        result = await env.grep('target_string')
        assert 'visible.py' in result
        assert '.hidden' not in result
        assert '.dotfile' not in result


# --- LocalEnvironment: create_process ---


async def test_local_create_process(tmp_path: Path):
    async with LocalEnvironment(tmp_path) as env:
        proc = await env.create_process('echo interactive')
        async with proc:
            data = await proc.recv(timeout=5)
            assert b'interactive' in data


async def test_local_create_process_env(tmp_path: Path):
    async with LocalEnvironment(tmp_path) as env:
        proc = await env.create_process('echo $PROC_VAR', env={'PROC_VAR': 'from_process'})
        async with proc:
            data = await proc.recv(timeout=5)
            assert b'from_process' in data


async def test_local_create_process_stdin(tmp_path: Path):
    async with LocalEnvironment(tmp_path) as env:
        # Use head -1 so the process exits after reading one line
        proc = await env.create_process('head -1')
        async with proc:
            await proc.send(b'hello from stdin\n')
            data = await proc.recv(timeout=5)
            assert b'hello from stdin' in data


async def test_local_process_wait(tmp_path: Path):
    async with LocalEnvironment(tmp_path) as env:
        proc = await env.create_process('exit 7')
        async with proc:
            rc = await proc.wait(timeout=5)
            assert rc == 7


async def test_local_process_kill(tmp_path: Path):
    async with LocalEnvironment(tmp_path) as env:
        proc = await env.create_process('sleep 60')
        # Don't use async with -- we want to test manual kill
        await proc.kill()
        assert proc.returncode is not None


# --- LocalEnvironment: path traversal ---


async def test_local_path_traversal_blocked(tmp_path: Path):
    async with LocalEnvironment(tmp_path) as env:
        with pytest.raises(PermissionError, match='outside the environment root'):
            await env.read_file('../../../etc/passwd')


async def test_local_path_traversal_write_blocked(tmp_path: Path):
    async with LocalEnvironment(tmp_path) as env:
        with pytest.raises(PermissionError, match='outside the environment root'):
            await env.write_file('../escape.txt', 'malicious')


# --- LocalEnvironment: creates root dir ---


async def test_local_creates_root_dir(tmp_path: Path):
    root = tmp_path / 'new_root'
    assert not root.exists()
    async with LocalEnvironment(root) as env:
        assert root.exists()
        result = await env.shell('echo works')
        assert 'works' in result.output


# --- ExecutionEnvironmentToolset ---


async def test_toolset_tool_names():
    toolset = ExecutionEnvironmentToolset(LocalEnvironment('.'))
    tool_names = sorted(toolset.tools.keys())
    assert tool_names == snapshot(['glob', 'grep', 'ls', 'read_file', 'replace_str', 'shell', 'write_file'])


async def test_toolset_include_flags():
    toolset = ExecutionEnvironmentToolset(
        LocalEnvironment('.'),
        include=frozenset(),
    )
    assert toolset.tools == {}


async def test_toolset_include_shell_only():
    toolset = ExecutionEnvironmentToolset(
        LocalEnvironment('.'),
        include=frozenset({'shell'}),
    )
    assert sorted(toolset.tools.keys()) == ['shell']


async def test_toolset_bash_tool(tmp_path: Path):
    env = LocalEnvironment(tmp_path)
    toolset = ExecutionEnvironmentToolset(env)
    ctx = build_run_context(None)
    manager = await ToolManager[None](toolset).for_run_step(ctx)

    async with env:
        result = await manager.handle_call(ToolCallPart(tool_name='shell', args={'command': 'echo hello'}))
        assert result == snapshot("""\
hello

Exit code: 0\
""")


async def test_toolset_read_write_tools(tmp_path: Path):
    env = LocalEnvironment(tmp_path)
    toolset = ExecutionEnvironmentToolset(env)
    ctx = build_run_context(None)
    manager = await ToolManager[None](toolset).for_run_step(ctx)

    async with env:
        # Write
        write_result = await manager.handle_call(
            ToolCallPart(tool_name='write_file', args={'path': 'test.txt', 'content': 'hello world'})
        )
        assert write_result == snapshot('File written: test.txt')

        # Read
        read_result = await manager.handle_call(ToolCallPart(tool_name='read_file', args={'path': 'test.txt'}))
        assert read_result == snapshot('     1\thello world\n')


async def test_toolset_edit_retry_on_error(tmp_path: Path):
    env = LocalEnvironment(tmp_path)
    toolset = ExecutionEnvironmentToolset(env, max_retries=0)
    ctx = build_run_context(None)
    manager = await ToolManager[None](toolset).for_run_step(ctx)

    async with env:
        await env.write_file('test.txt', 'content')

        # Edit with non-matching string: ModelRetry is raised by tool, but with max_retries=0
        # the ToolManager wraps it into UnexpectedModelBehavior
        with pytest.raises(UnexpectedModelBehavior, match='exceeded max retries count of 0'):
            await manager.handle_call(
                ToolCallPart(
                    tool_name='replace_str',
                    args={'path': 'test.txt', 'old': 'nonexistent', 'new': 'replacement'},
                )
            )


async def test_toolset_glob_tool(tmp_path: Path):
    env = LocalEnvironment(tmp_path)
    toolset = ExecutionEnvironmentToolset(env)
    ctx = build_run_context(None)
    manager = await ToolManager[None](toolset).for_run_step(ctx)

    async with env:
        await env.write_file('a.py', '# a')
        await env.write_file('b.py', '# b')

        result = await manager.handle_call(ToolCallPart(tool_name='glob', args={'pattern': '*.py'}))
        assert result == snapshot("""\
a.py
b.py\
""")


async def test_toolset_grep_tool(tmp_path: Path):
    env = LocalEnvironment(tmp_path)
    toolset = ExecutionEnvironmentToolset(env)
    ctx = build_run_context(None)
    manager = await ToolManager[None](toolset).for_run_step(ctx)

    async with env:
        await env.write_file('search.py', 'def find_me():\n    pass\n')

        result = await manager.handle_call(ToolCallPart(tool_name='grep', args={'pattern': 'find_me'}))
        assert result == snapshot('search.py:1:def find_me():')


# --- ExecutionEnvironmentToolset: error handling ---


async def test_toolset_read_nonexistent_returns_error(tmp_path: Path):
    """read_file on a nonexistent file returns an error string instead of crashing."""
    env = LocalEnvironment(tmp_path)
    toolset = ExecutionEnvironmentToolset(env)
    ctx = build_run_context(None)
    manager = await ToolManager[None](toolset).for_run_step(ctx)

    async with env:
        result = await manager.handle_call(ToolCallPart(tool_name='read_file', args={'path': 'nope.txt'}))
        assert 'Error:' in str(result)


async def test_toolset_read_path_traversal_returns_error(tmp_path: Path):
    """read_file with path traversal returns an error string."""
    env = LocalEnvironment(tmp_path)
    toolset = ExecutionEnvironmentToolset(env)
    ctx = build_run_context(None)
    manager = await ToolManager[None](toolset).for_run_step(ctx)

    async with env:
        result = await manager.handle_call(ToolCallPart(tool_name='read_file', args={'path': '../../etc/passwd'}))
        assert 'Error:' in str(result)


async def test_toolset_write_path_traversal_returns_error(tmp_path: Path):
    """write_file with path traversal returns an error string."""
    env = LocalEnvironment(tmp_path)
    toolset = ExecutionEnvironmentToolset(env)
    ctx = build_run_context(None)
    manager = await ToolManager[None](toolset).for_run_step(ctx)

    async with env:
        result = await manager.handle_call(
            ToolCallPart(tool_name='write_file', args={'path': '../../tmp/evil.txt', 'content': 'bad'})
        )
        assert 'Error:' in str(result)


async def test_toolset_glob_path_traversal_returns_error(tmp_path: Path):
    """glob with path traversal returns an error string."""
    env = LocalEnvironment(tmp_path)
    toolset = ExecutionEnvironmentToolset(env)
    ctx = build_run_context(None)
    manager = await ToolManager[None](toolset).for_run_step(ctx)

    async with env:
        result = await manager.handle_call(
            ToolCallPart(tool_name='glob', args={'pattern': '*.py', 'path': '../../etc'})
        )
        assert 'Error:' in str(result)


async def test_toolset_grep_invalid_regex_returns_error(tmp_path: Path):
    """grep with invalid regex returns an error string."""
    env = LocalEnvironment(tmp_path)
    toolset = ExecutionEnvironmentToolset(env)
    ctx = build_run_context(None)
    manager = await ToolManager[None](toolset).for_run_step(ctx)

    async with env:
        await env.write_file('test.txt', 'content')

        result = await manager.handle_call(ToolCallPart(tool_name='grep', args={'pattern': '[invalid'}))
        assert 'Error:' in str(result)


async def test_toolset_read_offset_out_of_bounds_returns_error(tmp_path: Path):
    """read_file with offset past EOF returns an error string."""
    env = LocalEnvironment(tmp_path)
    toolset = ExecutionEnvironmentToolset(env)
    ctx = build_run_context(None)
    manager = await ToolManager[None](toolset).for_run_step(ctx)

    async with env:
        await env.write_file('short.txt', 'one\ntwo\n')

        result = await manager.handle_call(
            ToolCallPart(tool_name='read_file', args={'path': 'short.txt', 'offset': 100})
        )
        assert 'Error:' in str(result)
        assert 'Offset 100 exceeds' in str(result)


async def test_toolset_read_continuation_hint(tmp_path: Path):
    """read_file includes continuation hint when there are more lines."""
    env = LocalEnvironment(tmp_path)
    toolset = ExecutionEnvironmentToolset(env)
    ctx = build_run_context(None)
    manager = await ToolManager[None](toolset).for_run_step(ctx)

    async with env:
        lines = '\n'.join(f'line {i}' for i in range(20))
        await env.write_file('long.txt', lines)

        result = await manager.handle_call(
            ToolCallPart(tool_name='read_file', args={'path': 'long.txt', 'offset': 0, 'limit': 5})
        )
        assert result == snapshot("""\
     1	line 0
     2	line 1
     3	line 2
     4	line 3
     5	line 4
... (15 more lines. Use offset=5 to continue reading.)
""")


# --- ExecutionEnvironmentToolset: approval flags ---


async def test_toolset_require_shell_approval():
    """require_shell_approval sets requires_approval on the shell tool."""
    toolset = ExecutionEnvironmentToolset(require_shell_approval=True)
    ctx = build_run_context(None)
    tools = await toolset.get_tools(ctx)
    assert tools['shell'].tool_def.kind == 'unapproved'
    # Other tools should be normal
    assert tools['read_file'].tool_def.kind == 'function'


async def test_toolset_require_write_approval():
    """require_write_approval sets requires_approval on write_file and replace_str."""
    toolset = ExecutionEnvironmentToolset(require_write_approval=True)
    ctx = build_run_context(None)
    tools = await toolset.get_tools(ctx)
    assert tools['write_file'].tool_def.kind == 'unapproved'
    assert tools['replace_str'].tool_def.kind == 'unapproved'
    # read_file and search tools should NOT require approval
    assert tools['read_file'].tool_def.kind == 'function'
    assert tools['glob'].tool_def.kind == 'function'
    assert tools['grep'].tool_def.kind == 'function'


async def test_toolset_default_no_approval():
    """By default, no tools require approval."""
    toolset = ExecutionEnvironmentToolset()
    ctx = build_run_context(None)
    tools = await toolset.get_tools(ctx)
    for tool in tools.values():
        assert tool.tool_def.kind == 'function'


# --- ExecutionEnvironmentToolset: environment management ---


async def test_toolset_environment_property():
    env = LocalEnvironment('.')
    toolset = ExecutionEnvironmentToolset(env)
    assert toolset.environment is env
    assert toolset.required_environment is env


async def test_toolset_no_environment_returns_none():
    toolset = ExecutionEnvironmentToolset()
    assert toolset.environment is None


async def test_toolset_no_environment_required_raises():
    toolset = ExecutionEnvironmentToolset()
    with pytest.raises(RuntimeError, match='No execution environment configured'):
        _ = toolset.required_environment


async def test_toolset_use_environment():
    env1 = LocalEnvironment('/tmp/env1')
    env2 = LocalEnvironment('/tmp/env2')
    toolset = ExecutionEnvironmentToolset(env1)

    assert toolset.environment is env1
    with toolset.use_environment(env2):
        assert toolset.environment is env2
    assert toolset.environment is env1


async def test_toolset_use_environment_no_default():
    env = LocalEnvironment('.')
    toolset = ExecutionEnvironmentToolset()

    assert toolset.environment is None

    with toolset.use_environment(env):
        assert toolset.environment is env

    assert toolset.environment is None


async def test_toolset_instructions():
    """Environment instructions is accessible for each tool."""
    env = LocalEnvironment('.')
    # LocalEnvironment returns None for all tool descriptions by default
    assert env.instructions('shell') is None
    assert env.instructions('read_file') is None


async def test_toolset_tool_name_conflict_hint():
    toolset = ExecutionEnvironmentToolset(LocalEnvironment('.'))
    assert 'PrefixedToolset' in toolset.tool_name_conflict_hint


# --- ExecutionEnvironmentToolset: lifecycle ---


async def test_toolset_lifecycle(tmp_path: Path):
    env = LocalEnvironment(tmp_path)
    toolset = ExecutionEnvironmentToolset(env)

    async with toolset:
        result = await env.shell('echo lifecycle')
        assert 'lifecycle' in result.output


# --- ExecutionEnvironmentToolset: image support ---


async def test_toolset_image_support_disabled(tmp_path: Path):
    env = LocalEnvironment(tmp_path)
    toolset = ExecutionEnvironmentToolset(env, image_support=False)
    ctx = build_run_context(None)
    manager = await ToolManager[None](toolset).for_run_step(ctx)

    async with env:
        await env.write_file('photo.png', b'\x89PNG\r\n\x1a\n')
        result = await manager.handle_call(ToolCallPart(tool_name='read_file', args={'path': 'photo.png'}))
        assert result == snapshot('[Image file: photo.png — image_support is disabled on this toolset]')


# --- LocalEnvironment: grep output modes ---


async def test_local_grep_files_with_matches(tmp_path: Path):
    async with LocalEnvironment(tmp_path) as env:
        await env.write_file('a.py', 'target = 1\nother = 2\n')
        await env.write_file('b.py', 'target = 3\ntarget = 4\n')
        await env.write_file('c.py', 'nothing here\n')

        result = await env.grep('target', output_mode='files_with_matches')
        lines = result.strip().splitlines()
        assert sorted(lines) == ['a.py', 'b.py']


async def test_local_grep_count(tmp_path: Path):
    async with LocalEnvironment(tmp_path) as env:
        await env.write_file('a.py', 'target = 1\nother = 2\n')
        await env.write_file('b.py', 'target = 3\ntarget = 4\n')
        await env.write_file('c.py', 'nothing here\n')

        result = await env.grep('target', output_mode='count')
        lines = sorted(result.strip().splitlines())
        assert lines == ['a.py:1', 'b.py:2']


async def test_local_grep_content_default(tmp_path: Path):
    """Default output_mode is 'content' with file:line:text format."""
    async with LocalEnvironment(tmp_path) as env:
        await env.write_file('test.py', 'hello\nworld\n')

        result = await env.grep('hello')
        assert result == snapshot('test.py:1:hello')


# --- LocalEnvironment: binary file detection ---


async def test_local_grep_skips_binary_files(tmp_path: Path):
    async with LocalEnvironment(tmp_path) as env:
        await env.write_file('text.py', 'findme = True\n')
        await env.write_file('binary.pyc', b'\x00\x01\x02findme\x03\x04')

        result = await env.grep('findme')
        assert 'text.py' in result
        assert 'binary.pyc' not in result


async def test_local_grep_binary_detection_first_8kb(tmp_path: Path):
    """Binary detection checks only the first 8KB."""
    async with LocalEnvironment(tmp_path) as env:
        # File with null byte after 8KB -- should be treated as text
        content = 'findme\n' + ('x' * 8200) + '\x00'
        await env.write_file('mostly_text.txt', content)

        result = await env.grep('findme')
        assert 'mostly_text.txt' in result


# --- Toolset: grep output_mode ---


async def test_toolset_grep_files_with_matches(tmp_path: Path):
    env = LocalEnvironment(tmp_path)
    toolset = ExecutionEnvironmentToolset(env)
    ctx = build_run_context(None)
    manager = await ToolManager[None](toolset).for_run_step(ctx)

    async with env:
        await env.write_file('a.py', 'target = 1\n')
        await env.write_file('b.py', 'other = 2\n')

        result = await manager.handle_call(
            ToolCallPart(tool_name='grep', args={'pattern': 'target', 'output_mode': 'files_with_matches'})
        )
        assert result == snapshot('a.py')


async def test_toolset_grep_count(tmp_path: Path):
    env = LocalEnvironment(tmp_path)
    toolset = ExecutionEnvironmentToolset(env)
    ctx = build_run_context(None)
    manager = await ToolManager[None](toolset).for_run_step(ctx)

    async with env:
        await env.write_file('a.py', 'x = 1\nx = 2\nx = 3\n')

        result = await manager.handle_call(
            ToolCallPart(tool_name='grep', args={'pattern': 'x', 'output_mode': 'count'})
        )
        assert result == snapshot('a.py:3')


# --- MemoryEnvironment ---


async def test_memory_read_write():
    async with MemoryEnvironment() as env:
        await env.write_file('test.txt', 'hello world\n')
        content = await env.read_file('test.txt')
        assert content == snapshot("""\
     1\thello world
""")


async def test_memory_initial_files():
    env = MemoryEnvironment(files={'a.txt': 'alpha', 'b.txt': 'beta'})
    async with env:
        a = await env.read_file('a.txt')
        assert isinstance(a, str)
        assert 'alpha' in a
        b = await env.read_file('b.txt')
        assert isinstance(b, str)
        assert 'beta' in b


async def test_memory_read_nonexistent():
    async with MemoryEnvironment() as env:
        with pytest.raises(FileNotFoundError):
            await env.read_file('nope.txt')


async def test_memory_read_directory_error():
    env = MemoryEnvironment(files={'dir/file.txt': 'content'})
    async with env:
        with pytest.raises(FileNotFoundError, match='is a directory'):
            await env.read_file('dir')


async def test_memory_read_offset_limit():
    lines = '\n'.join(f'line {i}' for i in range(20))
    env = MemoryEnvironment(files={'long.txt': lines})
    async with env:
        content = await env.read_file('long.txt', offset=5, limit=3)
        assert isinstance(content, str)
        assert 'line 5' in content
        assert 'line 7' in content
        assert 'line 4' not in content
        assert 'line 8' not in content


async def test_memory_read_continuation_hint():
    lines = '\n'.join(f'line {i}' for i in range(20))
    env = MemoryEnvironment(files={'long.txt': lines})
    async with env:
        content = await env.read_file('long.txt', offset=0, limit=5)
        assert isinstance(content, str)
        assert '15 more lines' in content
        assert 'offset=5' in content


async def test_memory_read_offset_out_of_bounds():
    env = MemoryEnvironment(files={'short.txt': 'one\ntwo\n'})
    async with env:
        with pytest.raises(ValueError, match='Offset 100 exceeds'):
            await env.read_file('short.txt', offset=100)


async def test_memory_edit_file():
    env = MemoryEnvironment(files={'code.py': 'old_value = 1'})
    async with env:
        count = await env.replace_str('code.py', 'old_value', 'new_value')
        assert count == 1
        content = await env.read_file('code.py')
        assert isinstance(content, str)
        assert 'new_value' in content
        assert 'old_value' not in content


async def test_memory_edit_file_not_found():
    async with MemoryEnvironment() as env:
        with pytest.raises(FileNotFoundError):
            await env.replace_str('nope.txt', 'a', 'b')


async def test_memory_edit_string_not_found():
    env = MemoryEnvironment(files={'f.txt': 'hello'})
    async with env:
        with pytest.raises(ValueError, match='not found'):
            await env.replace_str('f.txt', 'missing', 'replacement')


async def test_memory_edit_ambiguous():
    env = MemoryEnvironment(files={'f.txt': 'dup dup dup'})
    async with env:
        with pytest.raises(ValueError, match='3 times'):
            await env.replace_str('f.txt', 'dup', 'x')


async def test_memory_edit_replace_all():
    env = MemoryEnvironment(files={'f.txt': 'aaa bbb aaa'})
    async with env:
        count = await env.replace_str('f.txt', 'aaa', 'xxx', replace_all=True)
        assert count == 2
        content = await env.read_file('f.txt')
        assert isinstance(content, str)
        assert 'xxx bbb xxx' in content


async def test_memory_ls():
    env = MemoryEnvironment(
        files={
            'a.txt': 'a',
            'b.txt': 'bb',
            'sub/c.txt': 'ccc',
        }
    )
    async with env:
        entries = await env.ls('.')
        names = {e.name for e in entries}
        assert names == {'a.txt', 'b.txt', 'sub'}

        dirs = [e for e in entries if e.is_dir]
        files = [e for e in entries if not e.is_dir]
        assert len(dirs) == 1
        assert dirs[0].name == 'sub'
        assert all(f.size is not None for f in files)


async def test_memory_ls_subdirectory():
    env = MemoryEnvironment(files={'sub/a.txt': 'a', 'sub/b.txt': 'b'})
    async with env:
        entries = await env.ls('sub')
        names = {e.name for e in entries}
        assert names == {'a.txt', 'b.txt'}


async def test_memory_ls_not_a_directory():
    async with MemoryEnvironment() as env:
        with pytest.raises(NotADirectoryError):
            await env.ls('nonexistent')


async def test_memory_glob():
    env = MemoryEnvironment(
        files={
            'src/main.py': '# main',
            'src/utils.py': '# utils',
            'src/data.json': '{}',
        }
    )
    async with env:
        matches = await env.glob('*.py', path='src')
        assert sorted(matches) == ['src/main.py', 'src/utils.py']


async def test_memory_glob_no_matches():
    env = MemoryEnvironment(files={'a.py': ''})
    async with env:
        matches = await env.glob('*.rs')
        assert matches == []


async def test_memory_grep_content():
    env = MemoryEnvironment(
        files={
            'a.py': 'def hello():\n    pass\n',
            'b.py': 'x = 1\n',
        }
    )
    async with env:
        result = await env.grep('hello')
        assert result == snapshot('a.py:1:def hello():')


async def test_memory_grep_files_with_matches():
    env = MemoryEnvironment(
        files={
            'a.py': 'target = 1\n',
            'b.py': 'target = 2\ntarget = 3\n',
            'c.py': 'nothing\n',
        }
    )
    async with env:
        result = await env.grep('target', output_mode='files_with_matches')
        lines = sorted(result.strip().splitlines())
        assert lines == ['a.py', 'b.py']


async def test_memory_grep_count():
    env = MemoryEnvironment(
        files={
            'a.py': 'x = 1\n',
            'b.py': 'x = 2\nx = 3\n',
        }
    )
    async with env:
        result = await env.grep('x', output_mode='count')
        lines = sorted(result.strip().splitlines())
        assert lines == ['a.py:1', 'b.py:2']


async def test_memory_grep_skips_binary():
    env = MemoryEnvironment(
        files={
            'text.py': 'findme = True\n',
            'binary.dat': b'\x00\x01findme\x02',
        }
    )
    async with env:
        result = await env.grep('findme')
        assert 'text.py' in result
        assert 'binary.dat' not in result


async def test_memory_grep_skips_hidden():
    env = MemoryEnvironment(
        files={
            'visible.py': 'target\n',
            '.hidden/secret.py': 'target\n',
        }
    )
    async with env:
        result = await env.grep('target')
        assert 'visible.py' in result
        assert '.hidden' not in result


async def test_memory_grep_with_glob_pattern():
    env = MemoryEnvironment(
        files={
            'code.py': 'target\n',
            'code.js': 'target\n',
        }
    )
    async with env:
        result = await env.grep('target', glob_pattern='*.py')
        assert 'code.py' in result
        assert 'code.js' not in result


async def test_memory_execute_with_handler():
    def handler(cmd: str) -> ExecutionResult:
        return ExecutionResult(output=f'ran: {cmd}\n', exit_code=0)

    async with MemoryEnvironment(command_handler=handler) as env:
        result = await env.shell('echo hello')
        assert result.output == 'ran: echo hello\n'
        assert result.exit_code == 0


async def test_memory_execute_no_handler():
    async with MemoryEnvironment() as env:
        with pytest.raises(RuntimeError, match='no command_handler'):
            await env.shell('echo hello')


async def test_memory_create_process_not_supported():
    async with MemoryEnvironment() as env:
        with pytest.raises(NotImplementedError):
            await env.create_process('echo hello')


async def test_memory_write_binary():
    async with MemoryEnvironment() as env:
        await env.write_file('data.bin', b'\x00\x01\x02')
        # Non-image binary files are returned as text (decoded)
        content = await env.read_file('data.bin')
        assert isinstance(content, str)


async def test_memory_read_file_bytes():
    png_data = b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR'
    env = MemoryEnvironment(files={'img.png': png_data})
    async with env:
        result = await env.read_file('img.png')
        assert isinstance(result, bytes)
        assert result == png_data


# --- MemoryEnvironment with ExecutionEnvironmentToolset ---


async def test_memory_toolset_integration():
    """MemoryEnvironment works with ExecutionEnvironmentToolset for full agent testing."""
    env = MemoryEnvironment(files={'main.py': 'print("hello")\n'})
    toolset = ExecutionEnvironmentToolset(env, exclude=frozenset({'shell', 'run_code'}))
    ctx = build_run_context(None)
    manager = await ToolManager[None](toolset).for_run_step(ctx)

    async with env:
        # read_file
        result = await manager.handle_call(ToolCallPart(tool_name='read_file', args={'path': 'main.py'}))
        assert result == snapshot('     1\tprint("hello")\n')

        # write_file
        result = await manager.handle_call(
            ToolCallPart(tool_name='write_file', args={'path': 'new.py', 'content': 'x = 1'})
        )
        assert result == snapshot('File written: new.py')

        # glob
        result = await manager.handle_call(ToolCallPart(tool_name='glob', args={'pattern': '*.py'}))
        assert result == snapshot("""\
main.py
new.py\
""")

        # grep
        result = await manager.handle_call(ToolCallPart(tool_name='grep', args={'pattern': 'hello'}))
        assert result == snapshot('main.py:1:print("hello")')


# --- Agent-level integration test ---


async def test_agent_with_execution_toolset():
    """Agent with ExecutionEnvironmentToolset runs end-to-end using TestModel and MemoryEnvironment."""
    from pydantic_ai import Agent

    env = MemoryEnvironment(
        files={'data.txt': 'hello world\n'},
        command_handler=lambda cmd: ExecutionResult(output=f'executed: {cmd}\n', exit_code=0),
    )
    toolset = ExecutionEnvironmentToolset(env)

    agent = Agent('test', toolsets=[toolset])

    async with env:
        result = await agent.run('Read the file data.txt')
        # The TestModel will call tools and we verify it completes without error
        assert result.output is not None


# pyright: reportPrivateUsage=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false, reportPossiblyUnboundVariable=false


# --- _base.py helper functions ---


def test_shell_escape():
    assert shell_escape('hello') == "'hello'"
    assert shell_escape("it's") == "'it'\\''s'"
    assert shell_escape('') == "''"
    assert shell_escape('a b c') == "'a b c'"


def test_format_lines_empty_file():
    """format_lines on empty string returns just a newline."""
    result = format_lines('', 0, 2000)
    assert result == '\n'


def test_format_lines_trailing_newline():
    """format_lines adds trailing newline when text doesn't end with one."""
    result = format_lines('no trailing newline', 0, 2000)
    assert result.endswith('\n')
    assert '1\tno trailing newline' in result


def test_glob_match_simple():
    assert glob_match('foo.py', '*.py') is True
    assert glob_match('foo.txt', '*.py') is False


def test_glob_match_double_star():
    """glob_match with ** patterns for recursive matching."""
    assert glob_match('src/main.py', '**/*.py') is True
    assert glob_match('deep/nested/dir/file.py', '**/*.py') is True
    assert glob_match('file.py', '**/*.py') is True
    assert glob_match('src/main.txt', '**/*.py') is False


def test_glob_match_double_star_prefix():
    """glob_match with **/ prefix."""
    assert glob_match('a/b/c.txt', '**/c.txt') is True
    assert glob_match('c.txt', '**/c.txt') is True


def test_glob_match_double_star_suffix():
    """glob_match with ** at end."""
    assert glob_match('src/foo/bar', 'src/**') is True


def test_glob_match_question_mark():
    """glob_match with ? wildcard."""
    assert glob_match('test.py', 'tes?.py') is True
    assert glob_match('test.py', 'te??.py') is True
    assert glob_match('test.py', 't???.py') is True  # t + 3 chars (est) + .py
    assert glob_match('test.py', 't????.py') is False  # needs 4 chars between t and .py


def test_build_read_file_cmd_default():
    cmd = build_read_file_cmd('test.txt')
    assert 'awk' in cmd
    assert "'test.txt'" in cmd
    assert 'NR>=1' in cmd
    assert 'NR<=2000' in cmd


def test_build_read_file_cmd_with_offset():
    cmd = build_read_file_cmd('file.py', offset=10, limit=50)
    assert 'NR>=11' in cmd
    assert 'NR<=60' in cmd
    assert "'file.py'" in cmd


def test_build_read_file_cmd_continuation_hint():
    """build_read_file_cmd includes a continuation hint in the awk END block."""
    cmd = build_read_file_cmd('file.py', offset=0, limit=10)
    assert 'more lines' in cmd
    assert 'offset=10' in cmd


def test_build_grep_cmd_content():
    cmd = build_grep_cmd('pattern')
    assert 'grep -rI' in cmd
    assert '-n' in cmd
    assert "'pattern'" in cmd
    assert "'.'" in cmd


def test_build_grep_cmd_files_with_matches():
    cmd = build_grep_cmd('pat', output_mode='files_with_matches')
    assert '-l' in cmd
    assert '-n' not in cmd


def test_build_grep_cmd_count():
    cmd = build_grep_cmd('pat', output_mode='count')
    assert '-c' in cmd


def test_build_grep_cmd_with_path():
    cmd = build_grep_cmd('pat', path='src')
    assert "'src'" in cmd


def test_build_grep_cmd_with_glob_pattern():
    """glob_pattern is shell-escaped to prevent injection."""
    cmd = build_grep_cmd('pat', glob_pattern='*.py')
    assert '--include' in cmd
    assert "'*.py'" in cmd


def test_build_grep_cmd_glob_pattern_escaping():
    """Verify glob_pattern with special chars is properly shell-escaped."""
    cmd = build_grep_cmd('pat', glob_pattern='*.py')
    # The glob pattern should be shell-escaped (wrapped in single quotes)
    assert "--include '*.py'" in cmd

    # Even a malicious glob_pattern gets safely escaped
    cmd2 = build_grep_cmd('pat', glob_pattern='$(evil)')
    assert '$(evil)' not in cmd2.replace("'$(evil)'", '')  # Only appears inside quotes


def test_build_glob_cmd():
    cmd = build_glob_cmd('*.py')
    assert 'find' in cmd
    assert "'*.py'" in cmd
    assert "'.'" in cmd


def test_build_glob_cmd_with_path():
    cmd = build_glob_cmd('*.py', path='src')
    assert "'src'" in cmd


def test_parse_glob_output_empty():
    assert parse_glob_output('') == []
    assert parse_glob_output('  ') == []
    assert parse_glob_output('\n') == []


def test_parse_glob_output_multiline():
    assert parse_glob_output('a.py\nb.py\nc.py\n') == ['a.py', 'b.py', 'c.py']


def test_filter_grep_count_output():
    text = 'a.py:3\nb.py:0\nc.py:1'
    result = filter_grep_count_output(text)
    assert result == 'a.py:3\nc.py:1'


def test_filter_grep_count_output_all_zero():
    text = 'a.py:0\nb.py:0'
    result = filter_grep_count_output(text)
    assert result == ''


def test_apply_edit_basic():
    new_text, count = apply_edit('hello world', 'world', 'earth', 'test.txt', replace_all=False)
    assert new_text == 'hello earth'
    assert count == 1


def test_apply_edit_replace_all():
    new_text, count = apply_edit('aaa bbb aaa', 'aaa', 'xxx', 'test.txt', replace_all=True)
    assert new_text == 'xxx bbb xxx'
    assert count == 2


def test_apply_edit_not_found():
    with pytest.raises(ValueError, match='not found'):
        apply_edit('hello', 'missing', 'x', 'test.txt', replace_all=False)


def test_apply_edit_ambiguous():
    with pytest.raises(ValueError, match='2 times'):
        apply_edit('aa bb aa', 'aa', 'x', 'test.txt', replace_all=False)


# --- LocalEnvironment: additional edge cases ---


async def test_local_execute_no_timeout(tmp_path: Path):
    """execute() with timeout=None completes without timeout."""
    async with LocalEnvironment(tmp_path) as env:
        result = await env.shell('echo no_timeout', timeout=None)
        assert result.exit_code == 0
        assert 'no_timeout' in result.output


async def test_local_read_file_bytes_directory(tmp_path: Path):
    """read_file_bytes on a directory raises FileNotFoundError."""
    async with LocalEnvironment(tmp_path) as env:
        (tmp_path / 'adir').mkdir()
        with pytest.raises(FileNotFoundError, match='is a directory'):
            await env.read_file('adir')


async def test_local_read_file_bytes_nonexistent(tmp_path: Path):
    """read_file_bytes on a nonexistent file raises FileNotFoundError."""
    async with LocalEnvironment(tmp_path) as env:
        with pytest.raises(FileNotFoundError):
            await env.read_file('nope.bin')


async def test_local_grep_specific_file(tmp_path: Path):
    """grep targeting a specific file works."""
    async with LocalEnvironment(tmp_path) as env:
        await env.write_file('target.py', 'findme = True\n')
        await env.write_file('other.py', 'findme = False\n')

        result = await env.grep('findme', path='target.py')
        assert 'target.py' in result
        assert 'other.py' not in result


# --- MemoryEnvironment: additional edge cases ---


async def test_memory_normalize_paths():
    """MemoryEnvironment normalizes paths correctly."""
    async with MemoryEnvironment() as env:
        await env.write_file('./test.txt', 'content')
        content = await env.read_file('test.txt')
        assert isinstance(content, str)
        assert 'content' in content


async def test_memory_normalize_leading_slash():
    """MemoryEnvironment strips leading slashes."""
    async with MemoryEnvironment() as env:
        await env.write_file('/test.txt', 'content')
        content = await env.read_file('test.txt')
        assert isinstance(content, str)
        assert 'content' in content


async def test_memory_read_file_text():
    """read_file on text file returns formatted string."""
    env = MemoryEnvironment(files={'text.txt': 'hello'})
    async with env:
        result = await env.read_file('text.txt')
        assert isinstance(result, str)
        assert 'hello' in result


async def test_memory_read_file_not_found():
    """read_file on missing file raises FileNotFoundError."""
    async with MemoryEnvironment() as env:
        with pytest.raises(FileNotFoundError):
            await env.read_file('missing.txt')


async def test_memory_edit_binary():
    """edit_file works on binary content."""
    env = MemoryEnvironment(files={'data.txt': b'hello world'})
    async with env:
        count = await env.replace_str('data.txt', 'world', 'earth')
        assert count == 1


async def test_memory_grep_exact_path():
    """grep with path= targeting an exact file."""
    env = MemoryEnvironment(
        files={
            'src/a.py': 'target\n',
            'src/b.py': 'target\n',
        }
    )
    async with env:
        result = await env.grep('target', path='src/a.py')
        assert 'src/a.py' in result
        assert 'src/b.py' not in result


async def test_memory_grep_no_text_content():
    """grep with text bytes (non-binary) works."""
    env = MemoryEnvironment(files={'data.txt': b'findme in bytes'})
    async with env:
        result = await env.grep('findme')
        assert 'data.txt' in result


async def test_memory_glob_recursive():
    """glob with ** pattern."""
    env = MemoryEnvironment(
        files={
            'src/a.py': '',
            'src/sub/b.py': '',
            'other.txt': '',
        }
    )
    async with env:
        matches = await env.glob('**/*.py')
        assert 'src/a.py' in matches
        assert 'src/sub/b.py' in matches
        assert 'other.txt' not in matches


async def test_memory_glob_in_subdirectory():
    """glob with path= restricts to subdirectory."""
    env = MemoryEnvironment(
        files={
            'src/a.py': '',
            'lib/b.py': '',
        }
    )
    async with env:
        matches = await env.glob('*.py', path='src')
        assert 'src/a.py' in matches
        assert 'lib/b.py' not in matches


async def test_memory_ls_with_bytes():
    """ls reports size correctly for bytes content."""
    env = MemoryEnvironment(files={'data.bin': b'\x00\x01\x02'})
    async with env:
        entries = await env.ls('.')
        assert len(entries) == 1
        assert entries[0].size == 3
        assert entries[0].is_dir is False


# --- ExecutionEnvironmentToolset: additional coverage ---


async def test_toolset_bash_truncated(tmp_path: Path):
    """bash tool truncation message when output exceeds limit."""
    env = LocalEnvironment(tmp_path)
    toolset = ExecutionEnvironmentToolset(env)
    ctx = build_run_context(None)
    manager = await ToolManager[None](toolset).for_run_step(ctx)

    async with env:
        # Generate output longer than MAX_OUTPUT_CHARS (100_000)
        result = await manager.handle_call(
            ToolCallPart(tool_name='shell', args={'command': 'python3 -c "print(\'x\' * 200000)"'})
        )
        assert '[output truncated]' in str(result)
        assert 'Exit code: 0' in str(result)


async def test_toolset_image_too_large(tmp_path: Path):
    """read_file on an image that's too large returns error string."""
    env = LocalEnvironment(tmp_path)
    toolset = ExecutionEnvironmentToolset(env, max_image_bytes=10)  # Very small limit
    ctx = build_run_context(None)
    manager = await ToolManager[None](toolset).for_run_step(ctx)

    async with env:
        # Write a PNG file that exceeds the limit
        await env.write_file('big.png', b'\x89PNG\r\n\x1a\n' + b'\x00' * 100)
        result = await manager.handle_call(ToolCallPart(tool_name='read_file', args={'path': 'big.png'}))
        assert 'Image too large' in str(result)


async def test_toolset_image_read(tmp_path: Path):
    """read_file on an image returns BinaryContent."""
    from pydantic_ai.messages import BinaryContent

    env = LocalEnvironment(tmp_path)
    toolset = ExecutionEnvironmentToolset(env)
    ctx = build_run_context(None)
    manager = await ToolManager[None](toolset).for_run_step(ctx)

    async with env:
        png_data = (
            b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01'
            b'\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89'
            b'\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01'
            b'\r\n\xb4\x00\x00\x00\x00IEND\xaeB`\x82'
        )
        await env.write_file('img.png', png_data)
        result = await manager.handle_call(ToolCallPart(tool_name='read_file', args={'path': 'img.png'}))
        assert isinstance(result, BinaryContent)
        assert result.media_type == 'image/png'


async def test_toolset_grep_no_matches(tmp_path: Path):
    """grep with no matches returns 'No matches found.'."""
    env = LocalEnvironment(tmp_path)
    toolset = ExecutionEnvironmentToolset(env)
    ctx = build_run_context(None)
    manager = await ToolManager[None](toolset).for_run_step(ctx)

    async with env:
        await env.write_file('test.txt', 'nothing relevant\n')
        result = await manager.handle_call(ToolCallPart(tool_name='grep', args={'pattern': 'nonexistent_xyz'}))
        assert result == snapshot('No matches found.')


async def test_toolset_glob_no_matches(tmp_path: Path):
    """glob with no matches returns 'No files found.'."""
    env = LocalEnvironment(tmp_path)
    toolset = ExecutionEnvironmentToolset(env)
    ctx = build_run_context(None)
    manager = await ToolManager[None](toolset).for_run_step(ctx)

    async with env:
        result = await manager.handle_call(ToolCallPart(tool_name='glob', args={'pattern': '*.nonexistent'}))
        assert result == snapshot('No files found.')


async def test_toolset_edit_success(tmp_path: Path):
    """edit_file tool returns success message."""
    env = LocalEnvironment(tmp_path)
    toolset = ExecutionEnvironmentToolset(env)
    ctx = build_run_context(None)
    manager = await ToolManager[None](toolset).for_run_step(ctx)

    async with env:
        await env.write_file('code.py', 'old_value = 1\n')
        result = await manager.handle_call(
            ToolCallPart(
                tool_name='replace_str',
                args={'path': 'code.py', 'old': 'old_value', 'new': 'new_value'},
            )
        )
        assert result == snapshot('Replaced 1 occurrence in code.py.')


async def test_toolset_with_custom_env_instructions():
    """Environment instructions is used per-tool."""

    class CustomEnv(MemoryEnvironment):
        def instructions(self, capability: str) -> str | None:
            if capability == 'grep':
                return 'Custom grep description.'
            return None

    env = CustomEnv()
    assert env.instructions('grep') == 'Custom grep description.'
    assert env.instructions('read_file') is None


async def test_toolset_lifecycle_ref_counting(tmp_path: Path):
    """Multiple context manager entries share the environment."""
    env = LocalEnvironment(tmp_path)
    toolset = ExecutionEnvironmentToolset(env)

    async with toolset:
        async with toolset:
            # Both entries active
            result = await env.shell('echo shared')
            assert 'shared' in result.output
        # Still alive after one exit
        result = await env.shell('echo still_alive')
        assert 'still_alive' in result.output


# --- Additional coverage: _base.py ---


async def test_glob_match_question_mark_in_doublestar_pattern():
    """glob_match with ? inside a ** pattern."""
    assert glob_match('a/b/test.py', '**/?est.py') is True
    assert glob_match('test.py', '?est.py') is True


async def test_execution_environment_aenter_aexit():
    """ExecutionEnvironment base __aenter__/__aexit__ are exercised by subclasses."""
    # MemoryEnvironment exercises the base class path
    env = MemoryEnvironment()
    async with env:
        pass


# --- Additional coverage: _toolset.py ---


async def test_toolset_bash_empty_output(tmp_path: Path):
    """ExecutionEnvironmentToolset bash returns just exit code when no output."""
    env = LocalEnvironment(tmp_path)
    toolset = ExecutionEnvironmentToolset(env)
    ctx = build_run_context()
    manager = await ToolManager[None](toolset).for_run_step(ctx)

    async with env:
        result = await manager.handle_call(ToolCallPart(tool_name='shell', args={'command': 'true'}))
        assert 'Exit code: 0' in str(result)


async def test_toolset_glob_truncation(tmp_path: Path):
    """ExecutionEnvironmentToolset glob truncates after 100 matches."""
    env = LocalEnvironment(tmp_path)
    # Create 110 files
    for i in range(110):
        (tmp_path / f'file_{i:03d}.txt').write_text(f'content {i}')

    toolset = ExecutionEnvironmentToolset(env)
    ctx = build_run_context()
    manager = await ToolManager[None](toolset).for_run_step(ctx)

    async with env:
        result = await manager.handle_call(ToolCallPart(tool_name='glob', args={'pattern': '*.txt'}))
        assert 'truncated' in str(result)


async def test_toolset_grep_no_matches_returns_message(tmp_path: Path):
    """ExecutionEnvironmentToolset grep returns message when no matches."""
    (tmp_path / 'test.txt').write_text('hello world')
    env = LocalEnvironment(tmp_path)
    toolset = ExecutionEnvironmentToolset(env)
    ctx = build_run_context()
    manager = await ToolManager[None](toolset).for_run_step(ctx)

    async with env:
        result = await manager.handle_call(ToolCallPart(tool_name='grep', args={'pattern': 'zzz_nonexistent'}))
        assert 'No matches' in str(result)


async def test_toolset_lifecycle_error(tmp_path: Path):
    """ExecutionEnvironmentToolset handles environment startup failures."""

    class FailingEnv(LocalEnvironment):
        async def __aenter__(self):
            raise RuntimeError('Setup failed')

    env = FailingEnv(tmp_path)
    toolset = ExecutionEnvironmentToolset(env)
    with pytest.raises(RuntimeError, match='Setup failed'):
        async with toolset:
            pass


# --- Additional coverage: local.py ---


async def test_local_process_stdin_not_available():
    """LocalEnvironmentProcess.send raises when stdin is None."""
    from pydantic_harness.environments.local import LocalEnvironmentProcess

    mock_proc = MagicMock()
    mock_proc.stdin = None
    proc = LocalEnvironmentProcess(mock_proc)
    with pytest.raises(RuntimeError, match='stdin'):
        await proc.send(b'data')


async def test_local_process_stdout_not_available():
    """LocalEnvironmentProcess.recv raises when stdout is None."""
    from pydantic_harness.environments.local import LocalEnvironmentProcess

    mock_proc = MagicMock()
    mock_proc.stdout = None
    proc = LocalEnvironmentProcess(mock_proc)
    with pytest.raises(RuntimeError, match='stdout'):
        await proc.recv()


async def test_local_process_stderr_not_available():
    """LocalEnvironmentProcess.recv_stderr raises when stderr is None."""
    from pydantic_harness.environments.local import LocalEnvironmentProcess

    mock_proc = MagicMock()
    mock_proc.stderr = None
    proc = LocalEnvironmentProcess(mock_proc)
    with pytest.raises(RuntimeError, match='stderr'):
        await proc.recv_stderr()


async def test_local_process_recv_stderr_timeout(tmp_path: Path):
    """LocalEnvironmentProcess.recv_stderr with timeout."""
    env = LocalEnvironment(tmp_path)
    proc = await env.create_process('python -c "import sys; sys.stderr.write(\'err\\n\')"')
    async with proc:
        data = await proc.recv_stderr(timeout=5.0)
        assert b'err' in data


async def test_local_process_recv_stderr_eof(tmp_path: Path):
    """LocalEnvironmentProcess.recv_stderr returns empty on EOF."""
    env = LocalEnvironment(tmp_path)
    proc = await env.create_process('echo done')
    async with proc:
        await proc.wait(timeout=5.0)
        # After process exits, stderr should return empty
        data = await proc.recv_stderr()
        assert data == b''


async def test_local_process_kill_terminates_sleep(tmp_path: Path):
    """LocalEnvironmentProcess.kill terminates process."""
    env = LocalEnvironment(tmp_path)
    proc = await env.create_process('sleep 60')
    async with proc:
        await proc.kill()
        # After kill, returncode should be set


async def test_local_read_file_bytes_directory_raises_error(tmp_path: Path):
    """LocalEnvironment.read_file_bytes raises on directory."""
    (tmp_path / 'subdir').mkdir()
    env = LocalEnvironment(tmp_path)
    with pytest.raises(FileNotFoundError, match='directory'):
        await env.read_file('subdir')


async def test_local_read_file_bytes_not_found(tmp_path: Path):
    """LocalEnvironment.read_file_bytes raises on missing file."""
    env = LocalEnvironment(tmp_path)
    with pytest.raises(FileNotFoundError, match='not found'):
        await env.read_file('nonexistent.txt')


async def test_local_grep_on_file(tmp_path: Path):
    """LocalEnvironment.grep on a specific file path."""
    (tmp_path / 'target.py').write_text('found = True\nmissed = False\n')
    env = LocalEnvironment(tmp_path)
    result = await env.grep('found', path='target.py')
    assert 'found' in result
    assert 'missed' not in result


async def test_local_grep_with_glob_pattern_filters_by_extension(tmp_path: Path):
    """LocalEnvironment.grep with glob filtering."""
    (tmp_path / 'a.py').write_text('match_here\n')
    (tmp_path / 'b.txt').write_text('match_here\n')
    env = LocalEnvironment(tmp_path)
    result = await env.grep('match_here', glob_pattern='*.py')
    assert 'a.py' in result
    assert 'b.txt' not in result


async def test_local_grep_skips_binary_files_with_null_bytes(tmp_path: Path):
    """LocalEnvironment.grep skips files with null bytes."""
    (tmp_path / 'binary.bin').write_bytes(b'\x00binary content')
    (tmp_path / 'text.txt').write_text('searchable\n')
    env = LocalEnvironment(tmp_path)
    result = await env.grep('searchable')
    assert 'text.txt' in result
    assert 'binary' not in result


async def test_local_grep_skips_hidden_files_in_hidden_dirs(tmp_path: Path):
    """LocalEnvironment.grep skips hidden files/dirs."""
    hidden_dir = tmp_path / '.hidden'
    hidden_dir.mkdir()
    (hidden_dir / 'secret.txt').write_text('findme\n')
    (tmp_path / 'visible.txt').write_text('findme\n')
    env = LocalEnvironment(tmp_path)
    result = await env.grep('findme')
    assert 'visible.txt' in result
    assert '.hidden' not in result


# --- Base class run_python tests ---


async def test_base_run_python_success():
    """Base ExecutionEnvironment.run_python writes code to file and runs via shell."""
    from pydantic_harness.environments._base import Capability as EnvCapability, ExecutionEnvironment as BaseEnv

    class _ShellEnv(BaseEnv):
        """Env that records write_file/shell calls and returns canned results."""

        written: dict[str, str | bytes] = {}

        @property
        def capabilities(self) -> frozenset[EnvCapability]:
            return frozenset({'shell', 'write_file', 'run_python'})  # pragma: no cover

        async def write_file(self, path: str, content: str | bytes) -> None:
            self.written[path] = content

        async def shell(
            self, command: str, *, timeout: float | None = 120, env: dict[str, str] | None = None
        ) -> ExecutionResult:
            return ExecutionResult(output='hello world\n', exit_code=0)

    env = _ShellEnv()
    result = await env.run_python('print("hello world")')
    assert result == 'hello world\n'
    assert env.written['/tmp/_pydantic_ai_code.py'] == 'print("hello world")'


async def test_base_run_python_error():
    """Base ExecutionEnvironment.run_python raises CodeRuntimeError on non-zero exit."""
    from pydantic_harness.environments._base import Capability as EnvCapability, ExecutionEnvironment as BaseEnv
    from pydantic_harness.toolsets.code_execution._abstract import CodeRuntimeError

    class _ShellEnv(BaseEnv):
        @property
        def capabilities(self) -> frozenset[EnvCapability]:
            return frozenset({'shell', 'write_file', 'run_python'})  # pragma: no cover

        async def write_file(self, path: str, content: str | bytes) -> None:
            pass

        async def shell(
            self, command: str, *, timeout: float | None = 120, env: dict[str, str] | None = None
        ) -> ExecutionResult:
            return ExecutionResult(output='Exception: fail\n', exit_code=1)

    env = _ShellEnv()
    with pytest.raises(CodeRuntimeError, match='Exception: fail'):
        await env.run_python('raise Exception("fail")')


# --- Local environment additional tests ---


async def test_local_execute_output_truncation(tmp_path: Path):
    """LocalEnvironment.execute truncates long output."""
    script = tmp_path / 'big.py'
    script.write_text("print('x' * 200000)")
    env = LocalEnvironment(tmp_path)
    result = await env.shell(f'python3 {script}')
    assert result.truncated is True
    assert len(result.output) == 100_000


async def test_local_process_wait_no_timeout(tmp_path: Path):
    """LocalEnvironmentProcess.wait without timeout."""
    env = LocalEnvironment(tmp_path)
    proc = await env.create_process('true')
    async with proc:
        exit_code = await proc.wait()  # no timeout
        assert exit_code == 0


# --- Memory environment additional tests ---


async def test_memory_normalize_leading_slash_in_constructor():
    """MemoryEnvironment normalizes paths with leading /."""
    env = MemoryEnvironment(files={'/abs/path.txt': 'content'})
    content = await env.read_file('abs/path.txt')
    assert isinstance(content, str)
    assert 'content' in content


async def test_memory_read_file_directory_error():
    """MemoryEnvironment.read_file raises on directory paths."""
    env = MemoryEnvironment(files={'dir/file.txt': 'content'})
    with pytest.raises(FileNotFoundError, match='directory'):
        await env.read_file('dir')


async def test_memory_read_file_bytes_not_found_raises_error():
    """MemoryEnvironment.read_file raises on missing file."""
    env = MemoryEnvironment()
    with pytest.raises(FileNotFoundError):
        await env.read_file('missing.txt')


async def test_memory_ls_non_root_directory():
    """MemoryEnvironment.ls lists files in a subdirectory."""
    env = MemoryEnvironment(files={'sub/a.txt': 'a', 'sub/b.txt': 'b', 'other.txt': 'c'})
    entries = await env.ls('sub')
    assert len(entries) == 2
    names = {e.name for e in entries}
    assert names == {'a.txt', 'b.txt'}


async def test_memory_ls_with_subdirs():
    """MemoryEnvironment.ls shows directories in listing."""
    env = MemoryEnvironment(files={'dir/sub/file.txt': 'content'})
    entries = await env.ls('dir')
    assert len(entries) == 1
    assert entries[0].name == 'sub'
    assert entries[0].is_dir is True


async def test_memory_ls_skips_non_children():
    """MemoryEnvironment.ls skips files not under the directory."""
    env = MemoryEnvironment(files={'a/b.txt': 'x', 'c/d.txt': 'y'})
    entries = await env.ls('a')
    assert len(entries) == 1
    assert entries[0].name == 'b.txt'


async def test_memory_grep_binary_skip():
    """MemoryEnvironment.grep skips binary files."""
    env = MemoryEnvironment(files={'binary.bin': b'\x00binary data', 'text.txt': 'findme'})
    result = await env.grep('findme')
    assert 'text.txt' in result
    assert 'binary' not in result


async def test_memory_grep_path_filter():
    """MemoryEnvironment.grep filters by exact file path."""
    env = MemoryEnvironment(files={'sub/target.py': 'match_here', 'other.py': 'match_here'})
    result = await env.grep('match_here', path='sub')
    assert 'sub/target.py' in result
    assert 'other.py' not in result


async def test_memory_glob_in_subdirectory_with_path_filter():
    """MemoryEnvironment.glob works with path parameter."""
    env = MemoryEnvironment(files={'src/a.py': 'a', 'src/b.txt': 'b', 'other.py': 'c'})
    matches = await env.glob('*.py', path='src')
    assert 'src/a.py' in matches
    assert 'other.py' not in matches


async def test_memory_normalize_absolute_path():
    """MemoryEnvironment._normalize strips leading /."""
    env = MemoryEnvironment(files={'path.txt': 'content'})
    normalized = env._normalize('/path.txt')
    assert normalized == 'path.txt'


async def test_memory_read_file_that_is_also_directory_prefix():
    """MemoryEnvironment.read_file when path exists as both file and directory prefix."""
    env = MemoryEnvironment(files={'dir': 'I am a file', 'dir/child.txt': 'child content'})
    async with env:
        content = await env.read_file('dir')
        assert isinstance(content, str)
        assert 'I am a file' in content


async def test_memory_read_image_stored_as_string():
    """MemoryEnvironment returns bytes for image files even when stored as a string."""
    env = MemoryEnvironment(files={'image.png': 'fake png data'})
    async with env:
        result = await env.read_file('image.png')
    assert isinstance(result, bytes)
    assert result == b'fake png data'


# --- ExecutionEnvironmentToolset resolution tests ---


def test_resolve_edit_tool_explicit_strategy():
    """Passing edit_strategy to constructor overrides auto-detection."""
    env = MemoryEnvironment()
    toolset = ExecutionEnvironmentToolset(env, edit_strategy='apply_patch')
    strategy = toolset._resolve_edit_tool(env)
    assert strategy == 'apply_patch'


def test_resolve_edit_tool_apply_patch_fallback():
    """When env has apply_patch but not replace_str, resolves to apply_patch."""
    from pydantic_harness.environments._base import Capability as EnvCapability, ExecutionEnvironment as BaseEnv

    class _ApplyPatchEnv(BaseEnv):
        @property
        def capabilities(self) -> frozenset[EnvCapability]:
            return frozenset({'apply_patch'})

    toolset = ExecutionEnvironmentToolset(_ApplyPatchEnv())
    strategy = toolset._resolve_edit_tool(_ApplyPatchEnv())
    assert strategy == 'apply_patch'


def test_resolve_edit_tool_neither():
    """When env has neither replace_str nor apply_patch, returns None."""
    from pydantic_harness.environments._base import Capability as EnvCapability, ExecutionEnvironment as BaseEnv

    class _NoEditEnv(BaseEnv):
        @property
        def capabilities(self) -> frozenset[EnvCapability]:
            return frozenset({'ls'})

    toolset = ExecutionEnvironmentToolset(_NoEditEnv())
    strategy = toolset._resolve_edit_tool(_NoEditEnv())
    assert strategy is None


def test_resolve_capabilities_with_run_code_with_functions():
    """Env with run_python_with_functions maps to run_code_with_functions capability."""
    from pydantic_harness.environments._base import Capability as EnvCapability, ExecutionEnvironment as BaseEnv

    class _FunctionsEnv(BaseEnv):
        @property
        def capabilities(self) -> frozenset[EnvCapability]:
            return frozenset({'run_python_with_functions'})

    toolset = ExecutionEnvironmentToolset(
        _FunctionsEnv(),
        exclude=frozenset(),  # don't exclude run_code
    )
    caps = toolset._resolve_capabilities(_FunctionsEnv())
    assert 'run_code_with_functions' in caps


# --- Toolset ls formatting tests ---


async def test_toolset_ls_formats_dirs():
    """Toolset ls formats directory entries with trailing /."""
    env = MemoryEnvironment(files={'sub/a.txt': 'hello'})
    toolset = ExecutionEnvironmentToolset(env)
    ctx = build_run_context()
    tools = await toolset.get_tools(ctx)
    async with env:
        result = await toolset.call_tool('ls', {'path': '.'}, ctx, tools['ls'])
    assert 'sub/' in str(result)


async def test_toolset_ls_error_handling():
    """Toolset ls returns error string when environment raises."""
    from pydantic_harness.environments._base import Capability as EnvCapability, ExecutionEnvironment as BaseEnv

    class _ErrorLsEnv(BaseEnv):
        @property
        def capabilities(self) -> frozenset[EnvCapability]:
            return frozenset({'ls'})

        async def ls(self, path: str = '.') -> list[FileInfo]:
            raise NotADirectoryError(f'Not a directory: {path}')

    env = _ErrorLsEnv()
    toolset = ExecutionEnvironmentToolset(env)
    ctx = build_run_context()
    tools = await toolset.get_tools(ctx)
    result = await toolset.call_tool('ls', {'path': '/bad'}, ctx, tools['ls'])
    assert 'Error:' in str(result)


async def test_toolset_ls_formats_files_without_size():
    """Toolset ls formats file entries without size (just the name)."""
    from pydantic_harness.environments._base import Capability as EnvCapability, ExecutionEnvironment as BaseEnv

    class _NoSizeEnv(BaseEnv):
        @property
        def capabilities(self) -> frozenset[EnvCapability]:
            return frozenset({'ls'})

        async def ls(self, path: str = '.') -> list[FileInfo]:
            return [FileInfo(name='readme.txt', path='readme.txt', is_dir=False, size=None)]

    env = _NoSizeEnv()
    toolset = ExecutionEnvironmentToolset(env)
    ctx = build_run_context()
    tools = await toolset.get_tools(ctx)
    result = await toolset.call_tool('ls', {'path': '.'}, ctx, tools['ls'])
    assert str(result) == 'readme.txt'


async def test_toolset_ls_empty_directory():
    """Toolset ls returns 'Empty directory.' for empty listings."""
    from pydantic_harness.environments._base import Capability as EnvCapability, ExecutionEnvironment as BaseEnv

    class _EmptyLsEnv(BaseEnv):
        @property
        def capabilities(self) -> frozenset[EnvCapability]:
            return frozenset({'ls'})

        async def ls(self, path: str = '.') -> list[FileInfo]:
            return []

    env = _EmptyLsEnv()
    toolset = ExecutionEnvironmentToolset(env)
    ctx = build_run_context()
    tools = await toolset.get_tools(ctx)
    result = await toolset.call_tool('ls', {'path': '.'}, ctx, tools['ls'])
    assert str(result) == 'Empty directory.'


# --- Lazy import test ---


def test_lazy_import_code_execution_toolset():
    """CodeExecutionToolset is importable via pydantic_harness.toolsets."""
    from pydantic_harness.toolsets import CodeExecutionToolset

    assert CodeExecutionToolset is not None
