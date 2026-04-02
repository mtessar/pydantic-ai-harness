"""Tests for the Shell capability."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from pydantic_harness.shell import Shell

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_cwd(tmp_path: Path) -> Path:
    """Create a temporary working directory."""
    (tmp_path / 'greeting.txt').write_text('hello world\n')
    return tmp_path


@pytest.fixture
def sh(tmp_cwd: Path) -> Shell:
    """A Shell capability rooted at the test directory."""
    return Shell(cwd=tmp_cwd)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


class TestConfig:
    def test_defaults(self) -> None:
        sh = Shell()
        assert sh.default_timeout == 30.0
        assert sh.max_output_chars == 10_000

    def test_cannot_mix_allow_deny(self) -> None:
        with pytest.raises(ValueError, match='not both'):
            Shell(allowed_commands=['ls'], denied_commands=['rm'])


# ---------------------------------------------------------------------------
# Command validation
# ---------------------------------------------------------------------------


class TestCommandValidation:
    def test_denied_command(self) -> None:
        sh = Shell(denied_commands=['rm'])
        with pytest.raises(PermissionError, match='denied'):
            sh.check_command('rm -rf /')

    def test_allowed_command(self) -> None:
        sh = Shell(allowed_commands=['echo', 'cat'])
        sh.check_command('echo hello')  # should not raise
        with pytest.raises(PermissionError, match='not in the allowed'):
            sh.check_command('rm -rf /')

    def test_no_restrictions(self) -> None:
        sh = Shell()
        sh.check_command('anything goes')  # should not raise

    def test_malformed_command(self) -> None:
        sh = Shell(denied_commands=['rm'])
        # Unterminated quote: shlex.split raises ValueError.
        # The capability falls through and lets the shell handle it.
        sh.check_command("echo 'unterminated")  # should not raise

    def test_empty_command(self) -> None:
        sh = Shell(allowed_commands=['echo'])
        sh.check_command('')  # empty string should not raise


# ---------------------------------------------------------------------------
# Output truncation
# ---------------------------------------------------------------------------


class TestTruncation:
    def test_short_output(self) -> None:
        sh = Shell(max_output_chars=100)
        assert sh.truncate('short') == 'short'

    def test_long_output(self) -> None:
        sh = Shell(max_output_chars=10)
        result = sh.truncate('x' * 50)
        assert len(result.splitlines()[0]) == 10
        assert 'truncated' in result


# ---------------------------------------------------------------------------
# Command execution
# ---------------------------------------------------------------------------


class TestRunCommand:
    @pytest.mark.anyio
    async def test_echo(self, sh: Shell) -> None:
        result = await sh.run_command('echo hello')
        assert '[stdout]' in result
        assert 'hello' in result

    @pytest.mark.anyio
    async def test_stderr_label(self, sh: Shell) -> None:
        result = await sh.run_command('echo oops >&2')
        assert '[stderr]' in result
        assert 'oops' in result

    @pytest.mark.anyio
    async def test_stdout_and_stderr(self, sh: Shell) -> None:
        result = await sh.run_command(
            f"{sys.executable} -c \"import sys; print('out'); print('err', file=sys.stderr)\""
        )
        assert '[stdout]' in result
        assert '[stderr]' in result
        assert 'out' in result
        assert 'err' in result

    @pytest.mark.anyio
    async def test_exit_code(self, sh: Shell) -> None:
        result = await sh.run_command('exit 1')
        assert 'exit code: 1' in result

    @pytest.mark.anyio
    async def test_timeout(self) -> None:
        sh = Shell()
        result = await sh.run_command('sleep 10', timeout_seconds=0.1)
        assert 'timed out' in result.lower()

    @pytest.mark.anyio
    async def test_cwd(self, sh: Shell) -> None:
        result = await sh.run_command('cat greeting.txt')
        assert 'hello world' in result

    @pytest.mark.anyio
    async def test_truncated_output(self, tmp_cwd: Path) -> None:
        sh = Shell(cwd=tmp_cwd, max_output_chars=20)
        result = await sh.run_command(f'{sys.executable} -c "print(\'x\' * 100)"')
        assert 'truncated' in result

    @pytest.mark.anyio
    async def test_denied_command_async(self) -> None:
        sh = Shell(denied_commands=['rm'])
        with pytest.raises(PermissionError, match='denied'):
            await sh.run_command('rm -rf /')

    @pytest.mark.anyio
    async def test_allowed_command_async(self) -> None:
        sh = Shell(allowed_commands=['echo'])
        result = await sh.run_command('echo works')
        assert 'works' in result
        with pytest.raises(PermissionError, match='not in the allowed'):
            await sh.run_command('cat /etc/passwd')

    @pytest.mark.anyio
    async def test_empty_output(self, sh: Shell) -> None:
        result = await sh.run_command('true')
        assert result == ''


# ---------------------------------------------------------------------------
# Persistent working directory
# ---------------------------------------------------------------------------


class TestPersistCwd:
    @pytest.mark.anyio
    async def test_cd_updates_cwd(self, tmp_cwd: Path) -> None:
        subdir = tmp_cwd / 'subdir'
        subdir.mkdir()
        (subdir / 'marker.txt').write_text('found\n')
        sh = Shell(cwd=tmp_cwd, persist_cwd=True)
        await sh.run_command('cd subdir')
        result = await sh.run_command('cat marker.txt')
        assert 'found' in result

    @pytest.mark.anyio
    async def test_cd_disabled_by_default(self, tmp_cwd: Path) -> None:
        subdir = tmp_cwd / 'subdir'
        subdir.mkdir()
        (subdir / 'marker.txt').write_text('found\n')
        sh = Shell(cwd=tmp_cwd)
        await sh.run_command('cd subdir')
        # Without persist_cwd, cwd should not change
        result = await sh.run_command('cat marker.txt')
        assert 'exit code' in result  # cat fails because we're in the wrong dir

    @pytest.mark.anyio
    async def test_cd_chained_command(self, tmp_cwd: Path) -> None:
        subdir = tmp_cwd / 'sub'
        subdir.mkdir()
        sh = Shell(cwd=tmp_cwd, persist_cwd=True)
        await sh.run_command('cd sub && echo hi')
        assert sh._cwd == subdir

    @pytest.mark.anyio
    async def test_cd_quoted_path(self, tmp_cwd: Path) -> None:
        subdir = tmp_cwd / 'my dir'
        subdir.mkdir()
        sh = Shell(cwd=tmp_cwd, persist_cwd=True)
        await sh.run_command("cd 'my dir'")
        assert sh._cwd == subdir

    @pytest.mark.anyio
    async def test_cd_nonexistent_ignored(self, tmp_cwd: Path) -> None:
        sh = Shell(cwd=tmp_cwd, persist_cwd=True)
        original_cwd = sh._cwd
        await sh.run_command('cd nonexistent_dir')
        # cd failed (exit code != 0), so _cwd should not change
        assert sh._cwd == original_cwd

    @pytest.mark.anyio
    async def test_cd_not_updated_on_failure(self, tmp_cwd: Path) -> None:
        subdir = tmp_cwd / 'subdir'
        subdir.mkdir()
        sh = Shell(cwd=tmp_cwd, persist_cwd=True)
        original_cwd = sh._cwd
        # The cd itself succeeds but the second command fails
        await sh.run_command('cd subdir && false')
        assert sh._cwd == original_cwd

    @pytest.mark.anyio
    async def test_cd_absolute_path(self, tmp_cwd: Path) -> None:
        subdir = tmp_cwd / 'target'
        subdir.mkdir()
        sh = Shell(cwd=tmp_cwd, persist_cwd=True)
        await sh.run_command(f'cd {subdir}')
        assert sh._cwd == subdir

    @pytest.mark.anyio
    async def test_cd_home(self, tmp_cwd: Path) -> None:
        sh = Shell(cwd=tmp_cwd, persist_cwd=True)
        await sh.run_command('cd ~')
        assert sh._cwd == Path.home()

    @pytest.mark.anyio
    async def test_cd_home_subdir(self, tmp_cwd: Path) -> None:
        sh = Shell(cwd=tmp_cwd, persist_cwd=True)
        await sh.run_command('cd ~/.')
        assert sh._cwd == Path.home()

    def test_update_cwd_nonexistent_dir(self, tmp_cwd: Path) -> None:
        sh = Shell(cwd=tmp_cwd, persist_cwd=True)
        original = sh._cwd
        # Call _update_cwd directly with a cd to a non-existent path
        sh._update_cwd('cd does_not_exist')
        assert sh._cwd == original

    def test_extract_cd_target_none(self) -> None:
        sh = Shell()
        assert sh._extract_cd_target('echo hello') is None
        assert sh._extract_cd_target('ls -la') is None

    def test_extract_cd_target_simple(self) -> None:
        sh = Shell()
        assert sh._extract_cd_target('cd foo') == 'foo'
        assert sh._extract_cd_target('cd /tmp') == '/tmp'

    def test_extract_cd_target_with_chain(self) -> None:
        sh = Shell()
        assert sh._extract_cd_target('cd foo && ls') == 'foo'
        assert sh._extract_cd_target('cd bar; pwd') == 'bar'

    def test_extract_cd_double_quoted(self) -> None:
        sh = Shell()
        assert sh._extract_cd_target('cd "my dir"') == 'my dir'


# ---------------------------------------------------------------------------
# Toolset integration
# ---------------------------------------------------------------------------


class TestToolset:
    def test_get_toolset_returns_function_toolset(self, sh: Shell) -> None:
        from pydantic_ai.toolsets import FunctionToolset

        toolset = sh.get_toolset()
        assert isinstance(toolset, FunctionToolset)

    def test_toolset_has_run_command(self, sh: Shell) -> None:
        from pydantic_ai.toolsets import FunctionToolset

        toolset = sh.get_toolset()
        assert isinstance(toolset, FunctionToolset)
        assert set(toolset.tools.keys()) == {'run_command'}

    def test_serialization_name(self) -> None:
        assert Shell.get_serialization_name() == 'Shell'
