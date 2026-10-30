"""The agent sandboxes never let the agent read the user's secrets or shell startup files.

Every test runs against a fake home built on `tmp_path`, holding canary files where the
secrets would be: no real secret is ever read, copied or written. The profile/argv tests
run anywhere; the real-execution ones run only where their sandbox is available.

The worst case is the one tested: $HOME itself bound as the workspace (a dotfiles
repository), which brings every real file back into view unless the backend hides it.
"""

import shutil
import subprocess
import sys
from itertools import pairwise
from pathlib import Path

import pytest

from odev.plugins.odev_plugin_ai.common.sandbox.base import (
    DENY_USER_SECRETS_RELATIVE,
    ExecutionSpec,
    write_sanitized_config,
)
from odev.plugins.odev_plugin_ai.common.sandbox.bwrap import BwrapSandbox
from odev.plugins.odev_plugin_ai.common.sandbox.seatbelt import SANDBOX_EXEC, SeatbeltSandbox


CANARY = "canary-not-a-secret"

SECRET_FILES = (".zshrc", ".bashrc", ".profile", ".git-credentials", ".pgpass", ".ssh/id_ed25519")
"""Canaries written under the fake home, one per kind of secret."""


@pytest.fixture
def home(tmp_path: Path) -> Path:
    """Build a fake home holding canaries in place of the secrets, and odev's own tree."""
    home = tmp_path / "home"
    for rel in SECRET_FILES:
        (home / rel).parent.mkdir(parents=True, exist_ok=True)
        (home / rel).write_text(CANARY)
    (home / ".password-store").mkdir()
    (home / ".password-store/vault.gpg").write_text(CANARY)
    (home / ".config/gh").mkdir(parents=True)
    (home / ".config/gh/hosts.yml").write_text(CANARY)
    odev_config = home / ".config/odev"
    (odev_config / "venv/bin").mkdir(parents=True)
    (odev_config / "venv/pyvenv.cfg").write_text("home = /usr/bin\n")
    (odev_config / "plugins").mkdir()
    (odev_config / "code-review").mkdir()
    (odev_config / "code-review/state").write_text(CANARY)
    (odev_config / "odev.cfg").write_text(f"[paths]\nrepositories = /src\n\n[ai]\nllm_api_key = {CANARY}\n")
    (home / "notes.txt").write_text("not a secret")
    return home.resolve()


def make_spec(
    home: Path, tmp_path: Path, *, bind_home: bool = True, agent_dirs: list[Path] | None = None
) -> ExecutionSpec:
    """Return the spec of a run whose workspace is the whole home, with odev's venv and plugins bound."""
    playground = tmp_path / "playground"
    sandbox_tmp = tmp_path / "sandbox_tmp"
    for path in (playground, sandbox_tmp):
        path.mkdir(exist_ok=True)
    binds = [
        (home / ".config/odev/plugins", home / ".config/odev/plugins", True, False),
        (home / ".config/odev/venv", home / ".config/odev/venv", True, False),
    ]
    if bind_home:
        binds.insert(0, (home, home, False, True))
    return ExecutionSpec(
        agent_cmd=["/bin/sh"],
        final_binds=binds,
        agent_dirs=agent_dirs or [],
        agent_files=[],
        env={},
        secrets=[],
        cwd=str(home),
        playground=playground,
        sandbox_tmp=sandbox_tmp,
        proxy_dir=tmp_path / "proxy",
        pg_data_dir=tmp_path / "pgdata",
    )


def seatbelt_profile(spec: ExecutionSpec, home: Path) -> str:
    """Build the Seatbelt profile of `spec` without an odev framework behind the backend."""
    sandbox = SeatbeltSandbox.__new__(SeatbeltSandbox)
    sandbox._discover_ide_sockets = list  # the host's editor sockets are not under test
    return sandbox._build_profile(spec, home, Path("/unused"), None, Path("/unused"))


# --- odev's configuration copy ------------------------------------------------


def test_sanitized_config_drops_credentials_only(tmp_path: Path, home: Path):
    target = tmp_path / "odev.cfg"
    assert write_sanitized_config(home / ".config/odev/odev.cfg", target)
    copied = target.read_text()
    assert CANARY not in copied
    assert "llm_api_key" not in copied
    assert "repositories = /src" in copied


def test_unparsable_config_is_not_copied(tmp_path: Path):
    source = tmp_path / "broken.cfg"
    source.write_text(f"token = {CANARY}\n")  # no section header
    assert not write_sanitized_config(source, tmp_path / "copy.cfg")
    assert not (tmp_path / "copy.cfg").exists()


def test_odev_config_copy_skips_directories_and_secrets(tmp_path: Path, home: Path):
    playground = tmp_path / "playground"
    BwrapSandbox._prepare_odev_config(BwrapSandbox.__new__(BwrapSandbox), playground, home)
    copy = playground / ".config/odev"
    assert sorted(p.name for p in copy.iterdir()) == ["odev.cfg"]
    assert CANARY not in (copy / "odev.cfg").read_text()


# --- Seatbelt profile ---------------------------------------------------------


def test_seatbelt_denies_reading_every_secret(tmp_path: Path, home: Path):
    profile = seatbelt_profile(make_spec(home, tmp_path), home)
    for rel in DENY_USER_SECRETS_RELATIVE:
        assert f'(deny file-read* file-write* (subpath "{home / rel}"))' in profile


def test_seatbelt_secrets_denied_after_the_workspace_is_allowed(tmp_path: Path, home: Path):
    profile = seatbelt_profile(make_spec(home, tmp_path), home)
    allow_home = profile.index(f'(allow file-write* (subpath "{home}"))')
    deny_zshrc = profile.index(f'(deny file-read* file-write* (subpath "{home}/.zshrc"))')
    assert allow_home < deny_zshrc  # later rules win: binding $HOME does not give .zshrc back


def test_seatbelt_gives_back_only_what_is_bound_inside_a_secret(tmp_path: Path, home: Path):
    profile = seatbelt_profile(make_spec(home, tmp_path), home)
    deny = profile.index(f'(deny file-read* file-write* (subpath "{home}/.config/odev"))')
    venv = profile.index(f'(allow file-read* (subpath "{home}/.config/odev/venv"))')
    config = profile.index(f'(allow file-read* file-write* (subpath "{home}/.config/odev/odev.cfg"))')
    assert deny < venv
    assert deny < config
    assert f'(allow file-read-metadata (literal "{home}/.config/odev"))' in profile
    assert "code-review" not in profile
    assert f'(allow file-read* file-write* (subpath "{home}/.config/gh"))' not in profile


def test_seatbelt_gives_back_an_agent_dir_inside_a_secret(tmp_path: Path, home: Path):
    spec = make_spec(home, tmp_path, agent_dirs=[home / ".config/gh"])  # Copilot logs in through it
    profile = seatbelt_profile(spec, home)
    assert f'(allow file-read* file-write* (subpath "{home}/.config/gh"))' in profile


@pytest.mark.skipif(
    sys.platform != "darwin" or not Path(SANDBOX_EXEC).exists(),
    reason="needs macOS sandbox-exec",
)
class TestSeatbeltForReal:
    def run(self, spec: ExecutionSpec, home: Path, script: str) -> subprocess.CompletedProcess:
        profile = spec.sandbox_tmp / "profile.sb"
        profile.write_text(seatbelt_profile(spec, home))
        return subprocess.run(  # noqa: S603 - a fixed sandbox-exec argv around a test script
            [SANDBOX_EXEC, "-f", str(profile), "/bin/sh", "-c", script],
            capture_output=True,
            text=True,
            check=False,
        )

    @pytest.mark.parametrize("rel", [*SECRET_FILES, ".password-store/vault.gpg", ".config/gh/hosts.yml"])
    def test_secret_is_unreadable(self, tmp_path: Path, home: Path, rel: str):
        result = self.run(make_spec(home, tmp_path), home, f"cat {home / rel}")
        assert result.returncode != 0
        assert CANARY not in result.stdout

    def test_odev_data_is_unreadable_but_its_config_venv_and_workspace_are(self, tmp_path: Path, home: Path):
        spec = make_spec(home, tmp_path)
        assert self.run(spec, home, f"cat {home}/.config/odev/code-review/state").returncode != 0
        for readable in ("notes.txt", ".config/odev/odev.cfg", ".config/odev/venv/pyvenv.cfg"):
            assert self.run(spec, home, f"cat {home / readable}").returncode == 0, readable

    def test_secrets_and_venv_are_not_writable(self, tmp_path: Path, home: Path):
        spec = make_spec(home, tmp_path)
        for rel in (".zshrc", ".config/odev/venv/pyvenv.cfg", ".config/odev/plugins/evil.py"):
            assert self.run(spec, home, f"echo x >> {home / rel}").returncode != 0, rel
        assert (home / ".zshrc").read_text() == CANARY
        assert self.run(spec, home, f"echo x >> {home}/notes.txt").returncode == 0  # the workspace still is


# --- bwrap argv ---------------------------------------------------------------


def mask_args(spec: ExecutionSpec, home: Path) -> list[str]:
    cmd: list[str] = []
    BwrapSandbox._mask_user_secrets(cmd, spec, home)
    return cmd


def test_bwrap_masks_nothing_when_home_is_the_playground(tmp_path: Path, home: Path):
    assert mask_args(make_spec(home, tmp_path, bind_home=False), home) == []


def test_bwrap_masks_secrets_brought_back_by_a_home_bind(tmp_path: Path, home: Path):
    args = mask_args(make_spec(home, tmp_path), home)
    pairs = list(pairwise(args))
    triples = list(zip(args, args[1:], args[2:], strict=False))
    for rel in (".ssh", ".password-store", ".config/gh", ".config/odev"):
        assert ("--tmpfs", str(home / rel)) in pairs, rel
    for rel in (".zshrc", ".bashrc", ".profile", ".git-credentials", ".pgpass"):
        assert ("--ro-bind", "/dev/null", str(home / rel)) in triples, rel
    assert f"{home}/.aws" not in args  # absent on the host, nothing to hide


def test_bwrap_masks_odev_config_with_its_copy_then_rebinds_venv(tmp_path: Path, home: Path):
    spec = make_spec(home, tmp_path)
    (spec.playground / ".config/odev").mkdir(parents=True)
    args = mask_args(spec, home)
    mask = args.index(str(spec.playground / ".config/odev"))
    assert args[mask - 1] == "--bind"
    assert args[mask + 1] == f"{home}/.config/odev"
    venv = args.index(f"{home}/.config/odev/venv")
    assert args[venv - 1] == "--ro-bind-try"
    assert mask < venv


def test_bwrap_rebinds_an_agent_dir_inside_a_masked_secret(tmp_path: Path, home: Path):
    args = mask_args(make_spec(home, tmp_path, agent_dirs=[home / ".config/gh"]), home)
    tmpfs = args.index(f"{home}/.config/gh")
    rebind = len(args) - 1 - args[::-1].index(f"{home}/.config/gh")
    assert args[tmpfs - 1] == "--tmpfs"
    assert args[rebind - 2] == "--bind-try"
    assert tmpfs < rebind


def test_bwrap_does_not_mask_a_secret_bound_by_name(tmp_path: Path, home: Path):
    spec = make_spec(home, tmp_path, bind_home=False, agent_dirs=[home / ".config/gh"])
    assert mask_args(spec, home) == []


def _bwrap_works() -> bool:
    if not sys.platform.startswith("linux") or not shutil.which("bwrap"):
        return False
    probe = subprocess.run(
        ["bwrap", "--unshare-user", "--ro-bind", "/", "/", "true"],  # noqa: S607 - found on PATH
        capture_output=True,
        check=False,
    )
    return probe.returncode == 0


@pytest.mark.skipif(not _bwrap_works(), reason="needs a working bwrap (Linux, unprivileged user namespaces)")
class TestBwrapForReal:
    def run(self, spec: ExecutionSpec, home: Path, script: str) -> subprocess.CompletedProcess:
        cmd = ["bwrap", "--ro-bind", "/", "/", "--dev", "/dev", "--proc", "/proc", "--tmpfs", str(home.parent)]
        cmd += ["--bind", str(spec.playground), str(home)]
        for src, dst, ro, _primary in spec.final_binds:
            cmd += ["--ro-bind" if ro else "--bind", str(src), str(dst)]
        BwrapSandbox._mask_user_secrets(cmd, spec, home)
        return subprocess.run(  # noqa: S603 - a bwrap argv built by the test around a test script
            [*cmd, "/bin/sh", "-c", script], capture_output=True, text=True, check=False
        )

    @pytest.mark.parametrize("rel", [*SECRET_FILES, ".password-store/vault.gpg", ".config/gh/hosts.yml"])
    def test_secret_is_unreadable(self, tmp_path: Path, home: Path, rel: str):
        result = self.run(make_spec(home, tmp_path), home, f"cat {home / rel}")
        assert CANARY not in result.stdout

    def test_odev_config_is_the_sanitized_copy(self, tmp_path: Path, home: Path):
        spec = make_spec(home, tmp_path)
        BwrapSandbox._prepare_odev_config(BwrapSandbox.__new__(BwrapSandbox), spec.playground, home)
        result = self.run(spec, home, f"cat {home}/.config/odev/odev.cfg {home}/.config/odev/venv/pyvenv.cfg")
        assert result.returncode == 0, result.stderr
        assert "repositories = /src" in result.stdout
        assert "home = /usr/bin" in result.stdout
        assert CANARY not in result.stdout
        assert self.run(spec, home, f"cat {home}/.config/odev/code-review/state").returncode != 0

    def test_secrets_are_not_writable(self, tmp_path: Path, home: Path):
        self.run(make_spec(home, tmp_path), home, f"echo x >> {home}/.zshrc; echo x >> {home}/.ssh/id_ed25519")
        assert (home / ".zshrc").read_text() == CANARY
        assert (home / ".ssh/id_ed25519").read_text() == CANARY
