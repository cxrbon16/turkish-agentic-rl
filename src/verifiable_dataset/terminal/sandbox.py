"""Docker-backed terminal sandbox.

One container per episode, kept alive across turns so filesystem state
persists. Commands run via ``docker exec``, which means the working
directory and exported environment variables do NOT carry over between
turns -- each exec is a fresh shell. That is fine for filesystem tasks;
tasks that need a persistent venv or background processes will need a
PTY-attached session instead (see notes in README).
"""
from __future__ import annotations

import shutil
import subprocess
import uuid
from dataclasses import dataclass
from pathlib import Path

DOCKER = shutil.which("docker") or "docker"


class SandboxError(RuntimeError):
    """Raised when the container itself misbehaves (not when a command fails)."""


@dataclass
class ExecResult:
    stdout: str
    stderr: str
    exit_code: int
    timed_out: bool = False

    def to_observation(self, max_chars: int = 4000) -> str:
        """Render the result the way the agent sees it."""
        parts = [f"exit_code: {self.exit_code}"]
        if self.timed_out:
            parts.append("(komut zaman asimina ugradi)")
        if self.stdout.strip():
            parts.append(f"stdout:\n{self.stdout.rstrip()}")
        if self.stderr.strip():
            parts.append(f"stderr:\n{self.stderr.rstrip()}")
        if not self.stdout.strip() and not self.stderr.strip():
            parts.append("(cikti yok)")
        text = "\n".join(parts)
        if len(text) > max_chars:
            text = text[:max_chars] + f"\n... (kesildi, toplam {len(text)} karakter)"
        return text


def _run(args: list[str], timeout: int | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        args,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
    )


class DockerSandbox:
    """Start a container, exec commands in it, snapshot it, tear it down."""

    def __init__(
        self,
        image: str,
        workdir: str = "/workspace",
        cpus: str = "1",
        memory: str = "512m",
        pids_limit: int = 256,
        network: str = "none",
    ):
        self.image = image
        self.workdir = workdir
        self.cpus = cpus
        self.memory = memory
        self.pids_limit = pids_limit
        self.network = network
        self.container_id: str | None = None

    # -- lifecycle ----------------------------------------------------

    def start(self) -> str:
        name = f"vds-{uuid.uuid4().hex[:12]}"
        args = [
            DOCKER, "run", "-d", "--rm",
            "--name", name,
            "--network", self.network,
            "--cpus", self.cpus,
            "--memory", self.memory,
            "--pids-limit", str(self.pids_limit),
            "--workdir", self.workdir,
            self.image,
            "sleep", "infinity",
        ]
        proc = _run(args, timeout=120)
        if proc.returncode != 0:
            raise SandboxError(f"container baslatilamadi: {proc.stderr.strip()}")
        self.container_id = proc.stdout.strip()
        return self.container_id

    def stop(self) -> None:
        if not self.container_id:
            return
        _run([DOCKER, "kill", self.container_id], timeout=60)
        self.container_id = None

    def __enter__(self) -> "DockerSandbox":
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()

    # -- operations ---------------------------------------------------

    def exec(self, command: str, timeout: int = 30) -> ExecResult:
        """Run a shell command inside the container."""
        if not self.container_id:
            raise SandboxError("container calismiyor")
        args = [
            DOCKER, "exec", "--workdir", self.workdir,
            self.container_id, "bash", "-lc", command,
        ]
        try:
            proc = _run(args, timeout=timeout)
        except subprocess.TimeoutExpired:
            return ExecResult(stdout="", stderr="", exit_code=124, timed_out=True)
        return ExecResult(stdout=proc.stdout, stderr=proc.stderr, exit_code=proc.returncode)

    def snapshot(self, dest: Path) -> Path:
        """Copy the workdir out to the host for grading.

        Grading never runs inside the container, so a tampered interpreter
        or a planted checker cannot influence the reward.
        """
        if not self.container_id:
            raise SandboxError("container calismiyor")
        dest = Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            shutil.rmtree(dest)
        proc = _run(
            [DOCKER, "cp", f"{self.container_id}:{self.workdir}", str(dest)],
            timeout=120,
        )
        if proc.returncode != 0:
            raise SandboxError(f"snapshot alinamadi: {proc.stderr.strip()}")
        return dest


def docker_available() -> tuple[bool, str]:
    """Check that the Docker daemon is reachable."""
    try:
        proc = _run([DOCKER, "info", "--format", "{{.ServerVersion}}"], timeout=30)
    except (OSError, subprocess.TimeoutExpired) as e:
        return False, str(e)
    if proc.returncode != 0:
        return False, proc.stderr.strip() or "docker daemon yanit vermiyor"
    return True, proc.stdout.strip()
