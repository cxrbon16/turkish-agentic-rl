"""Task definition loading and host-side grading."""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from verifiable_dataset.terminal.sandbox import DockerSandbox


@dataclass
class Task:
    id: str
    image: str
    prompt: str
    task_dir: Path
    split: str = "single_step"
    workdir: str = "/workspace"
    max_turns: int = 8
    reference_solution: str = ""
    metadata: dict = field(default_factory=dict)

    @classmethod
    def load(cls, task_dir: str | Path) -> "Task":
        task_dir = Path(task_dir)
        spec = yaml.safe_load((task_dir / "task.yaml").read_text(encoding="utf-8"))
        return cls(
            id=spec["id"],
            image=spec["image"],
            prompt=spec["prompt"],
            task_dir=task_dir,
            split=spec.get("split", "single_step"),
            workdir=spec.get("workdir", "/workspace"),
            max_turns=spec.get("max_turns", 8),
            reference_solution=spec.get("reference_solution", ""),
            metadata=spec.get("metadata", {}),
        )

    @property
    def checker(self) -> Path:
        return self.task_dir / "tests" / "check.py"

    def make_sandbox(self) -> DockerSandbox:
        return DockerSandbox(image=self.image, workdir=self.workdir)


@dataclass
class GradeResult:
    reward: float
    solved: bool
    passed: int
    total: int
    checks: list[dict]

    def to_dict(self) -> dict[str, Any]:
        return {
            "reward": self.reward,
            "solved": self.solved,
            "passed": self.passed,
            "total": self.total,
            "checks": self.checks,
        }


def grade(task: Task, sandbox: DockerSandbox) -> GradeResult:
    """Snapshot the container and run the task's checker on the host."""
    with tempfile.TemporaryDirectory(prefix="vds-grade-") as tmp:
        snapshot = sandbox.snapshot(Path(tmp) / "ws")
        proc = subprocess.run(
            [sys.executable, str(task.checker), str(snapshot)],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=120,
        )
    if proc.returncode != 0:
        raise RuntimeError(f"checker hata verdi: {proc.stderr.strip()}")
    report = json.loads(proc.stdout)
    return GradeResult(
        reward=report["reward"],
        solved=report["solved"],
        passed=report["passed"],
        total=report["total"],
        checks=report["checks"],
    )
