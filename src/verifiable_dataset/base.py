"""Core, domain-agnostic verification machinery.

A Task is a plain JSON-serializable dict (see generate.py for the schema).
Verification works by replaying a candidate list of tool calls against a
fresh instance of the task's mock environment, then checking assertions
against the resulting state. Reward therefore depends on what the tools
actually did, not on what the caller claims -- arguments can't be
hardcoded to fake a result.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import jsonschema


class ToolCallError(Exception):
    """Raised when a tool call is invalid or fails during execution."""


@dataclass
class ToolCall:
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict) -> "ToolCall":
        return cls(name=d["name"], arguments=d.get("arguments", {}))


@dataclass
class VerificationResult:
    passed: bool
    reward: float
    total_assertions: int
    passed_assertions: int
    errors: list[str]
    failed_assertions: list[dict]

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "reward": self.reward,
            "total_assertions": self.total_assertions,
            "passed_assertions": self.passed_assertions,
            "errors": self.errors,
            "failed_assertions": self.failed_assertions,
        }


class BaseToolEnv:
    """Base class for stateful mock tool backends.

    Subclasses implement one method per tool (matching the tool schema's
    ``name``) and ``state_dict()`` to expose current state for assertions.
    """

    def state_dict(self) -> dict:
        raise NotImplementedError

    def call(self, name: str, arguments: dict) -> Any:
        method = getattr(self, name, None)
        if method is None or not callable(method):
            raise ToolCallError(f"unknown tool method: {name}")
        try:
            return method(**arguments)
        except TypeError as e:
            raise ToolCallError(f"bad arguments for {name}: {e}") from e


def resolve_path(state: Any, path: str) -> Any:
    """Resolve a dotted path (with numeric list indices) into a state tree."""
    cur = state
    for part in path.split("."):
        if isinstance(cur, list):
            cur = cur[int(part)]
        elif isinstance(cur, dict):
            if part not in cur:
                raise KeyError(f"path segment {part!r} not found (path={path!r})")
            cur = cur[part]
        else:
            cur = getattr(cur, part)
    return cur


def check_assertion(state: dict, assertion: dict) -> tuple[bool, str | None]:
    """Evaluate one assertion against final state. Returns (ok, error_if_any)."""
    op = assertion.get("op", "eq")
    try:
        val = resolve_path(state, assertion["path"])
    except (KeyError, IndexError, ValueError, TypeError) as e:
        return False, f"path error: {e}"

    expected = assertion.get("value")
    try:
        if op == "eq":
            return val == expected, None
        if op == "approx":
            return abs(val - expected) <= assertion.get("tol", 1e-6), None
        if op == "set_eq":
            return set(val) == set(expected), None
        if op == "contains":
            return expected in val, None
        if op == "len_eq":
            return len(val) == expected, None
        return False, f"unknown op: {op}"
    except Exception as e:  # noqa: BLE001 - comparison failures are assertion failures
        return False, f"comparison error: {e}"


def verify_trace(task: dict, tool_calls: list[dict], domains: dict) -> VerificationResult:
    """Replay tool_calls against task's mock env and score against assertions.

    Args:
        task: a task dict as produced by generate.py / stored in dataset.jsonl
        tool_calls: list of {"name": str, "arguments": dict}
        domains: mapping of domain name -> domain module (see domains/__init__.py)
    """
    errors: list[str] = []
    domain = domains[task["domain"]]
    env = domain.make_env(task["initial_state"])

    tool_schema_by_name = {t["name"]: t for t in task["tools"]}

    for raw_call in tool_calls:
        call = ToolCall.from_dict(raw_call)
        schema = tool_schema_by_name.get(call.name)
        if schema is None:
            errors.append(f"call to undeclared tool: {call.name}")
            continue
        try:
            jsonschema.validate(call.arguments, schema["parameters"])
        except jsonschema.ValidationError as e:
            errors.append(f"invalid arguments for {call.name}: {e.message}")
            continue
        try:
            env.call(call.name, call.arguments)
        except ToolCallError as e:
            errors.append(str(e))

    final_state = env.state_dict()
    assertions = task["expected_assertions"]
    failed_assertions = []
    passed_count = 0
    for assertion in assertions:
        ok, err = check_assertion(final_state, assertion)
        if ok:
            passed_count += 1
        else:
            failed_assertions.append({**assertion, "error": err})

    total = len(assertions)
    reward = 0.0 if errors else (passed_count / total if total else 1.0)
    passed = reward == 1.0 and not errors

    return VerificationResult(
        passed=passed,
        reward=reward,
        total_assertions=total,
        passed_assertions=passed_count,
        errors=errors,
        failed_assertions=failed_assertions,
    )
