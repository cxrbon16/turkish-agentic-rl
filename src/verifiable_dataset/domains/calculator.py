"""RPN stack calculator.

Deliberately stack-based rather than pure-function (add(a,b)) so that the
final answer is a genuine consequence of the executed call sequence, not of
whatever numbers happen to appear in arguments. You cannot fake a correct
result by hardcoding a value into an unrelated call's argument.
"""
from __future__ import annotations

import math

from verifiable_dataset.base import BaseToolEnv, ToolCallError

TOOLS = [
    {
        "name": "push",
        "description": "Push a numeric literal onto the stack.",
        "parameters": {
            "type": "object",
            "properties": {"value": {"type": "number"}},
            "required": ["value"],
            "additionalProperties": False,
        },
    },
    {
        "name": "add",
        "description": "Pop the top two values and push their sum.",
        "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
    },
    {
        "name": "subtract",
        "description": "Pop b then a (a was pushed first); push a - b.",
        "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
    },
    {
        "name": "multiply",
        "description": "Pop the top two values and push their product.",
        "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
    },
    {
        "name": "divide",
        "description": "Pop b then a (a was pushed first); push a / b.",
        "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
    },
    {
        "name": "sqrt",
        "description": "Pop the top value and push its square root.",
        "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
    },
]


class CalculatorEnv(BaseToolEnv):
    def __init__(self, initial_state: dict):
        self.stack: list[float] = list(initial_state.get("stack", []))

    def _pop(self) -> float:
        if not self.stack:
            raise ToolCallError("pop from empty stack")
        return self.stack.pop()

    def push(self, value: float) -> None:
        self.stack.append(value)

    def add(self) -> None:
        b, a = self._pop(), self._pop()
        self.stack.append(a + b)

    def subtract(self) -> None:
        b, a = self._pop(), self._pop()
        self.stack.append(a - b)

    def multiply(self) -> None:
        b, a = self._pop(), self._pop()
        self.stack.append(a * b)

    def divide(self) -> None:
        b, a = self._pop(), self._pop()
        if b == 0:
            raise ToolCallError("division by zero")
        self.stack.append(a / b)

    def sqrt(self) -> None:
        a = self._pop()
        if a < 0:
            raise ToolCallError("sqrt of negative number")
        self.stack.append(math.sqrt(a))

    def state_dict(self) -> dict:
        return {"stack": list(self.stack)}


def make_env(initial_state: dict) -> CalculatorEnv:
    return CalculatorEnv(initial_state)
