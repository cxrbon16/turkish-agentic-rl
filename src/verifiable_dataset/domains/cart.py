"""Shopping cart with a fixed catalog and coupon rules.

The catalog and coupon rules are part of initial_state so tasks are
self-contained; the reference pricing logic below is also used by the
generator to compute ground truth.
"""
from __future__ import annotations

import copy

from verifiable_dataset.base import BaseToolEnv, ToolCallError

TOOLS = [
    {
        "name": "add_item",
        "description": "Add qty units of the given sku to the cart.",
        "parameters": {
            "type": "object",
            "properties": {"sku": {"type": "string"}, "qty": {"type": "integer", "minimum": 1}},
            "required": ["sku", "qty"],
            "additionalProperties": False,
        },
    },
    {
        "name": "remove_item",
        "description": "Remove qty units of the given sku from the cart.",
        "parameters": {
            "type": "object",
            "properties": {"sku": {"type": "string"}, "qty": {"type": "integer", "minimum": 1}},
            "required": ["sku", "qty"],
            "additionalProperties": False,
        },
    },
    {
        "name": "apply_coupon",
        "description": "Apply a coupon code to the cart. Fails if the code is invalid or unmet.",
        "parameters": {
            "type": "object",
            "properties": {"code": {"type": "string"}},
            "required": ["code"],
            "additionalProperties": False,
        },
    },
    {
        "name": "checkout",
        "description": "Finalize the cart and compute the total price after any applied coupon.",
        "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
    },
]


def price_cart(items: dict[str, int], catalog: dict[str, float], coupon: dict | None) -> float:
    """Reference pricing logic, shared by the env and the generator."""
    subtotal = sum(catalog[sku] * qty for sku, qty in items.items())
    if coupon is None:
        return round(subtotal, 2)
    if coupon["type"] == "percent_off":
        if subtotal < coupon.get("min_subtotal", 0):
            return round(subtotal, 2)
        return round(subtotal * (1 - coupon["amount"] / 100), 2)
    if coupon["type"] == "flat_off":
        if subtotal < coupon.get("min_subtotal", 0):
            return round(subtotal, 2)
        return round(max(0.0, subtotal - coupon["amount"]), 2)
    return round(subtotal, 2)


class CartEnv(BaseToolEnv):
    def __init__(self, initial_state: dict):
        self.catalog: dict[str, float] = copy.deepcopy(initial_state["catalog"])
        self.coupons: dict[str, dict] = copy.deepcopy(initial_state.get("coupons", {}))
        self.items: dict[str, int] = copy.deepcopy(initial_state.get("items", {}))
        self.applied_coupon: str | None = initial_state.get("applied_coupon")
        self.checked_out = False
        self.total: float | None = None

    def add_item(self, sku: str, qty: int) -> None:
        if sku not in self.catalog:
            raise ToolCallError(f"unknown sku: {sku}")
        self.items[sku] = self.items.get(sku, 0) + qty

    def remove_item(self, sku: str, qty: int) -> None:
        if self.items.get(sku, 0) < qty:
            raise ToolCallError(f"cannot remove {qty} of {sku}: not enough in cart")
        self.items[sku] -= qty
        if self.items[sku] == 0:
            del self.items[sku]

    def apply_coupon(self, code: str) -> None:
        if code not in self.coupons:
            raise ToolCallError(f"invalid coupon: {code}")
        subtotal = sum(self.catalog[sku] * qty for sku, qty in self.items.items())
        coupon = self.coupons[code]
        if subtotal < coupon.get("min_subtotal", 0):
            raise ToolCallError(f"coupon {code} requires min subtotal {coupon.get('min_subtotal', 0)}")
        self.applied_coupon = code

    def checkout(self) -> float:
        coupon = self.coupons.get(self.applied_coupon) if self.applied_coupon else None
        self.total = price_cart(self.items, self.catalog, coupon)
        self.checked_out = True
        return self.total

    def state_dict(self) -> dict:
        return {
            "items": dict(self.items),
            "applied_coupon": self.applied_coupon,
            "checked_out": self.checked_out,
            "total": self.total,
        }


def make_env(initial_state: dict) -> CartEnv:
    return CartEnv(initial_state)
