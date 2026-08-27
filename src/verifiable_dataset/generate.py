"""Generate the verifiable tool-use dataset.

Each task is generated with a reference solution (a real tool-call trace)
that was actually executed against the mock environment to derive the
ground-truth assertions -- so the dataset can never drift from what its own
verifier considers correct.

Usage:
    python -m verifiable_dataset.generate --n-per-domain 50 --seed 0 --out data/dataset.jsonl
"""
from __future__ import annotations

import argparse
import json
import random

from verifiable_dataset.domains import calculator, calendar, cart, filesystem
from verifiable_dataset.domains.calendar import CalendarEnv
from verifiable_dataset.domains.cart import price_cart
from verifiable_dataset.domains.filesystem import FilesystemEnv
from verifiable_dataset.base import ToolCallError

WORDS = ["alpha", "beta", "report", "draft", "notes", "budget", "plan", "index", "archive", "memo"]
TITLES = ["Standup", "1:1", "Design Review", "Client Call", "Retro", "Planning"]
NEW_TITLES = ["Team Sync", "Onboarding", "Budget Review", "Roadmap Chat", "Interview"]


def _task(idx: int, domain: str, prompt: str, tools: list[dict], initial_state: dict,
          expected_assertions: list[dict], reference_tool_calls: list[dict], metadata: dict) -> dict:
    return {
        "id": f"{domain}-{idx:04d}",
        "domain": domain,
        "prompt": prompt,
        "tools": tools,
        "initial_state": initial_state,
        "expected_assertions": expected_assertions,
        "reference_tool_calls": reference_tool_calls,
        "metadata": metadata,
    }


def gen_calculator_task(rng: random.Random, idx: int) -> dict:
    op_symbols = {"add": "+", "subtract": "-", "multiply": "*", "divide": "/"}
    n_numbers = rng.randint(2, 4)
    numbers = [rng.randint(1, 20) for _ in range(n_numbers)]

    running = float(numbers[0])
    expr = str(numbers[0])
    steps: list[tuple[str, float]] = []  # (op, operand)

    for num in numbers[1:]:
        op = rng.choice(["add", "subtract", "multiply", "divide"])
        if op == "divide":
            divisors = [d for d in range(1, 21) if running != 0 and running % d == 0]
            if divisors:
                num = rng.choice(divisors)
            else:
                op = "add"
        steps.append((op, num))
        if op == "add":
            running += num
        elif op == "subtract":
            running -= num
        elif op == "multiply":
            running *= num
        elif op == "divide":
            running /= num
        expr = f"({expr} {op_symbols[op]} {num})"

    apply_sqrt = rng.random() < 0.25 and running >= 0
    if apply_sqrt:
        running = running ** 0.5
        expr = f"sqrt({expr})"

    value = round(running, 6)

    reference_tool_calls = [{"name": "push", "arguments": {"value": numbers[0]}}]
    for op, num in steps:
        reference_tool_calls.append({"name": "push", "arguments": {"value": num}})
        reference_tool_calls.append({"name": op, "arguments": {}})
    if apply_sqrt:
        reference_tool_calls.append({"name": "sqrt", "arguments": {}})

    prompt = (
        f"Using the calculator's stack-based tools (push/add/subtract/multiply/divide/sqrt), "
        f"compute the value of {expr} and leave the result as the only value on the stack."
    )
    expected_assertions = [
        {"path": "stack", "op": "len_eq", "value": 1},
        {"path": "stack.0", "op": "approx", "value": value, "tol": 1e-6},
    ]
    return _task(
        idx, "calculator", prompt, calculator.TOOLS, {"stack": []},
        expected_assertions, reference_tool_calls, {"expression": expr, "value": value},
    )


def gen_filesystem_task(rng: random.Random, idx: int) -> dict:
    env = FilesystemEnv({"root": {"type": "dir", "children": {}}})
    dirs: list[str] = []
    files: list[str] = []
    reference_tool_calls: list[dict] = []
    nl_steps: list[str] = []

    n_ops = rng.randint(3, 6)
    for _ in range(n_ops):
        choices = ["mkdir", "write"] + (["mv", "rm"] if (dirs or files) else [])
        action = rng.choice(choices)
        try:
            if action == "mkdir":
                name = f"{rng.choice(WORDS)}{rng.randint(1, 99)}"
                parent = rng.choice(dirs) if dirs and rng.random() < 0.5 else ""
                path = f"{parent}/{name}".strip("/")
                env.mkdir(path)
                dirs.append(path)
                reference_tool_calls.append({"name": "mkdir", "arguments": {"path": path}})
                nl_steps.append(f"create a directory at '{path}'")
            elif action == "write":
                name = f"{rng.choice(WORDS)}{rng.randint(1, 99)}.txt"
                parent = rng.choice(dirs) if dirs and rng.random() < 0.6 else ""
                path = f"{parent}/{name}".strip("/")
                content = f"{rng.choice(WORDS)} {rng.choice(WORDS)} {rng.randint(1, 100)}"
                env.write(path, content)
                files.append(path)
                reference_tool_calls.append({"name": "write", "arguments": {"path": path, "content": content}})
                nl_steps.append(f"write a file at '{path}' with the exact content \"{content}\"")
            elif action == "mv":
                pool = dirs + files
                src = rng.choice(pool)
                newname = f"{rng.choice(WORDS)}{rng.randint(1, 99)}"
                parent = "/".join(src.split("/")[:-1])
                dst = f"{parent}/{newname}".strip("/")
                env.mv(src, dst)
                if src in dirs:
                    dirs[dirs.index(src)] = dst
                if src in files:
                    files[files.index(src)] = dst
                reference_tool_calls.append({"name": "mv", "arguments": {"src": src, "dst": dst}})
                nl_steps.append(f"move '{src}' to '{dst}'")
            elif action == "rm":
                pool = dirs + files
                target = rng.choice(pool)
                env.rm(target)
                if target in dirs:
                    dirs.remove(target)
                if target in files:
                    files.remove(target)
                reference_tool_calls.append({"name": "rm", "arguments": {"path": target}})
                nl_steps.append(f"remove '{target}'")
        except ToolCallError:
            continue

    if not reference_tool_calls:
        env.mkdir("docs")
        reference_tool_calls.append({"name": "mkdir", "arguments": {"path": "docs"}})
        nl_steps.append("create a directory at 'docs'")

    final_state = env.state_dict()
    prompt = "Perform the following filesystem operations, in order: " + "; ".join(
        f"{i + 1}) {s}" for i, s in enumerate(nl_steps)
    ) + "."
    expected_assertions = [{"path": "root", "op": "eq", "value": final_state["root"]}]
    return _task(
        idx, "filesystem", prompt, filesystem.TOOLS, {"root": {"type": "dir", "children": {}}},
        expected_assertions, reference_tool_calls, {"n_steps": len(reference_tool_calls)},
    )


def _fmt_time(minutes: int) -> str:
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


def gen_calendar_task(rng: random.Random, idx: int) -> dict:
    day_start, day_end = 480, 1080  # 08:00-18:00

    events: list[dict] = []
    n_initial = rng.randint(0, 3)
    next_id = 1
    attempts = 0
    while len(events) < n_initial and attempts < 50:
        attempts += 1
        start = rng.randrange(day_start, day_end - 30, 15)
        dur = rng.choice([30, 45, 60])
        end = start + dur
        if any(start < e["end"] and e["start"] < end for e in events):
            continue
        events.append({"id": f"e{next_id}", "title": rng.choice(TITLES), "start": start, "end": end})
        next_id += 1

    initial_state = {"events": events, "next_id": next_id}
    duration = rng.choice([30, 45, 60])
    title = rng.choice(NEW_TITLES)

    env = CalendarEnv(initial_state)
    slot = env.find_free_slot(duration_minutes=duration, day_start_min=day_start, day_end_min=day_end)
    if slot is None:
        # exceedingly packed day; fall back to an empty calendar for this task
        events = []
        initial_state = {"events": events, "next_id": 1}
        env = CalendarEnv(initial_state)
        slot = env.find_free_slot(duration_minutes=duration, day_start_min=day_start, day_end_min=day_end)

    env.create_event(title=title, start=slot, end=slot + duration)
    final_state = env.state_dict()

    if events:
        busy_desc = "; ".join(
            f"'{e['title']}' from {_fmt_time(e['start'])} to {_fmt_time(e['end'])}"
            for e in sorted(events, key=lambda e: e["start"])
        )
    else:
        busy_desc = "no events yet"
    prompt = (
        f"My calendar today ({_fmt_time(day_start)}-{_fmt_time(day_end)} available window) currently has: "
        f"{busy_desc}. Schedule a new {duration}-minute meeting titled '{title}' at the earliest available "
        f"slot in that window, avoiding conflicts with existing events."
    )

    reference_tool_calls = [
        {"name": "find_free_slot", "arguments": {"duration_minutes": duration, "day_start_min": day_start, "day_end_min": day_end}},
        {"name": "create_event", "arguments": {"title": title, "start": slot, "end": slot + duration}},
    ]
    expected_assertions = [{"path": "event_signatures", "op": "eq", "value": final_state["event_signatures"]}]
    return _task(
        idx, "calendar", prompt, calendar.TOOLS, initial_state,
        expected_assertions, reference_tool_calls, {"expected_slot": slot, "duration": duration, "title": title},
    )


def gen_cart_task(rng: random.Random, idx: int) -> dict:
    catalog_pool = {
        "widget": 9.99, "gadget": 14.5, "gizmo": 3.25, "thingamajig": 22.0,
        "doohickey": 7.75, "contraption": 18.3, "sprocket": 5.0, "widgetpro": 29.99,
    }
    skus = rng.sample(list(catalog_pool), rng.randint(3, 5))
    catalog = {s: catalog_pool[s] for s in skus}

    n_items = rng.randint(2, min(3, len(skus)))
    chosen = rng.sample(skus, n_items)
    items = {s: rng.randint(1, 4) for s in chosen}
    subtotal = sum(catalog[s] * q for s, q in items.items())

    coupons: dict[str, dict] = {}
    coupon_code = None
    if rng.random() < 0.7:
        coupon_type = rng.choice(["percent_off", "flat_off"])
        amount = rng.choice([5, 10, 15, 20]) if coupon_type == "percent_off" else rng.choice([5, 10])
        coupon_code = rng.choice(["SAVE", "DEAL", "PROMO"]) + str(rng.randint(10, 99))
        coupons[coupon_code] = {"type": coupon_type, "amount": amount, "min_subtotal": round(subtotal * 0.5, 2)}

    initial_state = {"catalog": catalog, "coupons": coupons, "items": {}, "applied_coupon": None}
    coupon = coupons.get(coupon_code) if coupon_code else None
    expected_total = price_cart(items, catalog, coupon)

    item_lines = ", ".join(f"{q} x {s}" for s, q in items.items())
    coupon_clause = f" Apply the coupon code '{coupon_code}'." if coupon_code else ""
    prompt = f"Add the following items to the cart: {item_lines}.{coupon_clause} Then check out and finalize the total."

    reference_tool_calls = [{"name": "add_item", "arguments": {"sku": s, "qty": q}} for s, q in items.items()]
    if coupon_code:
        reference_tool_calls.append({"name": "apply_coupon", "arguments": {"code": coupon_code}})
    reference_tool_calls.append({"name": "checkout", "arguments": {}})

    expected_assertions = [
        {"path": "checked_out", "op": "eq", "value": True},
        {"path": "total", "op": "approx", "value": expected_total, "tol": 1e-6},
    ]
    if coupon_code:
        expected_assertions.append({"path": "applied_coupon", "op": "eq", "value": coupon_code})

    metadata = {"catalog": catalog, "coupons": coupons, "expected_total": expected_total}
    return _task(idx, "cart", prompt, cart.TOOLS, initial_state, expected_assertions, reference_tool_calls, metadata)


GENERATORS = {
    "calculator": gen_calculator_task,
    "filesystem": gen_filesystem_task,
    "calendar": gen_calendar_task,
    "cart": gen_cart_task,
}


def generate_dataset(n_per_domain: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    tasks = []
    for domain, gen_fn in GENERATORS.items():
        for i in range(n_per_domain):
            tasks.append(gen_fn(rng, i))
    return tasks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-per-domain", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, default="data/dataset.jsonl")
    args = parser.parse_args()

    tasks = generate_dataset(args.n_per_domain, args.seed)
    with open(args.out, "w", encoding="utf-8") as f:
        for task in tasks:
            f.write(json.dumps(task) + "\n")
    print(f"wrote {len(tasks)} tasks to {args.out}")


if __name__ == "__main__":
    main()
