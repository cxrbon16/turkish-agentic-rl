"""Run a terminal task -- either with the reference solution or with a model.

    # harness'i modelsiz dogrula (referans cozumu calistirir)
    python -m verifiable_dataset.terminal.runner --task envs/smoke-001 --reference

    # bir modele cozdur
    python -m verifiable_dataset.terminal.runner --task envs/smoke-001 \
        --model qwen2.5-7b-instruct --rollouts 4
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from verifiable_dataset.terminal.sandbox import DockerSandbox, docker_available
from verifiable_dataset.terminal.task import GradeResult, Task, grade, run_script

SYSTEM_PROMPT = (
    "Sen bir Linux terminalinde calisan bir yardimcisin. "
    "Verilen gorevi tamamlamak icin run_command aracini kullanarak shell komutlari calistir. "
    "Her adimda komutun ciktisini gorursun. Gorev bitince arac cagirmayi birak ve kisaca ozetle."
)

# Native tool calling'i olmayan modeller icin (orn. Gemma) metin protokolu.
# Gemma system rolunu de kabul etmedigi icin bu talimat ilk user mesajina
# gomulur ve gozlemler user mesaji olarak geri verilir.
TEXT_PROTOCOL_PROMPT = """Bir Linux terminaline erisimin var. Calisma dizinin /workspace.

Komut calistirmak icin komutu aynen su etiketlerin arasina yaz:

<komut>
ls -la
</komut>

Kurallar:
- Her mesajda en fazla bir <komut> blogu kullan.
- Komutun ciktisini bir sonraki mesajda goreceksin, sonra devam edebilirsin.
- Gorev tamamlandiginda <komut> etiketi KULLANMA, sadece ne yaptigini kisaca ozetle.

Gorev:
"""

# Sunucunun parse etmeyi beceremedigi ham arac cagrisi izleri.
_RAW_TOOLCALL_RE = re.compile(r"<tool_call>|<function=|\"name\"\s*:\s*\"run_command\"")

_TAG_RE = re.compile(r"<komut>\s*(.*?)\s*</komut>", re.DOTALL | re.IGNORECASE)
_FENCE_RE = re.compile(r"```(?:bash|sh|shell)?\s*\n(.*?)```", re.DOTALL)


def extract_commands(content: str) -> list[str]:
    """Pull shell commands out of a plain-text assistant reply.

    Prefers explicit <komut> tags; falls back to fenced code blocks because
    small models reach for markdown out of habit.
    """
    if not content:
        return []
    matches = [m.strip() for m in _TAG_RE.findall(content)]
    if not matches:
        matches = [m.strip() for m in _FENCE_RE.findall(content)]
    return [m for m in matches if m]

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "run_command",
            "description": (
                "Sandbox terminalinde tek bir shell komutu calistirir ve "
                "stdout, stderr ile exit code dondurur."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "Calistirilacak shell komutu, ornegin: ls -la",
                    }
                },
                "required": ["command"],
                "additionalProperties": False,
            },
        },
    }
]


@dataclass
class Episode:
    task_id: str
    solved: bool
    reward: float      # ikili
    turns: int
    partial: float = 0.0  # teshis icin
    commands: list[str] = field(default_factory=list)
    messages: list[dict] = field(default_factory=list)
    grade: dict = field(default_factory=dict)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "solved": self.solved,
            "reward": self.reward,
            "partial": self.partial,
            "turns": self.turns,
            "commands": self.commands,
            "messages": self.messages,
            "grade": self.grade,
            "error": self.error,
        }


def _assert_not_pre_solved(task: Task, sandbox: DockerSandbox) -> GradeResult:
    """The checker must FAIL on a fresh container.

    Otherwise the task is trivially satisfied and rewards nothing real.
    """
    initial = grade(task, sandbox)
    if initial.solved:
        raise RuntimeError(
            f"{task.id}: checker baslangic durumunda zaten geciyor -- gorev anlamsiz."
        )
    return initial


def run_reference(task: Task, verbose: bool = True) -> Episode:
    """Execute the task's reference solution and grade it."""
    if not task.reference_solution.strip():
        raise ValueError(f"{task.id}: reference_solution tanimli degil")

    with task.make_sandbox() as sandbox:
        task.prepare(sandbox)
        _assert_not_pre_solved(task, sandbox)
        result = run_script(sandbox, task.reference_solution, timeout=60)
        if verbose:
            print(f"[referans] exit={result.exit_code}")
            if result.stderr.strip():
                print(f"[referans] stderr: {result.stderr.strip()}")
        g = grade(task, sandbox)

    return Episode(
        task_id=task.id,
        solved=g.solved,
        reward=g.reward,
        turns=1,
        partial=g.partial,
        commands=[task.reference_solution.strip()],
        grade=g.to_dict(),
    )


def run_model_text(task: Task, client, model: str, verbose: bool = True) -> Episode:
    """Drive the terminal with a plain-text command protocol.

    For models without native tool calling (Gemma, most base models). No
    system role is used -- Gemma's chat template rejects it -- and command
    output comes back as a user message.
    """
    messages: list[dict] = [
        {"role": "user", "content": TEXT_PROTOCOL_PROMPT + task.prompt}
    ]
    commands: list[str] = []
    turns = 0
    error: str | None = None

    with task.make_sandbox() as sandbox:
        task.prepare(sandbox)
        _assert_not_pre_solved(task, sandbox)

        for turns in range(1, task.max_turns + 1):
            try:
                response = client.chat.completions.create(
                    model=model, messages=messages, temperature=0.7, max_tokens=1024,
                )
            except Exception as e:  # noqa: BLE001 - surface API failures as episode errors
                error = f"model cagrisi basarisiz: {e}"
                break

            content = response.choices[0].message.content or ""
            messages.append({"role": "assistant", "content": content})

            found = extract_commands(content)
            if not found:
                if verbose:
                    print(f"[model] {content.strip()[:300]}")
                break

            observations = []
            for command in found:
                commands.append(command)
                if verbose:
                    print(f"[turn {turns}] $ {command}")
                result = sandbox.exec(command)
                if verbose:
                    print(f"           exit={result.exit_code}")
                observations.append(f"$ {command}\n{result.to_observation()}")
            messages.append({
                "role": "user",
                "content": "Komut ciktisi:\n\n" + "\n\n".join(observations),
            })

        g = grade(task, sandbox)

    return Episode(
        task_id=task.id, solved=g.solved, reward=g.reward, turns=turns,
        partial=g.partial, commands=commands, messages=messages,
        grade=g.to_dict(), error=error,
    )


def run_model(task: Task, client, model: str, protocol: str = "auto",
              verbose: bool = True) -> Episode:
    """Run an episode, picking the tool-calling protocol.

    ``auto`` tries native tool calling and falls back to the text protocol
    when the server rejects it (no tool-call parser, or the chat template
    has no tool support).
    """
    if protocol == "text":
        return run_model_text(task, client, model, verbose)
    if protocol == "native":
        return run_model_native(task, client, model, verbose)

    episode = run_model_native(task, client, model, verbose)
    if episode.error and not episode.commands:
        if verbose:
            print("[auto] native tool calling calismadi, metin protokoluna geciliyor")
        return run_model_text(task, client, model, verbose)
    return episode


def run_model_native(task: Task, client, model: str, verbose: bool = True) -> Episode:
    """Let a model drive the terminal using OpenAI-style tool calls."""
    messages: list[dict] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": task.prompt},
    ]
    commands: list[str] = []
    turns = 0
    error: str | None = None

    with task.make_sandbox() as sandbox:
        task.prepare(sandbox)
        _assert_not_pre_solved(task, sandbox)

        for turns in range(1, task.max_turns + 1):
            try:
                response = client.chat.completions.create(
                    model=model, messages=messages, tools=TOOLS, tool_choice="auto",
                )
            except Exception as e:  # noqa: BLE001 - surface API failures as episode errors
                error = f"model cagrisi basarisiz: {e}"
                break

            msg = response.choices[0].message
            tool_calls = msg.tool_calls or []
            messages.append({
                "role": "assistant",
                "content": msg.content,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                    }
                    for tc in tool_calls
                ] or None,
            })

            if not tool_calls:
                # Model arac cagirdi ama sunucu parse edemediyse, cagri ham
                # metin olarak content'te kalir ve episode sessizce biter.
                # Bu neredeyse her zaman yanlis --tool-call-parser demektir.
                if msg.content and _RAW_TOOLCALL_RE.search(msg.content):
                    error = ("sunucu tool call'u parse edemedi -- vLLM'de "
                             "--tool-call-parser yanlis ya da eksik olabilir")
                    print(f"[uyari] {error}")
                elif verbose and msg.content:
                    print(f"[model] {msg.content.strip()[:300]}")
                break

            for tc in tool_calls:
                try:
                    args = json.loads(tc.function.arguments or "{}")
                    command = args["command"]
                except (json.JSONDecodeError, KeyError) as e:
                    observation = f"gecersiz arac argumani: {e}"
                else:
                    commands.append(command)
                    if verbose:
                        print(f"[turn {turns}] $ {command}")
                    observation = sandbox.exec(command).to_observation()
                    if verbose:
                        print(f"           {observation.splitlines()[0]}")
                messages.append({
                    "role": "tool", "tool_call_id": tc.id, "content": observation,
                })

        g = grade(task, sandbox)

    return Episode(
        task_id=task.id, solved=g.solved, reward=g.reward, turns=turns,
        partial=g.partial, commands=commands, messages=messages,
        grade=g.to_dict(), error=error,
    )


def main() -> int:
    # Turkce cikti Windows konsolunun cp1254 kodlamasinda UnicodeEncodeError
    # firlatir; episode'u bu yuzden kaybetmeyelim.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, help="task dizini, orn. envs/smoke-001")
    parser.add_argument("--reference", action="store_true", help="model yerine referans cozumu calistir")
    parser.add_argument("--model", default=os.environ.get("MODEL", ""))
    parser.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL", ""))
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", "dummy"))
    parser.add_argument("--protocol", choices=["auto", "native", "text"], default="auto",
                        help="native: OpenAI tool calling; text: <komut> etiketleri (Gemma vb.)")
    parser.add_argument("--rollouts", type=int, default=1, help="pass@N icin deneme sayisi")
    parser.add_argument("--out", default="", help="episode kayitlarini bu JSONL dosyasina yaz")
    args = parser.parse_args()

    ok, info = docker_available()
    if not ok:
        print(f"Docker daemon'a ulasilamiyor: {info}")
        print("Docker Desktop'i baslatip tekrar dene.")
        return 1

    task = Task.load(args.task)
    print(f"task={task.id} split={task.split} image={task.image} docker={info}")

    client = None
    if not args.reference:
        if not args.model:
            print("--model verilmedi (ya da MODEL env degiskeni bos)")
            return 2
        from openai import OpenAI
        base_url = args.base_url.rstrip("/") if args.base_url else ""
        if base_url and not base_url.endswith("/v1"):
            # vLLM'in OpenAI uyumlu API'si /v1 altinda; tunel URL'lerine
            # eklemeyi unutmak 404 uretiyor.
            base_url += "/v1"
            print(f"base-url'e /v1 eklendi: {base_url}")
        client = OpenAI(base_url=base_url or None, api_key=args.api_key)

    episodes: list[Episode] = []
    for i in range(args.rollouts):
        if args.rollouts > 1:
            print(f"--- rollout {i + 1}/{args.rollouts} ---")
        try:
            ep = (run_reference(task) if args.reference
                  else run_model(task, client, args.model, protocol=args.protocol))
        except Exception as e:  # noqa: BLE001 - one bad rollout must not kill a pass@N sweep
            print(f"HATA  rollout {i + 1} coktu: {type(e).__name__}: {e}")
            episodes.append(Episode(
                task_id=task.id, solved=False, reward=0.0, turns=0,
                grade={"passed": 0, "total": 0, "checks": []},
                error=f"{type(e).__name__}: {e}",
            ))
            continue
        episodes.append(ep)
        status = "SOLVED" if ep.solved else "FAILED"
        print(f"{status}  reward={ep.reward:.0f}  "
              f"(check {ep.grade['passed']}/{ep.grade['total']}, kismi={ep.partial:.2f})  "
              f"turns={ep.turns}")
        for check in ep.grade["checks"]:
            if not check["ok"]:
                print(f"   x {check['name']}: {check['detail']}")
        if ep.error:
            print(f"   ! {ep.error}")

    if args.rollouts > 1:
        n_solved = sum(1 for e in episodes if e.solved)
        mean_reward = sum(e.reward for e in episodes) / len(episodes)
        print(f"\npass@{args.rollouts}: {n_solved}/{args.rollouts}   ortalama reward: {mean_reward:.3f}")

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            for ep in episodes:
                f.write(json.dumps(ep.to_dict(), ensure_ascii=False) + "\n")
        print(f"episodes -> {out_path}")

    return 0 if any(e.solved for e in episodes) else 1


if __name__ == "__main__":
    raise SystemExit(main())
