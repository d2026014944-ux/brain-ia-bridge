#!/usr/bin/env python3
"""PreToolUse hook: enforces anti-vibe coding guardrails."""

from __future__ import annotations

import json
import re
import sys
from typing import Any


def _load_payload() -> dict[str, Any]:
    raw = sys.stdin.read().strip()
    if not raw:
        return {}
    try:
        data = json.loads(raw)
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        pass
    return {}


def _find_tool_name(payload: dict[str, Any]) -> str:
    for key in ("toolName", "tool", "name"):
        value = payload.get(key)
        if isinstance(value, str) and value:
            return value
    hook_input = payload.get("hookInput")
    if isinstance(hook_input, dict):
        for key in ("toolName", "tool", "name"):
            value = hook_input.get(key)
            if isinstance(value, str) and value:
                return value
    return ""


def _find_tool_args(payload: dict[str, Any]) -> dict[str, Any]:
    for key in ("toolInput", "input", "arguments", "args"):
        value = payload.get(key)
        if isinstance(value, dict):
            return value
    hook_input = payload.get("hookInput")
    if isinstance(hook_input, dict):
        for key in ("toolInput", "input", "arguments", "args"):
            value = hook_input.get(key)
            if isinstance(value, dict):
                return value
    return {}


def _pretool(decision: str, reason: str, system_message: str | None = None) -> None:
    out: dict[str, Any] = {
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": decision,
            "permissionDecisionReason": reason,
        }
    }
    if system_message:
        out["systemMessage"] = system_message
    print(json.dumps(out, ensure_ascii=True))


def _is_destructive_command(command: str) -> bool:
    patterns = [
        r"(^|\s)git\s+reset\s+--hard(\s|$)",
        r"(^|\s)git\s+checkout\s+--(\s|$)",
        r"(^|\s)rm\s+-rf\s+/(\s|$)",
        r"(^|\s)rm\s+-rf\s+\.(\s|$)",
        r"(^|\s)rm\s+-rf\s+\$PWD(\s|$)",
    ]
    return any(re.search(pattern, command) for pattern in patterns)


def _extract_patch_paths(patch_text: str) -> list[str]:
    paths: list[str] = []
    for line in patch_text.splitlines():
        if line.startswith("*** Update File: "):
            paths.append(line.replace("*** Update File: ", "").strip())
        elif line.startswith("*** Add File: "):
            paths.append(line.replace("*** Add File: ", "").strip())
        elif line.startswith("*** Delete File: "):
            paths.append(line.replace("*** Delete File: ", "").strip())
    return paths


def _needs_tdd_confirmation(paths: list[str]) -> bool:
    if not paths:
        return False

    impl_touched = False
    test_touched = False

    for path in paths:
        norm = path.replace("\\", "/").lower()
        if "/tests/" in norm or norm.startswith("tests/"):
            test_touched = True
        if "/src/" in norm or norm.startswith("src/"):
            # Ignore docs/spec edits in src-like names if any appear.
            if norm.endswith(".md"):
                continue
            impl_touched = True

    return impl_touched and not test_touched


def main() -> None:
    payload = _load_payload()
    tool_name = _find_tool_name(payload).lower()
    tool_args = _find_tool_args(payload)

    if tool_name in {"run_in_terminal", "create_and_run_task"}:
        command = str(tool_args.get("command", ""))
        if _is_destructive_command(command):
            _pretool(
                "deny",
                "Comando destrutivo bloqueado pela politica Anti-Vibe Coding.",
                "Use alternativa segura e explicite objetivo + validacao antes de prosseguir.",
            )
            return

    if tool_name == "apply_patch":
        patch_text = str(tool_args.get("input", ""))
        paths = _extract_patch_paths(patch_text)
        if _needs_tdd_confirmation(paths):
            _pretool(
                "ask",
                "Akita Way: implementacao em src sem alteracao em testes detectada.",
                "Confirme que os testes da feature foram escritos primeiro (TDD) ou ajuste o patch para incluir testes.",
            )
            return

    if tool_name == "create_file":
        file_path = str(tool_args.get("filePath", ""))
        if _needs_tdd_confirmation([file_path]):
            _pretool(
                "ask",
                "Akita Way: novo arquivo de implementacao sem testes no mesmo passo.",
                "Confirme TDD (testes primeiro) ou inclua arquivo de teste correspondente.",
            )
            return

    _pretool("allow", "Operacao permitida pela politica Anti-Vibe Coding.")


if __name__ == "__main__":
    main()
