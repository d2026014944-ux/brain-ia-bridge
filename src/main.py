from __future__ import annotations

import argparse
import time
from pathlib import Path

from core.noma_core import DEFAULT_MEMORY_FILE, DEFAULT_STATE_FILE, NomaCore


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MNHI 4.0 unified runtime")
    parser.add_argument("--tick-ms", type=float, default=100.0, help="Main loop interval in milliseconds.")
    parser.add_argument("--state-file", type=Path, default=DEFAULT_STATE_FILE, help="Realtime state JSON for Mind Panel.")
    parser.add_argument("--memory-file", type=Path, default=DEFAULT_MEMORY_FILE, help="Persistent synaptic memory binary.")
    parser.add_argument("--max-steps", type=int, default=None, help="Optional debug cap. Omit for continuous runtime.")
    return parser


def main() -> None:
    args = _build_parser().parse_args()

    core = NomaCore(
        state_file=args.state_file,
        memory_file=args.memory_file,
        tick_ms=args.tick_ms,
    )
    core.boot_log()

    steps_executed = 0
    try:
        while args.max_steps is None or steps_executed < args.max_steps:
            cycle_start = time.perf_counter()
            core.step()
            steps_executed += 1

            elapsed_ms = (time.perf_counter() - cycle_start) * 1000.0
            remaining_ms = args.tick_ms - elapsed_ms
            if remaining_ms > 0.0:
                time.sleep(remaining_ms / 1000.0)
    except KeyboardInterrupt:
        print("\nKeyboardInterrupt recebido. Encerrando MNHI 4.0...")
    finally:
        try:
            core.shutdown()
            print(f"Memoria salva em {args.memory_file}")
        except Exception as exc:
            print(f"Falha ao salvar memoria em {args.memory_file}: {exc}")


if __name__ == "__main__":
    main()
