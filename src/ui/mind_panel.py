from __future__ import annotations

import json
from pathlib import Path

import pygame


WIDTH, HEIGHT = 800, 600
FPS = 10
DEFAULT_NODE_COUNT = 64
STATE_PATH = Path(__file__).resolve().parent.parent / "mind_panel_state.json"

BLACK = (10, 10, 15)
WHITE = (255, 255, 255)
DARK_GRAY = (40, 40, 45)
CYAN_GLOW = (0, 255, 255)
GREEN = (50, 255, 50)
YELLOW = (255, 210, 90)


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _safe_load_state(path: Path) -> dict:
    try:
        if not path.exists():
            return {}
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _extract_node_positions(payload: dict) -> dict[int, tuple[float, float]]:
    nodes = payload.get("nodes")
    positions: dict[int, tuple[float, float]] = {}

    if isinstance(nodes, list):
        for idx, node in enumerate(nodes):
            if not isinstance(node, dict):
                continue
            node_id = node.get("id", idx)
            try:
                node_id_i = int(node_id)
                x = float(node.get("x", idx % 8))
                y = float(node.get("y", idx // 8))
            except (TypeError, ValueError):
                continue
            positions[node_id_i] = (x, y)

    if positions:
        return positions

    for i in range(DEFAULT_NODE_COUNT):
        positions[i] = (float(i % 8), float(i // 8))
    return positions


def _extract_active_nodes(payload: dict) -> list[int]:
    active_ids: list[int] = []
    thought_trace = payload.get("thought_trace")

    if isinstance(thought_trace, dict):
        sequence = thought_trace.get("sequence")
        if isinstance(sequence, list):
            for step in sequence:
                if not isinstance(step, dict):
                    continue
                node_id = step.get("node_id")
                if isinstance(node_id, int):
                    active_ids.append(node_id)
                elif isinstance(node_id, str) and node_id.isdigit():
                    active_ids.append(int(node_id))

    meta = payload.get("meta")
    if isinstance(meta, dict):
        thought_input = meta.get("thought_input")
        node_names = payload.get("node_names")
        if isinstance(thought_input, str) and isinstance(node_names, dict):
            for raw_id, name in node_names.items():
                if name == thought_input:
                    try:
                        active_ids.append(int(raw_id))
                    except (TypeError, ValueError):
                        pass
                    break

    return active_ids


def _extract_metrics(payload: dict) -> tuple[float, float, str]:
    intent_level = 0.0
    coherence = 0.0
    command_text = "IDLE / AGUARDANDO"

    if not payload:
        return intent_level, coherence, command_text

    try:
        intent_level = _clamp(float(payload.get("intent_level", 0.0)), 0.0, 1.0)
    except (TypeError, ValueError):
        intent_level = 0.0

    meta = payload.get("meta")
    if isinstance(meta, dict):
        try:
            coherence = _clamp(float(meta.get("ressonancia_progenitor", 0.0)), 0.0, 1.0)
        except (TypeError, ValueError):
            coherence = 0.0

    thought_trace = payload.get("thought_trace")
    if isinstance(thought_trace, dict):
        try:
            coherence = _clamp(float(thought_trace.get("coherence_score", coherence)), 0.0, 1.0)
        except (TypeError, ValueError):
            pass

        natural = thought_trace.get("natural_language")
        if isinstance(natural, str) and natural.strip():
            text = natural.strip()
            command_text = text[:34] + "..." if len(text) > 37 else text
            return intent_level, coherence, command_text

    if intent_level > 0.7:
        command_text = "FOCO CONFIRMADO"
    elif intent_level > 0.2:
        command_text = "TRANSICAO"

    return intent_level, coherence, command_text


def _draw_hud(surface: pygame.Surface, intent_level: float, coherence: float, command_text: str) -> None:
    pygame.draw.rect(surface, DARK_GRAY, (40, HEIGHT - 240, 20, 200))
    bar_height = int(intent_level * 200)
    color = GREEN if intent_level > 0.7 else CYAN_GLOW
    pygame.draw.rect(surface, color, (40, HEIGHT - 40 - bar_height, 20, bar_height))

    font = pygame.font.SysFont("Courier", 16, bold=True)
    text = font.render("INTENT", True, WHITE)
    surface.blit(text, (25, HEIGHT - 265))

    pygame.draw.circle(surface, DARK_GRAY, (WIDTH - 150, 100), 40, 2)
    radius = int(coherence * 40)
    if radius > 0:
        pygame.draw.circle(surface, CYAN_GLOW, (WIDTH - 150, 100), radius)
    coh_text = font.render(f"COHERENCE: {coherence:.2f}", True, WHITE)
    surface.blit(coh_text, (WIDTH - 230, 150))

    cmd_font = pygame.font.SysFont("Courier", 18, bold=True)
    cmd_color = GREEN if "CONFIRMADO" in command_text else YELLOW
    cmd_surface = cmd_font.render(command_text, True, cmd_color)
    surface.blit(cmd_surface, (WIDTH - 360, HEIGHT // 2))


def _draw_nodes(surface: pygame.Surface, positions: dict[int, tuple[float, float]], brightness: dict[int, float]) -> None:
    xs = [pos[0] for pos in positions.values()]
    ys = [pos[1] for pos in positions.values()]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    span_x = max(1.0, max_x - min_x)
    span_y = max(1.0, max_y - min_y)

    origin_x, origin_y = WIDTH // 2 - 180, HEIGHT // 2 - 180
    draw_w, draw_h = 360, 360

    for node_id, (x, y) in positions.items():
        cx = int(origin_x + ((x - min_x) / span_x) * draw_w)
        cy = int(origin_y + ((y - min_y) / span_y) * draw_h)
        b = int(brightness.get(node_id, 0.0))

        if b > 20:
            color = (min(255, b + 40), min(255, b + 40), 255)
            pygame.draw.circle(surface, color, (cx, cy), 10)
            pygame.draw.circle(surface, CYAN_GLOW, (cx, cy), 14, 1)
        else:
            pygame.draw.circle(surface, DARK_GRAY, (cx, cy), 7)


def main() -> None:
    pygame.init()
    win = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Mind Control Panel (Neuromorphic UI)")
    clock = pygame.time.Clock()

    running = True
    neuron_brightness: dict[int, float] = {}

    while running:
        win.fill(BLACK)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        payload = _safe_load_state(STATE_PATH)
        positions = _extract_node_positions(payload)

        for node_id in positions.keys():
            neuron_brightness.setdefault(node_id, 0.0)

        active_ids = _extract_active_nodes(payload)
        for node_id in active_ids:
            if node_id in neuron_brightness:
                neuron_brightness[node_id] = 255.0

        for node_id in list(neuron_brightness.keys()):
            neuron_brightness[node_id] = max(0.0, neuron_brightness[node_id] - 15.0)

        intent_level, coherence, command_text = _extract_metrics(payload)

        _draw_nodes(win, positions, neuron_brightness)
        _draw_hud(win, intent_level, coherence, command_text)

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()


if __name__ == "__main__":
    main()
