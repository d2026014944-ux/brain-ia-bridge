from __future__ import annotations

import argparse
import json
import random
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

from core.lif_neuron import LIFNeuron
from core.spiking_network import SpikingNetwork


GRID_SIZE = 8
N_NEURONS = GRID_SIZE * GRID_SIZE
HTML_PATH = Path(__file__).resolve().parent / "mind_panel.html"
DEFAULT_EXTERNAL_STATE_PATH = Path(__file__).resolve().parent / "mind_panel_state.json"


def _node_xy(node_id: int) -> tuple[float, float]:
    row = node_id // GRID_SIZE
    col = node_id % GRID_SIZE
    return float(col), float(row)


def build_network() -> SpikingNetwork:
    net = SpikingNetwork(learning_enabled=True)

    for node_id in range(N_NEURONS):
        net.add_neuron(node_id=node_id, neuron_instance=LIFNeuron(v_thresh=1.0, tau=20.0, refractory_period=5.0))

    # 8x8 mesh: connect right and bottom neighbors.
    for node_id in range(N_NEURONS):
        row = node_id // GRID_SIZE
        col = node_id % GRID_SIZE

        if col < GRID_SIZE - 1:
            right = node_id + 1
            net.add_connection(pre_id=node_id, post_id=right, weight=0.32, delay_ms=1.0)
        if row < GRID_SIZE - 1:
            down = node_id + GRID_SIZE
            net.add_connection(pre_id=node_id, post_id=down, weight=0.28, delay_ms=1.0)

    return net


def render_panel_html() -> None:
    if HTML_PATH.exists():
        return

    html = """<!doctype html>
<html lang=\"pt-BR\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>Mind Panel - Forca Sinaptica</title>
  <style>
    :root {
      --bg-1: #0a1325;
      --bg-2: #111f38;
      --ink: #e5ecff;
      --muted: #96a8cf;
      --accent: #43d9bd;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      font-family: \"Space Grotesk\", \"Segoe UI\", sans-serif;
      color: var(--ink);
      background:
        radial-gradient(1200px 700px at 20% 15%, #18325a 0%, transparent 60%),
        radial-gradient(1000px 600px at 85% 85%, #142749 0%, transparent 55%),
        linear-gradient(165deg, var(--bg-1), var(--bg-2));
      display: grid;
      place-items: center;
      padding: 16px;
    }
    .card {
      width: min(96vw, 980px);
      border: 1px solid #244166;
      border-radius: 18px;
      padding: 14px 14px 10px 14px;
      background: rgba(9, 18, 35, 0.78);
      backdrop-filter: blur(4px);
      box-shadow: 0 12px 42px rgba(0, 0, 0, 0.45);
    }
    .head {
      display: flex;
      justify-content: space-between;
      align-items: baseline;
      gap: 10px;
      margin-bottom: 10px;
    }
    .title {
      font-size: clamp(1rem, 2vw, 1.25rem);
      letter-spacing: 0.04em;
      font-weight: 700;
      text-transform: uppercase;
    }
    .meta {
      font-size: 0.9rem;
      color: var(--muted);
      display: flex;
      gap: 16px;
      flex-wrap: wrap;
    }
    .meta strong { color: var(--accent); font-weight: 700; }
    canvas {
      width: 100%;
      aspect-ratio: 1.25;
      border-radius: 12px;
      border: 1px solid #2a4f75;
      background:
        linear-gradient(rgba(255,255,255,0.02), rgba(255,255,255,0.01)),
        repeating-linear-gradient(90deg, rgba(90,140,210,0.06) 0 1px, transparent 1px 48px),
        repeating-linear-gradient(0deg, rgba(90,140,210,0.06) 0 1px, transparent 1px 48px);
    }
  </style>
</head>
<body>
  <section class=\"card\">
    <div class=\"head\">
      <div class=\"title\">Mind Panel · Forca Sinaptica</div>
      <div class=\"meta\">
        <span>learning: <strong id=\"learning\">on</strong></span>
        <span>peso medio: <strong id=\"wmean\">0.000</strong></span>
        <span>ultima atualizacao: <strong id=\"ts\">-</strong></span>
      </div>
    </div>
    <canvas id=\"mind\" width=\"940\" height=\"720\"></canvas>
  </section>

  <script>
    const canvas = document.getElementById('mind');
    const ctx = canvas.getContext('2d');
    const learningEl = document.getElementById('learning');
    const wmeanEl = document.getElementById('wmean');
    const tsEl = document.getElementById('ts');

    function toCanvasPos(x, y) {
      const padX = 90;
      const padY = 60;
      const width = canvas.width - padX * 2;
      const height = canvas.height - padY * 2;
      return {
        x: padX + (x / 7) * width,
        y: padY + (y / 7) * height,
      };
    }

    function draw(data) {
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      const nodes = data.nodes || [];
      const edges = data.synapses || [];

      for (const edge of edges) {
        const pre = nodes.find((n) => n.id === edge.pre_id);
        const post = nodes.find((n) => n.id === edge.post_id);
        if (!pre || !post) continue;

        const p1 = toCanvasPos(pre.x, pre.y);
        const p2 = toCanvasPos(post.x, post.y);
        const w = Math.max(0, Math.min(1, Number(edge.weight || 0)));
        const alpha = 0.15 + w * 0.75;
        const width = 0.6 + w * 3.8;

        ctx.beginPath();
        ctx.moveTo(p1.x, p1.y);
        ctx.lineTo(p2.x, p2.y);
        ctx.lineWidth = width;
        ctx.strokeStyle = `rgba(67, 217, 189, ${alpha})`;
        ctx.shadowBlur = 2 + w * 12;
        ctx.shadowColor = `rgba(67, 217, 189, ${0.25 + w * 0.7})`;
        ctx.stroke();
      }

      ctx.shadowBlur = 0;
      for (const node of nodes) {
        const p = toCanvasPos(node.x, node.y);
        ctx.beginPath();
        ctx.arc(p.x, p.y, 6, 0, Math.PI * 2);
        ctx.fillStyle = 'rgba(229, 236, 255, 0.95)';
        ctx.fill();
      }
    }

    async function refresh() {
      try {
        const res = await fetch('/state', { cache: 'no-store' });
        const data = await res.json();

        const syn = data.synapses || [];
        const mean = syn.length ? syn.reduce((acc, s) => acc + Number(s.weight || 0), 0) / syn.length : 0;

        learningEl.textContent = data.learning_enabled ? 'on' : 'off';
        wmeanEl.textContent = mean.toFixed(3);
        tsEl.textContent = new Date((data.timestamp || 0) * 1000).toLocaleTimeString('pt-BR');

        draw(data);
      } catch (_) {
        // Keep last frame if fetch fails momentarily.
      }
    }

    refresh();
    setInterval(refresh, 180);
  </script>
</body>
</html>
"""
    HTML_PATH.write_text(html, encoding="utf-8")


class _StateStore:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._state: dict[str, Any] = {
            "timestamp": time.time(),
            "learning_enabled": True,
            "nodes": [],
            "synapses": [],
        }

    def set(self, payload: dict[str, Any]) -> None:
        with self._lock:
            self._state = payload

    def get(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._state)


def run_simulation(store: _StateStore, stop: threading.Event, hz: float = 12.0) -> None:
    net = build_network()
    random.seed(7)

    step_ms = 1000.0 / hz
    current_t = 0.0

    nodes = []
    for node_id in range(N_NEURONS):
        x, y = _node_xy(node_id)
        nodes.append({"id": node_id, "x": x, "y": y})

    while not stop.is_set():
        active = [random.randint(0, N_NEURONS - 1) for _ in range(5)]
        for node_id in active:
            net.schedule_event(time_ms=current_t, target_id=node_id, weight=1.25)

        net.run_until_empty()

        store.set(
            {
                "timestamp": time.time(),
                "learning_enabled": bool(net.learning_enabled),
                "nodes": nodes,
                "synapses": net.get_synaptic_strengths(),
            }
        )

        current_t += step_ms
        time.sleep(1.0 / hz)


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(json.dumps(payload), encoding="utf-8")
    temp_path.replace(path)


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _emotion_from_amplitude(amplitude: float) -> str:
    if amplitude >= 0.85:
        return "euforia_sincronica"
    if amplitude >= 0.65:
        return "foco_elevado"
    if amplitude >= 0.40:
        return "atencao_estavel"
    if amplitude >= 0.20:
        return "calma"
    return "repouso"


def _parse_telemetry_text(text: str) -> dict[str, float] | None:
    if not isinstance(text, str):
        return None

    block_match = re.search(r"\[\s*NOMA_NEURAL\s*\](.*?)\[\s*/\s*NOMA_NEURAL\s*\]", text, re.IGNORECASE | re.DOTALL)
    if block_match is None:
        return None

    block = block_match.group(1)

    def _read_number(pattern: str) -> float | None:
        match = re.search(pattern, block, re.IGNORECASE)
        if match is None:
            return None
        numeric_raw = match.group(1).strip().replace(",", ".")
        try:
            return float(numeric_raw)
        except ValueError:
            return None

    freq_hz = _read_number(r"frequ(?:e|ê|é)ncia\s*_?\s*dominante\s*:\s*([+-]?\d+(?:[.,]\d+)?)")
    amplitude = _read_number(r"amplitude\s*_?\s*afetiva\s*:\s*([+-]?\d+(?:[.,]\d+)?)")
    resonance = _read_number(r"resson(?:a|â)ncia\s*_?\s*progenitor\s*:\s*([+-]?\d+(?:[.,]\d+)?)")

    if freq_hz is None or amplitude is None:
        return None

    return {
        "freq_hz": freq_hz,
        "amplitude_afetiva": amplitude,
        "ressonancia_progenitor": 0.9 if resonance is None else resonance,
    }


def _safe_load_external_state(path: Path) -> dict[str, Any] | None:
    try:
        if not path.exists():
            return None
        raw = path.read_text(encoding="utf-8")
        payload = json.loads(raw)
        if not isinstance(payload, dict):
            return None
        return payload
    except (OSError, json.JSONDecodeError):
        return None


def make_handler(store: _StateStore, external_state_path: Path | None = None):
    class Handler(BaseHTTPRequestHandler):
        def _write_json(self, status_code: int, payload: dict[str, Any]) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status_code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            if self.path == "/":
                body = HTML_PATH.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return

            if self.path == "/state":
                payload = None
                if external_state_path is not None:
                    payload = _safe_load_external_state(external_state_path)
                if payload is None:
                    payload = store.get()

                self._write_json(200, payload)
                return

            self.send_response(404)
            self.end_headers()

        def do_POST(self) -> None:
            if self.path != "/telemetry":
                self._write_json(404, {"ok": False, "error": "Endpoint nao encontrado"})
                return

            try:
                content_length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                content_length = 0

            if content_length <= 0:
                self._write_json(400, {"ok": False, "error": "Corpo vazio"})
                return

            try:
                raw_body = self.rfile.read(content_length)
                body = json.loads(raw_body.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                self._write_json(400, {"ok": False, "error": "JSON invalido"})
                return

            telemetry_text = body.get("text") if isinstance(body, dict) else None
            parsed = _parse_telemetry_text(telemetry_text)
            if parsed is None:
                self._write_json(400, {"ok": False, "error": "Bloco NOMA_NEURAL invalido"})
                return

            base_payload = None
            if external_state_path is not None:
                base_payload = _safe_load_external_state(external_state_path)
            if base_payload is None:
                base_payload = store.get()

            amplitude = _clamp(float(parsed["amplitude_afetiva"]), 0.0, 1.0)
            resonance = _clamp(float(parsed["ressonancia_progenitor"]), 0.0, 1.0)
            freq_hz = max(0.001, float(parsed["freq_hz"]))

            updated_payload = dict(base_payload)
            updated_meta = dict(updated_payload.get("meta", {}))
            updated_meta["freq_hz"] = freq_hz
            updated_meta["amplitude_afetiva"] = amplitude
            updated_meta["ressonancia_progenitor"] = resonance
            updated_meta["source"] = "mind_panel.telemetry"

            updated_payload["timestamp"] = time.time()
            updated_payload["intent_level"] = amplitude
            updated_payload["estado_emocional"] = _emotion_from_amplitude(amplitude)
            updated_payload["meta"] = updated_meta

            store.set(updated_payload)
            if external_state_path is not None:
                _atomic_write_json(external_state_path, updated_payload)

            self._write_json(200, {"ok": True, "payload": updated_payload})

        def log_message(self, format: str, *args: Any) -> None:
            return

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description="Mind Panel with live synaptic strength")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--state-file",
        default=str(DEFAULT_EXTERNAL_STATE_PATH),
        help="JSON file produced by run_teacher.py; if present, /state follows this file.",
    )
    parser.add_argument(
        "--disable-internal-sim",
        action="store_true",
        help="Disable built-in random simulation and rely only on --state-file.",
    )
    args = parser.parse_args()

    render_panel_html()

    store = _StateStore()
    stop = threading.Event()
    external_state_path: Path | None = None

    if args.state_file:
        external_state_path = Path(args.state_file).resolve()
        external_state_path.parent.mkdir(parents=True, exist_ok=True)

    if not args.disable_internal_sim:
        sim_thread = threading.Thread(target=run_simulation, args=(store, stop), daemon=True)
        sim_thread.start()
    else:
        print("Mind Panel internal simulation disabled.")

    server = HTTPServer((args.host, args.port), make_handler(store, external_state_path=external_state_path))
    print(f"Mind Panel running at http://{args.host}:{args.port}")
    if external_state_path is not None:
        print(f"Following external state file when available: {external_state_path}")

    try:
        server.serve_forever(poll_interval=0.2)
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        server.server_close()


if __name__ == "__main__":
    main()
