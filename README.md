# brain-ia-bridge

Bridge integrating EEG (Crown / Neurosity), HyperBitnet and TRIBE for
biofeedback experiments and real-time intent decoding.

## Overview

Neuroadaptive MVP pipeline:

```
EEG → features (focus / calm / gamma) → adaptive threshold → HyperBitnet → TRIBE
```

## If the repository looks "empty"

If you are on `main` and still do not see scaffold files (even when you see
"14 files changed" in the PR), the changes are still on a feature branch and
**have not been merged into `main` yet**.

Quick checklist:

1. Open the scaffold PR on GitHub and confirm it is **Merged**.
2. In your local clone, run:
   ```bash
   git checkout main
   git pull origin main
   ```
3. If you want to inspect the PR branch before merge:
   ```bash
   git fetch origin
   git checkout copilot/initialize-repository-with-mvp-scaffold
   ```

## Repository Structure

```
src/
├── core/
│   ├── eeg_adapter.py        # Compute scalar score from EEG features
│   ├── calibration.py        # Adaptive threshold calibration from baseline
│   ├── hyperbitnet.py        # HyperBitnet node state propagation
│   └── fusion.py             # Fuse classical + quantum-inspired state vectors
├── integration/
│   ├── tribe_adapter.py      # Map intent vector to TRIBE high-level command
│   └── neurosity_adapter.py  # Neurosity SDK adapter (stub + mock stream)
├── main.py                   # Simple one-shot MVP demo
├── realtime_loop.py          # Realtime loop with simulated EEG stream
└── run_realtime_neurosity.py # Realtime loop using the Neurosity adapter
requirements.txt
```

---

## Quickstart

### 1. Create a virtual environment

```bash
python -m venv .venv
```

### 2. Activate the environment

**Linux / macOS**
```bash
source .venv/bin/activate
```

**Windows (PowerShell)**
```powershell
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> All core functionality uses the Python standard library, so this step is a
> no-op until you add external packages (e.g. the real Neurosity SDK).

### 4. Run the simple MVP demo

```bash
python src/main.py
```

Expected output:

```
=== Calibration ===
{'baseline_mean': 0.322, 'baseline_std': 0.0226, 'low': 0.3446, 'high': 0.3672}

=== Live Inference ===
sample_1: {'score': 0.41, 'state': 1, 'intent_energy': ..., 'command': 'CONFIRM_INTENT'}
sample_2: {'score': 0.78, 'state': 1, 'intent_energy': ..., 'command': 'CONFIRM_INTENT'}
sample_3: {'score': 0.216, 'state': -1, 'intent_energy': ..., 'command': 'IDLE'}
```

### 5. Run the simulated realtime loop

```bash
python src/realtime_loop.py
```

Press **Ctrl+C** to stop early (the loop finishes automatically after ~5 s).

### 6. Run the Neurosity adapter (mock stream)

```bash
python src/run_realtime_neurosity.py
```

> `use_mock_stream=True` is enabled by default in
> `src/run_realtime_neurosity.py`.  No real hardware is required.

Press **Ctrl+C** to stop.

### 7. Run unit tests

```bash
pytest -q
```

### 8. Run Mind Panel (8x8 synaptic strength UI)

```bash
python src/mind_panel.py --state-file src/mind_panel_state.json
```

Open [http://127.0.0.1:8765](http://127.0.0.1:8765) to view the panel.

### 9. Run AI Teacher loop (40Hz gamma + STDP growth)

```bash
python src/run_teacher.py --state-file src/mind_panel_state.json
```

This process updates `src/mind_panel_state.json` every 100 ms so Mind Panel reflects
spikes and synaptic strength changes in real time.

### 10. Trigger Layer 4 causal thought decoding

With Mind Panel running, send a thought command to decode a causal path:

```bash
curl -X POST http://127.0.0.1:8765/telemetry \
   -H "Content-Type: application/json" \
   -d '{"text":"pensar: Chuva"}'
```

Expected behavior:

- The backend reconstructs the current graph from `/state`
- It injects a pulse at the requested concept
- It decodes the causal geodesic with Layer 4 (`ThoughtDecoder`)
- The panel shows `[PENSAMENTO]: Chuva -> Molhado -> Oceano` (example)
- The terminal prints the same chain for runtime observability

---

## Layer 4 Thought Decoder

The project now includes a causal thought decoder in [src/core/thought_decoder.py](src/core/thought_decoder.py).

Key APIs:

- `SpikingNetwork.run_and_trace()`
: Executes pending events and returns a deterministic firing trace.
- `read_thought(network, start_concept, energy, ...)`
: Injects a pulse, reads causal firing progression, maps IDs to concepts, and returns:
   - causal chain (`concept_chain`)
   - natural language rendering (Option C: template-based concrete clusters + generic abstract fallback)
   - `confidence_score`
   - `coherence_score`
   - `thermodynamic_state`

TDD coverage lives in [tests/test_thought_decoder.py](tests/test_thought_decoder.py), including:

- Canonical chain: `Chuva -> Molhado -> Oceano`
- Exclusion of disconnected node (`Fogo`)
- Geodesic redirection under controlled local distortion

---

## How calibration works

1. The system collects a resting-state baseline (e.g. 8 seconds).
2. It computes the mean and standard deviation of the EEG score.
3. Adaptive thresholds are derived:
   - `low  = mean + 1 × std` — transition boundary
   - `high = mean + 2 × std` — confirmation boundary

Discrete intent states:

| State | Meaning              |
|-------|----------------------|
| `-1`  | Disconnection / idle |
|  `0`  | Ambiguous / rising   |
|  `1`  | Confirmed intent     |

---

## Integrating the real Neurosity SDK

Open `src/integration/neurosity_adapter.py` and follow the `TODO` comments:

1. Replace the stub in `connect()` with real SDK authentication and device
   selection.
2. Subscribe to the `focus`, `calm`, and `brainwaves` observables.
3. Merge the async signals into the standard feature dict:
   ```python
   {"focus": 0..1, "calm": 0..1, "gamma": 0..1, "timestamp": float}
   ```
4. Update `requirements.txt` to pin the Neurosity SDK version.

---

## Roadmap

- [ ] Integrate real Crown / Neurosity SDK
- [ ] Persist session logs (CSV / JSON)
- [ ] Per-user intent training (e.g. start / confirm)
- [ ] Unit tests for the decision pipeline
- [ ] Live score / state visualisation

---

## Ethical notice

This project is experimental and educational.

- Do **not** use for medical diagnosis.
- Do **not** use for clinical decision-making.
- Do **not** use for controlling safety-critical devices without formal
  validation and safety protocols.
