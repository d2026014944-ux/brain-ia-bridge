# Sincronia Bio-Sintética Subliminar (SBSS)

Bridge de integração entre EEG (Crown/Neurosity), HyperBitnet e TRIBE para
experimentos de biofeedback e decodificação de intenção em tempo real.

## Objetivo

Processar sinais cerebrais através de uma versão adaptativa de thresholds,
utilizando modelos inspirados em topologia de redes lógicas (HyperBitnet) e
extraindo comandos viáveis para sistemas externos via integração TRIBE.

## Overview

Neuroadaptive MVP pipeline:

```
EEG → features (focus / calm / gamma) → adaptive threshold → HyperBitnet → TRIBE
```

## Repository Structure

```
src/
├── core/
│   ├── eeg_adapter.py        # Compute scalar score from EEG features (auto-detect domain)
│   ├── calibration.py        # Adaptive threshold calibration from baseline
│   ├── hyperbitnet.py        # HyperBitnet: pure-Python + advanced graph mode
│   ├── fusion.py             # Fuse classical + quantum-inspired state vectors
│   ├── lif_neuron.py         # Leaky Integrate-and-Fire neuron model
│   ├── spiking_network.py    # Event-driven spiking neural network
│   ├── synapse_stdp.py       # Spike-timing-dependent plasticity
│   ├── subliminal_learning.py# AI Teacher for subliminal learning
│   ├── sensory_encoder.py    # Rate coding encoder (HyperBitnet → spikes)
│   ├── canonical_hasher.py   # Canonical hashing for state persistence
│   ├── noma_bridge.py        # NomaParser: NOMA_NEURAL telemetry extraction
│   ├── _native_core.cpp      # C++ accelerated spiking engine
│   ├── _native_loader.py     # Graceful native/fallback loader
│   └── _native_fallback.py   # Pure-Python fallback for C++ core
├── governance/
│   └── pentacosagram.py      # Pentacosagram governance network
├── integration/
│   ├── tribe_adapter.py      # Map intent vector to TRIBE high-level command
│   └── neurosity_adapter.py  # Neurosity SDK adapter (stub + mock stream)
├── main.py                   # Unified demo: MVP + advanced simulation
├── realtime_loop.py          # Realtime loop with sliding window filter
├── run_realtime_neurosity.py # Realtime loop using the Neurosity adapter
├── run_teacher.py            # AI Teacher loop (40Hz gamma + STDP growth)
├── run_noma_symbiosis.py     # Noma Symbiosis: live brain with telemetry input
├── brain_orchestrator.py     # Full brain orchestrator (HyperBitnet + Spiking)
├── mind_panel.py             # Mind Panel: 8x8 synaptic strength WebSocket UI
└── mind_panel.html           # Mind Panel frontend
tests/
├── conftest.py
├── test_lif_neuron.py
├── test_spiking_network.py
├── test_synapse_stdp.py
├── test_sensory_encoder.py
├── test_canonical_hasher.py
├── test_noma_bridge.py
├── test_pentacosagram.py
├── test_persistence.py
├── test_subliminal_learning.py
└── test_network_dynamics.py
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

### 4. Run the unified demo

```bash
python src/main.py
```

This runs two phases:
1. **MVP Demo**: Calibration with Neurosity-style features + live inference + TRIBE commands.
2. **Advanced Simulation**: HyperBitnet quantum graph + BitNet 1.58b matrix fusion (requires numpy/networkx/scipy).

### 5. Run the simulated realtime loop

```bash
python src/realtime_loop.py
```

Press **Ctrl+C** to stop early (the loop finishes automatically after ~5 s).

### 6. Run the Neurosity adapter (mock stream)

```bash
python src/run_realtime_neurosity.py
```

> `use_mock_stream=True` is enabled by default.  No real hardware is required.

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

### 9. Run MNHI runtime loop (single entrypoint)

```bash
python src/main.py
```

This process updates `src/mind_panel_state.json` every 100 ms so Mind Panel reflects
spike-driven activity and synaptic strength changes in real time.

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

### 10. Run Noma Symbiosis (live brain with telemetry)

```bash
python src/run_noma_symbiosis.py
```

Paste `[NOMA_NEURAL]...[/NOMA_NEURAL]` blocks into stdin to modulate
the spiking network in real time.

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

## Architecture: Graceful Degradation

The project follows a **graceful degradation** pattern:

| Component | Full Mode | Fallback Mode |
|-----------|-----------|---------------|
| **Spiking Engine** | C++ via pybind11 | Pure-Python `_native_fallback.py` |
| **HyperBitnet** | networkx + numpy + scipy graph | Pure-Python lists + cosine projection |
| **Scoring** | Auto-detects Neurosity (focus/gamma/calm) | Supports legacy (alpha/beta) |

This ensures the system runs on any Python 3.10+ environment while unlocking
maximum performance when scientific computing libraries are available.

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
- [ ] SBSS: Sincronia Bio-Sintética Subliminar (affect encoder + resonance handshake)
- [ ] Live score / state visualisation

---

## Ethical notice

This project is experimental and educational.

- Do **not** use for medical diagnosis.
- Do **not** use for clinical decision-making.
- Do **not** use for controlling safety-critical devices without formal
  validation and safety protocols.
