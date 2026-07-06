# snn-pong-rstdp

Spiking neural network (Brian2) learns to play Pong via reward-modulated STDP (R-STDP) — no backprop, no gradients. Biologically-inspired conductance-based LIF neurons with dopamine-gated eligibility traces as the only learning signal.

## Run

```
python main.py
```

Runs a live Pygame window for up to 6000s (Ctrl+C to stop early), then dumps telemetry to `brain_telemetry.h5` and opens analysis plots. Press `T` in-game to toggle turbo (headless-speed) mode vs. visual mode.

## Architecture

- **`brain.py`** — `Brain_rstdp`: builds the Brian2 `Network` and owns the game loop (`network_operation` callbacks running inside the simulation clock, not a separate loop). This is where almost all interesting logic lives.
- **`game.py`** — `PongGame`: single-paddle Pong. No player 2 — the ball bounces off the right wall's paddle only; missing = episode reset. Feedback is deliberately coarse: the y-axis is divided into `num_chunks` bins that grow by +1 each time the agent clears a level (75% hit-rate over `hits_per_chunk` hits per chunk), i.e. curriculum learning via progressively finer motor precision required.
- **`telemitry.py`** *(sic — filename typo, kept for compatibility)* — `BrainTelemetry`: chunked HDF5 (LZF-compressed) logger, flushes every `chunk_size` steps.
- **`analysis.py`** — post-run matplotlib plots: cellular traces, synaptic weight heatmaps, tracking-error/dopamine trend with per-level regression, spike raster.

## Network topology

`topology = [100, 125]` — sensory layer (100 exc, ball-position encoded as a Gaussian bump of Poisson rates) → motor layer (100 exc + 25 inh). Motor spikes are read out via a windowed, chunk-histogram winner-take-all with hysteresis (paddle only switches chunk if the new one has ≥1.25× the spike count of the current one — damps jitter).

Key mechanisms:
- **R-STDP**: eligibility trace `c` (STDP-driven) × dopamine `D` (reward-driven) → weight update every 10ms in a `network_operation`, not Brian2's native synaptic on_pre/on_post weight change. Soft weight bounds, small constant decay.
- **iSTDP**: local inhibitory feedback per area for excitation/inhibition balance (homeostasis), separate from R-STDP.
- **Adaptive thresholds**: per-neuron `v_thresh` creeps up on spiking, decays back to baseline — a second homeostatic mechanism.
- **Episodic normalization**: on every miss, incoming synaptic weight sums per post-neuron are pulled softly toward `w_target`.
- **Reward shaping** in `game.py`: hit = +1, moving the paddle toward the ball's chunk = +0.075, scaled by a dynamic per-chunk multiplier that boosts underperforming chunks — a form of automatic curriculum/attention.

## Conventions

- The entire codebase (comments, print statements, plot titles/labels) is English for publication — no German text should be reintroduced. Comments use a unified style (`# --- Section ---` headers, sentence-case inline notes).
- `main.py` currently reconstructs the recurrent-synapse block twice in `brain.py` (`_create_synapses`) — one version is a docstring-disabled experiment (fully recurrent layers) kept for reference above the active one (feedforward + optional bounded recurrence). Don't delete the disabled block without asking; it's a live experiment note, not dead code.
- No test suite, no requirements.txt/pyproject — dependencies are Brian2, numpy, pygame, h5py, matplotlib, installed ad hoc.
- Not currently a git repo.
