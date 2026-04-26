# Split architecture for spatial conv / maxpool modules

The monolithic "Foundry generates everything" approach fails on spatial
convolutions because the cycle-aligned coupling between coord_scheduler,
line buffer, shift-register window, and MAC pipeline is exactly the kind
of RTL LLMs get wrong. This document pins the contract so the generated
top-level shrinks to a thin wiring wrapper over three library modules.

## Components

### `coord_scheduler` (library, handwritten)

Owns all coordinate arithmetic: row/col counters, wrap-at-IW-1+PW,
stride/padding divisibility, real vs padded region, output-completion
count. Exposes:
- `advance` — combinational, high this cycle when the scheduler moves
  forward (handshake in REAL region, pad_step in PADDED region).
- `output_fires` — REGISTERED one-cycle pulse emitted the cycle AFTER an
  advance past a firing coord. The cycle this pulses, `advance` is 0
  (scheduler internally freezes on output_fires so the datapath can
  latch start_mac and transition to ST_MAC on the next posedge).
- `stall_in` — external input, driven by top-level as `= mac_busy`. That
  is all. The scheduler's own internal `eff_stall = stall_in ||
  output_fires` handles firing-coord freezing without external help.

### `line_buf_window` (library, handwritten)

Owns the receptive-field state: KH-row line buffer, KH×KW×IC registered
shift-register window, and the vertical rotation across input-row
transitions. Unit-testable in isolation. Consumes the scheduler's
`advance` output directly — no replicated logic in the top-level.

**Storage architecture (post-BRAM rework).** The line buffer is KH
separate per-slot single-port memories (`reg [IC*8-1:0] mem [0:IW+PW-1]`
with `(* ram_style = "block" *)`), one BRAM18/BRAM36 instance per slot
on Artix-7. The legacy bulk vertical shift across rows is replaced by
a single-counter `oldest_slot` rotation: each slot's role (oldest, …,
currently-writing) is determined by `(oldest_slot + i) mod KH`, and at
row_wrap the pointer increments instead of shuffling 14k flip-flops.

**Multi-frame correctness.** BRAM cells retain frame-N-1 data across a
new `frame_start`. A `row_valid[KH-1:0]` register, cleared at
`frame_start` and set/cleared at each `row_wrap`, masks reads from any
slot that has not been fully written in the current frame to zero.
This is the sole guarantee that top-pad rows read as zero — it is
load-bearing for cross-frame correctness, so test changes to the
rotation/row_valid logic against the tiny-geometry testbench in
`rtl_library/test/line_buf_window_tb.cpp`.

**Right-pad correctness.** Writes are gated by `!right_padded`, so
`mem_s[MAX_IN_COL]` is never touched. Vivado initialises BRAMs to
zero at FPGA configuration, so reads of the right-pad column return
zero without an explicit gate in the read path.

**Latency.** This rewrite preserves the legacy 1-cycle read pipeline
exactly. The legacy async-read+window-FF chain became sync-read into
the BRAM's output register `q_array[s]`, which directly drives the
window's KW-1 column as a wire. Same total cycle count;
`compute_conv2d_latency_cycles` in `scripts/golden_impl.py` is
unchanged.

### `conv_datapath` (library, handwritten)

Owns the MAC pipeline only: serialized MP-lane MAC loop over `window`,
BIAS, SCALE (current fixed-point half-up/toward-positive approximation,
INT8 saturation), output packing. Emits
`valid_out` on the last ST_OUTPUT cycle and exposes `mac_busy` so the
top-level can drive `stall_in = mac_busy`.

### Top-level (Foundry-generated — ~60 lines)

Just instantiates the three submodules with LayerIR parameters and wires
the canonical 7-port interface. No cycle-aligned FSM logic in this
module; one small always block for `start_pulse` generation.

## Firing-coord timing (pixel-delivery-safe)

The bug the registered-output_fires design fixes:

- Old: `output_fires` was combinational on `at_output_coord`; normal
  `advance` was gated by `!at_output_coord`. When the scheduler first
  reached a firing coord, handshake completed from upstream's view
  (`ready_in = 1`) but nothing internal captured the pixel. When
  `mac_done` later pulsed, the scheduler advanced past the firing coord
  but line_buf_window wrote the NEXT pixel into the firing coord's slot.
  First-output RF correct; every subsequent one misaligned.

- New: `output_fires` is registered, pulses one cycle AFTER advance past
  firing coord. Advance is NOT gated by `at_output_coord`. On the cycle
  the scheduler reaches the firing coord (pre-edge), handshake fires
  (eff_stall = 0 since output_fires = 0). Pixel delivered, scheduler
  advances, `output_fires <= 1` registers. On the NEXT cycle,
  `output_fires = 1` → `eff_stall = 1` → scheduler freezes. Datapath
  observes `start_mac = 1`, transitions ST_IDLE → ST_MAC next posedge.
  Then `mac_busy = 1` keeps scheduler frozen across MAC pipeline. When
  `mac_busy` drops, scheduler resumes advance on next handshake.

No `mac_done` signal, no `release_advance` special case.

## Top-level wiring (what Foundry generates)

See `knowledge/patterns/03_conv3x3_pad1.md` for the concrete wiring
template. Summary:
- 15 localparams derived from LayerIR
- 1 small always block for `start_pulse`
- 3 module instantiations
- `assign ready_in = sched_ready_in;`
- `assign stall_in = mac_busy;`
- That's it.

## Edge cases this handles

- **Multi-frame**: scheduler's `start` re-arms on each `out_frame_done`;
  top-level re-pulses start on next valid_in.
- **Pointwise (1×1)**: uses the older monolithic
  `conv1x1_passing_reference.v` (simpler, no line buffer / window / scheduler
  needed). Spatial convs all use the split architecture.
- **Stride > 1**: handled entirely inside coord_scheduler's
  `row_stride_ok` / `col_stride_ok`. Line_buf and datapath don't care.
- **Padding**: all of it (top / bottom / left / right) handled by
  line_buf_window's zero-load on padded cycles + scheduler's real vs
  padded region classification.
