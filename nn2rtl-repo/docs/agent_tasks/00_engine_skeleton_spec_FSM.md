# Engine skeleton — FSM specification

State machine for the shared compute engine. Encoded as a 3-bit `state`
register inside `shared_engine` (see
[output/rtl/shared_engine_skeleton.v](../../output/rtl/shared_engine_skeleton.v)).
Authoritative description for what each state does, which transition
conditions fire, and which sub-block is active during it.

The skeleton encodes **next-state logic only**. Per-state data-path
control (driving sub-block enables, weight/bias muxes, accumulator
clears, output writeback strobes) is owned by the Wave 2 tasks that
implement each sub-block.

## State encoding

| Name | Encoding | Role |
| --- | --- | --- |
| ST_IDLE | 3'd0 | No layer dispatched. `engine_busy = 0`, `engine_done = 0`. Waits for `engine_start_pulse`. |
| ST_LOAD_CONFIG | 3'd1 | One-cycle latch of the cfg_* wires into address_generator's internal registers. |
| ST_RUN | 3'd2 | MAC accumulate loop. address_generator walks K_TOTAL; bram_to_stream_bridge feeds mac_array one IC byte/cycle. |
| ST_REQUANT | 3'd3 | 3-stage requant pipeline drains the current OC pass (256 lanes). |
| ST_DRAIN | 3'd4 | bram_to_stream_bridge writes the assembled output beat to the activation-output BRAM. |
| ST_DONE | 3'd5 | `engine_done = 1`. Waits for scheduler to deassert `engine_start`. |

## Transition arcs

Each row is one outgoing arc. Every state has at least one. Arcs marked
`(else)` are the implicit hold-in-state — fall-through when no listed
condition fires.

| From | To | Condition | Sub-block that emits the condition |
| --- | --- | --- | --- |
| ST_IDLE | ST_LOAD_CONFIG | `engine_start_pulse` | config_register_block |
| ST_IDLE | ST_IDLE | (else) | — |
| ST_LOAD_CONFIG | ST_RUN | unconditional, 1 cycle after entering ST_LOAD_CONFIG | (FSM-internal) |
| ST_RUN | ST_REQUANT | `ag_mac_done` (all K_TOTAL accumulations for current OC pass complete) | address_generator |
| ST_RUN | ST_RUN | (else) | — |
| ST_REQUANT | ST_RUN | `requant_valid_out && oc_pass_idx != MAX_OC/MAC_COUNT − 1` (more OC slices still owed for the current output pixel) | requant_pipeline |
| ST_REQUANT | ST_DRAIN | `requant_valid_out && oc_pass_idx == MAX_OC/MAC_COUNT − 1` (last OC slice of this pixel; output beat ready) | requant_pipeline |
| ST_REQUANT | ST_REQUANT | (else) | — |
| ST_DRAIN | ST_RUN | `!bridge_busy && !ag_pixel_done` (output beat written, more pixels owed) | bram_to_stream_bridge, address_generator |
| ST_DRAIN | ST_DONE | `!bridge_busy && ag_pixel_done` (last pixel written, layer complete) | bram_to_stream_bridge, address_generator |
| ST_DRAIN | ST_DRAIN | (else) | — |
| ST_DONE | ST_IDLE | `!engine_start` (scheduler has acknowledged completion by deasserting the start pin) | external |
| ST_DONE | ST_DONE | (else) | — |

## Per-state sub-block activity

| State | Active sub-block(s) | What it drives |
| --- | --- | --- |
| ST_IDLE | config_register_block | Services AXI4-Lite reads/writes; samples engine_start pin. Holds engine_busy_ext low. |
| ST_LOAD_CONFIG | config_register_block, address_generator | Final config wires settle; address_generator latches them into its own scratch counters. |
| ST_RUN | address_generator, bram_to_stream_bridge (read half), mac_array | address_generator emits weight/bias/activation addresses; bridge presents one IC byte/cycle; mac_array accumulates across 256 OC lanes. |
| ST_REQUANT | requant_pipeline | 3-stage bias+scale+shift+saturate across all 256 lanes in parallel. mac_array's `acc_out` is captured into stage-1 register on entry. |
| ST_DRAIN | bram_to_stream_bridge (write half) | Packs 256 INT8 outputs into one 2048-bit BRAM beat; pulses act_out_wr_en. |
| ST_DONE | config_register_block | Holds engine_done_ext high; AXI4-Lite STATUS register reads back `done=1`. |

## Notes for Wave 2 implementers

- **mac_clear timing.** Assert `mac_clear` for one cycle on entering
  ST_RUN at the start of each OC pass (i.e. coming from ST_LOAD_CONFIG
  or from the ST_REQUANT → ST_RUN arc). This zeroes the 256
  accumulators before the new K_TOTAL accumulation. Do NOT assert it
  while ST_RUN is mid-loop or you destroy the partial sum.
- **bias / scale registers are stable across OC passes.** The bias
  values change per OC pass (different OC channel range); the
  `cfg_scale_mult` / `cfg_scale_shift` are layer-wide and do not
  change inside one layer's dispatch.
- **`ag_pixel_done` is sampled in ST_DRAIN**, not in ST_RUN. The
  address_generator pulses it the cycle after the last output pixel's
  REQUANT completes; sampling it earlier would race the OC-pass
  counter wrap.
- **`engine_start` is an external pin.** The FSM uses the registered
  `engine_start_pulse` to leave ST_IDLE but uses the raw external pin
  to leave ST_DONE so the scheduler controls the IDLE↔DONE handshake
  end-to-end. This matches the
  [01_context.md `[INVARIANT:READY_IN_GATING]`](../../knowledge/patterns/protected/01_context.md)
  convention of using direct external strobes for cross-clock
  handshakes.
- **Hold-in-state arcs are NOT optional.** Every state has a `(else)`
  arc that keeps `state <= state` until its transition condition
  fires. The skeleton's combinational next-state block (`always @*`)
  defaults `next_state = state` at the top so this is automatic; do
  not add explicit "fall-through to default" arcs.
