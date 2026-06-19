# K1-MBV2 — FDCE→FDRE datapath recode, MobileNetV2 extension (2026-06-10)

**Applier:** `scripts/apply_k1_mbv2_ext.py` (anchor-asserted, idempotent
`[K1-MBV2]` marker, `--dry-run`, `.prek1m` backups, two-phase
validate-then-commit — NO file is written if ANY anchor drifts; per-file
encoding utf-8/cp1252 and EOL CRLF/LF preserved).

**Parent method:** `scripts/apply_k1_fdce_recode.py` +
`docs/agent_tasks/K1_FDCE_RECODE_ANALYSIS.md` (ResNet K1, commit `be16f61`,
91 files / ~975K FF, byte-exact + cycle-exact). This extension applies the
SAME register-class discipline to the **MBV2-OWN** files.

## 0. Scope derivation

`scripts/run_mbv2_synth.ts collectSources()` ships:

* `rtl_library/`: conv_datapath, conv_datapath_parallel, **conv_datapath_mp_k**,
  coord_scheduler, **line_buf_window**, retile_bridge
* `output/rtl/`: **shared_engine_skeleton** + engine/{**mac_array**,
  address_generator, config_register_block, **requant_pipeline**,
  bram_to_stream_bridge}
* `output/mobilenet-v2/rtl/*.v` minus `nn2rtl_top.v` (all-spatial duplicate) and
  snapshots — i.e. the engine top `nn2rtl_top_engine.v`, `nn2rtl_scheduler.v`,
  and the ~100 per-layer modules.

The 5 **bold** files are the SHARED set already recoded by ResNet K1 (verified:
all carry `[K1-FDCE]` markers in this tree) — excluded here. Everything else
that is *instantiated by the shipped engine top* and holds async-reset
datapath state is in scope.

Elaboration facts the safety analysis depends on (verified against
`nn2rtl_top_engine.v`):

* every node instance has `ENABLE_BACKPRESSURE(1)` (66 insts);
* the 6 final depthwise convs 878/884/890/896/902/908 are
  `NATIVE_TILED(1)` → their `g_in_native` + `g_emit_native` branches elaborate;
  the legacy/bp emitter branches do NOT;
* the 34 engine-dispatched pointwise `node_conv_*.v` files are NOT instantiated
  (the engine computes them) → zero FF contribution, untouched;
* `retile_scatter` is instantiated only by the EXCLUDED all-spatial
  `nn2rtl_top.v`; the engine top instantiates 7 × `retile_gather`;
* `bram_to_stream_bridge.v`, `conv_datapath.v`, `conv_datapath_parallel.v` are
  shipped but not instantiated by the MBV2 top (810 uses conv_datapath_mp_k);
* the top-module body itself (lines 15–3441) holds exactly ONE control FF
  (`sched_started_r`) — nothing to recode there.

## 1. Universal byte-exactness argument (inherited verbatim from ResNet K1)

The e2e gate runs Verilator `--x-initial 0` (FPGA power-on zeros). For every
register moved out of the reset clause: (1) power-on value is unchanged
(no-reset reg = 0 under `--x-initial 0`; FDRE INIT=0 on hardware — exactly the
old reset value); (2) no Block-A write can fire during the single t=0 `rst_n`
window because every write-enable traces exclusively to control registers that
are still async-reset-held (upstream `valid_*` chains, FSM `state`,
`sched_ready_in`, `accept`, `mac_valid_q*`, `skid_valid`, `wr_req`, pointers,
`emit_ready`, `do_write`, `load_skid`, `pix_out_ready`, …); (3) therefore the
full machine state at reset release is bit-identical and, by induction, every
later cycle and byte is identical — **byte-exact AND cycle-exact by
construction**. (4) Defense in depth: each class below is additionally
write-before-read per frame/pixel/OC-pass or only sampled under a reset-kept
valid bit.

## 2. Classes converted + FF table (elaborated configuration)

FF counts are computed from the live per-file parameters (C, MP, bus widths,
BEATS_PER_PIXEL) and, for the top helpers, from all 83 instantiation
parameter sets in `nn2rtl_top_engine.v`. Only the elaborated generate branch
is counted (e.g. one of g_legacy/g_bp per n4).

| # | Class (files) | Registers moved | FFs |
|---|---|---|---|
| C1a | 11 DW convs 812–872 (inline datapath) | `prod_q[9]`, `acc/biased/scaled[MP=16]`, `v_tmp`, `dp_data_out[C*8]`, skid `out_data[C*8]` | 61,590 |
| C1b | 3 DW convs 878/884/890 (C=576, NATIVE_TILED) | same pipes + `pix_out[4608]`, `tile_acc[4608]`, `out_lat[4608]` | 47,238 |
| C1c | 3 wide DW convs 896/902/908 (C=960, NATIVE_TILED) | same pipes + `out_pix[7680]`, `tile_acc[7680]`, `out_lat[7680]` | 74,886 |
| C2 | 22 single-beat n4 relus (n4, n4_2–n4_22) | `data_out_r` (g_legacy) / `out_data` (g_bp) — one branch elaborates | 40,448 |
| C3 | 13 multi-beat n4 relus (n4_23–n4_35) | `beat_buf[BEATS][256]` (ResNet P8 pattern) | 83,968 |
| C4 | 10 residual adds | skid `out_data`, `input_buf`, `dp_data_out` | 25,344 |
| C5 | node_conv_810 (stem wrapper) | skid `out_data[256]` (datapath/lbw are SHARED K1 files) | 256 |
| C6 | node_linear | skid `out_data[8000]` (in_buf2d/out_buf/dp_data_out ALREADY sync-only) | 8,000 |
| C7 | node_mean | `emit_data[10240]` (BRAM-critical acc/scaled/rounded block untouched) | 10,240 |
| C8 | output_serializer | `buf_data[8192]`, `data_out[256]` | 8,448 |
| C9 | nn2rtl_top_engine.v helpers | `skip_fifo.out_data_r` (15 insts, 10,624) + `engine_output_fifo.out_data` (2,048) + `stream_to_act_bram_bridge.{wr_data, skid_data[, beat_buf]}` (33 insts / 3 branches, 137,472) + `engine_output_bridge.{beat_buf, data_out | gather_buf}` (34 insts / 3 OUT_KINDs, 113,664) | 263,808 |
| C10 | rtl_library/retile_bridge.v `retile_gather` | `buf0/buf1` (2×FULL_W; 7 insts, 184 tiles) | 94,208 |
| | **TOTAL moved off rst_n** | **68 files** | **~718,434** |

(The task's prior sizing ~610–619K matches per class where the templates
matched — DW 178.6K≈183.7K, n4 122.5K≈124.4K, adds 25.8K≈25.3K, mean
10.7K≈10.2K, serializer 8.3K≈8.4K — and was LOW on the bridge classes
(engine_output_bridge 113.7K vs 81.8K, stream_to_act 137.5K vs 80.7K,
retile 94.2K vs 83.4K, exact instantiation-parameter sums) and HIGH on
node_linear (26.8K counted in_buf2d/dp_data_out which are ALREADY
sync-only; only the 8,000-bit output skid still carried a reset).)

Post-extension rst_n load = the surviving control FF (FSM/valid/ready/
counters/pointers across 100 modules + scheduler + engine control), the
~45–50K-FF class predicted by the prior analysis.

## 3. Per-class write-before-read proofs

### C1 depthwise convs (ResNet-K1-P2 analog, inline)
The DW datapath is an inline fork of `conv_datapath` (per-channel 9-tap dot
product). Identical argument to ResNet P2:
* `prod_q[9]`: rewritten EVERY cycle from `weight_q/tap_q` (both already
  no-reset; `tap_q` is 0 at power-on because line_buf_window's window regs are
  K1-recoded power-on-0 and unwritten during reset) → during reset Block A
  writes 0×w = 0 = old reset value. Consumed (`sum_comb`) only under
  `mac_valid_q2` (reset-kept).
* `acc[MP]`: sync-cleared on `ST_IDLE && start_mac` and on the `ST_OUTPUT`
  oc-advance BEFORE the first gated accumulate of every pass (`mac_valid_q1/q2`
  are reset-held 0 until the FSM runs). Clears placed LAST in Block A —
  NBA last-write-wins parity with the original single block.
* `biased`/`scaled`: written in `ST_BIAS`/`ST_SCALE`, read one state later —
  write-before-read within every OC pass; guards `state == ST_*` (reset-kept).
* staged output pixel (`dp_data_out`/`pix_out`/`out_pix`): every consumed byte
  is written during that pixel's OC passes before the (reset-kept)
  `dp_valid_out`/`pix_out_ready` pulse; `out_shift`/`out_round`/`v_tmp` are
  blocking temps referenced ONLY by Block A after the move (no cross-always
  shared-variable race; `i`/`lane_i`/`bias_oc`/`sc_oc`/`out_oc` likewise).
* Control kept async-reset: `state`, `lane_counter`, `oc_group`,
  `mac_valid_q1/q2`, `mac_lane_q*`, `mac_global_oc_q*` (lane-address gating,
  conservative — same call as ResNet's `mac_oc_group_q*`), `mac_done_issuing`,
  `dp_valid_out`/`pix_out_ready`, start/rearm regs.
* A-family skid `out_data`: written only under `dp_valid_out` (reset-kept),
  sampled only under `out_full` (reset-kept).
* B/C `tile_acc` (native 18/30-tile gather): every consumed slice is rewritten
  during the pixel's gather before the last-tile `core_valid_in` pulse; write
  gate `accept_tile = valid_in_t && sched_ready_in` (producer valid + scheduler
  ready, both reset-held). `in_tile` counter keeps reset.
* B/C `out_lat` (native drain): latched whole-pixel under `pix_out_ready`
  (1-cycle reset-kept pulse; `skid_block` guarantees `!out_busy` at that edge),
  consumed (`data_out_t`) only while `out_busy` (reset-kept).

### C2 single-beat n4 relus
`data_out_r` (g_legacy): written under `valid_in` (upstream valid chain,
reset-held at t=0), consumed downstream only under `valid_out_r` (reset-kept,
1-cycle echo). `out_data` (g_bp): written under `accept && valid_in`, consumed
only under `out_full`; both controls keep reset. The requant value
(`requant_comb`, a pure function of `data_in` + elaboration-time ROM) is
computed on the same admitted beat as before — only the register's reset arm
is dropped. Exactly one branch elaborates per instance (EB=1 ⇒ g_bp).

### C3 multi-beat n4 relus — byte-for-byte the ResNet P8 precedent
`beat_buf` is gather DATA, fully rewritten each pixel (beats 0..BPP-1) before
the `sending` phase reads it; the moved write replicates the original nested
guard `!sending && valid_in && ready_in` exactly (`ready_in` read pre-edge in
both forms; identical text in both generate branches, one elaborates). The
multi-site `data_out` emit regs are SKIPPED (interleaved with `valid_out`
control — same skip ResNet made for its 49 relus' data_out).

### C4 residual adds (ResNet-P9/P10 analog; 3 template shapes auto-detected)
* `input_buf`: fully rewritten on the accept edge (guard replicated verbatim:
  `state==ST_IDLE && valid_in && !skid_block`, or `…&& ready_in && !skid_block`
  for the 546/1038 shapes) strictly before the RUN pipe reads it (reads start
  the following cycle at ch_idx=0).
* `dp_data_out`: every consumed byte is written by the 3-stage pipe under
  `stage2_valid` (covers ch 0..OC-1) before `dp_valid_out` pulses; the
  single-line (`sat_out`/`sat_byte`/`sat_val`) and 3-arm saturate-cascade
  shapes are moved wholesale with the `state==ST_RUN/S_RUN && stage2_valid`
  guard replicated where a case statement provided the state term (the data
  conditions `out_pre >/< SAT_*` move with the writes — they are pure
  functions of the moved-along pipe).
* skid `out_data`: as C2/C5.
* KEPT async-reset (conservative, matches ResNet's add skip): `lhs_term`,
  `rhs_term`, `sum_term` MAC pipes (~100 FF/add), all `stage*_valid/idx`,
  `ch_idx`, `state`, `ready_in`, `dp_valid_out`.

### C5/C6 node_conv_810 + node_linear — output skid only
Same skid proof. 810's compute lives in the SHARED K1 files
(conv_datapath_mp_k + line_buf_window). node_linear's whole datapath
(in_buf2d, banked-ROM read pipe, acc_reg, out_buf, dp_data_out) was ALREADY in
a sync-only block (the SYNTH-FIT BANKED rewrite) — only the 8000b skid still
had a reset arm.

### C7 node_mean — `emit_data`
All 1280 bytes are written during ST_PACK (pack_idx 0..79, guard
`state == ST_PACK` reset-kept) BEFORE `emit_busy` rises on the last pack step;
`data_out` is sampled only under `valid_out = emit_busy` (reset-kept). The
moved loop goes into its OWN new sync-only block — the existing BRAM-critical
tiled-accumulate block (`acc_mem`/`scaled_mem`/`rounded_mem`, 1R/1W wide-word
pattern that fixed the 96GB synth OOM) is NOT touched, preserving its RAM
inference. `plane` becomes Block-A-only (no shared-loop-var race; `lane` stays
with the untouched block).

### C8 output_serializer — `buf_data` + `data_out`
`buf_data` is fully written on the accept edge (`!busy && valid_in`; `busy`
reset-kept) and read strictly afterwards (beats 1..NBEATS-1, `busy` phase).
`data_out` is sampled by m_axis only under `valid_out` (reset-kept). The two
write sites are mutually exclusive on `busy` and their guards are replicated
exactly (`valid_out && ready_in && beat != NBEATS-1` for the stream site).
`busy`/`beat`/`valid_out`/`last_out` keep reset. Block A sits inside the
existing `lint_off WIDTH` region.

### C9 top helpers (nn2rtl_top_engine.v)
* `skip_fifo.out_data_r` (BRAM-backed skid variant): written only under
  `load_skid` (pointer-derived; pointers reset-kept), sampled only under
  `out_valid_r` (reset-kept). `mem` write was already sync-only.
* `engine_output_fifo.out_data`: identical text + identical recode as ResNet
  K1 P6.
* `stream_to_act_bram_bridge`: g_w_eq is the verbatim ResNet P6 text/recode;
  the MBV2-specific g_w_lt (1-px/word zero-extension) and g_w_gt
  (cont_slice fixed-mux) branches get the same split: `wr_data` consumed only
  while `wr_req` pending, `skid_data` only while `skid_valid`, `beat_buf` only
  while `buf_active` — all three controls keep reset. g_w_gt textual order
  preserved: the drain `wr_data` write overrides the continue-slice write on a
  shared edge exactly as in the single block. `wr_addr`/`word_count`/
  `slice_idx` (address/count control) keep reset.
* `engine_output_bridge` (all 3 OUT_KINDs): `beat_buf` consumed
  (`current_tile`) only while `buf_valid`; `gather_buf`'s consumed bits are all
  rewritten during the position's BEATS_PER_POS-beat gather before `buf_full`
  rises; `data_out` sampled only under `valid_out`. `dispatch_count`,
  `tile_idx`, `beat_in_pos`, `pull_idx`, `tiles_emitted`, `drain_complete`
  (slot/position control) keep reset.

### C10 retile_gather — `buf0/buf1`
Ping-pong gather DATA: every consumed tile slice is rewritten during that
pixel's N_TILES-beat gather (g_idx walks 0..N_TILES-1 from reset) before
`full0/full1` (reset-kept) marks the buffer drainable; the emit side reads
`rbuf` only while `valid_out = rsel_full`. Writes gated by
`do_write = valid_in & wsel_empty` (producer valid reset-held; `wsel`/`full*`
reset-kept). `full0/full1/wsel/rsel/g_idx/e_idx` all keep reset.

## 4. Deliberately SKIPPED (and why)

| Item | FFs | Reason |
|---|---|---|
| DW B/C legacy+bp emitter branches (`lo_latch`/`lo_hold`, `em_buf`, `bp_hi`, legacy `data_out`) and legacy 2-beat input assemblers | 0 (elab.) | NOT elaborated — all six are `NATIVE_TILED(1)` in the shipped top; un-elaborated branches contribute no FF and the legacy `data_out` doubles as a port held-at-reset in native mode (left exactly as-is). |
| multi-beat n4 `data_out` emit regs | ~9.2K | Written at 2+ sites interleaved with `valid_out` control — the exact class ResNet K1 skipped for its 49 relus (drift-prone for ~1% extra). |
| add MAC pipes `lhs_term/rhs_term/sum_term` | ~1K | Tiny; heterogeneous across 3 add template generations (ResNet precedent: skipped). |
| `mac_lane_q*`, `mac_global_oc_q*` (DW), `wr_addr`/`word_count`/`slice_idx` (bridges), `tile_idx`/`beat_in_pos`/`pull_idx` (output bridges) | <2K | Lane-/address-select gating — treated as control (conservative; same call as ResNet's `mac_oc_group_q*`/`wr_addr`). |
| `retile_scatter.buf0/buf1` | 0 (elab.) | Instantiated only by the EXCLUDED all-spatial `nn2rtl_top.v`; also its write is a full-width read-modify-write (`wbuf_next` reads `wbuf_cur`) — weaker proof, zero payoff. |
| `uram_weight_bank`, `act_unified_mem`, `bias_mem`, `skip_fifo.mem`, `engine_output_fifo.mem`, node_linear datapath, node_mean acc/scaled/rounded | — | Already sync-only / ROM. |
| `coord_scheduler`, `nn2rtl_scheduler`, `address_generator`, `config_register_block`, engine FSM, `sched_started_r`, all valid/ready/state/counter/pointer bits | — | Control by definition. |
| `conv_datapath.v`, `conv_datapath_parallel.v`, `bram_to_stream_bridge.v` | — | Shipped but not instantiated by the MBV2 top. |
| 34 engine-dispatched pointwise `node_conv_*.v`, `nn2rtl_top.v`, `*.prek1m` | — | Not instantiated / excluded / backups. |

## 5. Verification performed (2026-06-10, worktree)

1. **Anchor audit**: `--dry-run` validates 68/68 files, every fixed anchor and
   every regex-derived anchor exactly-once (the multi-beat n4 class asserts
   exactly-two for its dual-branch anchors); abort-before-write on any drift.
2. **Applied**; immediate re-run reports `0 to patch, 68 already applied`
   (idempotency).
3. **Lint** (`verilator_bin --lint-only --top-module nn2rtl_top
   -DNN2RTL_ENGINE_SUBBLOCKS_PROVIDED --x-initial 0`, full collectSources set):
   **0 errors**; warning histogram = 1 DEFOVERRIDE + 7 TIMESCALEMOD, both
   pre-existing (command-line define override + mixed timescale), no new
   warning classes.
4. **E2E gate** (`bash scripts/run_mbv2_e2e_parallel.sh`, Verilator
   `--x-initial 0`, `--threads 1` per vector):
   **RESULT: PASS (8/8 byte-exact), TOTAL mismatch = 0** — and per-vector
   `e2e_cycles = 7,592,966`, IDENTICAL to the pre-K1 inherited-tree baseline
   (2026-06-10 01:12 run, all 8 vectors 7,592,966) → latency-neutral, as the
   construction predicts (no handshake/control bit touched).

Apply / rollback:
```
python scripts/apply_k1_mbv2_ext.py --dry-run
python scripts/apply_k1_mbv2_ext.py            # writes .prek1m backups
```
