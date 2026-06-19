# Engine skeleton — port specification

Authoritative interface contract for the Phase 2 shared compute engine
([deployment plan §6.1-6.3](../nn2rtl_u250_deployment_plan.md)). The
Wave 2 review gate enforces every sub-block table below via
`scripts/check_subblock_ports.py`. Drift the RTL from this document and
the gate rejects the sub-block.

## Conventions

- Direction is from the named sub-block's point of view. `input` = the
  sub-block samples it; `output` = the sub-block drives it.
- Width column accepts:
  - `1` for scalar (one wire),
  - a positive integer for an `[N-1:0]` vector,
  - `<a>*<b>` for a packed bus (e.g. `256*8` = 2048 bits).
- All signed values are `reg signed` / `wire signed` in the RTL; the
  signedness convention follows the seed in
  [output/rtl/node_conv_288.v](../../output/rtl/node_conv_288.v) and the
  protected pattern in
  [knowledge/patterns/protected/01_context.md](../../knowledge/patterns/protected/01_context.md).
- Reset is canonical: `rst_n`, active-low, asynchronous-assert,
  synchronous-deassert. Sub-blocks that write to `reg` arrays MUST split
  the array write into a sibling `always @(posedge clk)` block per
  [knowledge/patterns/protected/08_common_bugs.md §"Array memory write in an async-reset always block"](../../knowledge/patterns/protected/08_common_bugs.md).

## Engine top-level interface (drives Wave 2's wiring, not check_subblock_ports.py)

| Port | Direction | Width | Role |
| --- | --- | --- | --- |
| clk | input | 1 | Single clock. |
| rst_n | input | 1 | Async-assert, sync-deassert. |
| s_axil_awvalid | input | 1 | AXI4-Lite write-addr handshake. |
| s_axil_awready | output | 1 | AXI4-Lite write-addr handshake. |
| s_axil_awaddr | input | 8 | Config-register byte offset. |
| s_axil_wvalid | input | 1 | AXI4-Lite write-data handshake. |
| s_axil_wready | output | 1 | AXI4-Lite write-data handshake. |
| s_axil_wdata | input | 32 | Config-register write payload. |
| s_axil_wstrb | input | 4 | Byte-lane strobes for partial writes. |
| s_axil_bvalid | output | 1 | AXI4-Lite write-response valid. |
| s_axil_bready | input | 1 | AXI4-Lite write-response ready. |
| s_axil_bresp | output | 2 | OKAY / SLVERR / DECERR. |
| s_axil_arvalid | input | 1 | AXI4-Lite read-addr handshake. |
| s_axil_arready | output | 1 | AXI4-Lite read-addr handshake. |
| s_axil_araddr | input | 8 | Config-register byte offset. |
| s_axil_rvalid | output | 1 | AXI4-Lite read-data valid. |
| s_axil_rready | input | 1 | AXI4-Lite read-data ready. |
| s_axil_rdata | output | 32 | Config-register read payload. |
| s_axil_rresp | output | 2 | OKAY / SLVERR. |
| engine_start | input | 1 | Scheduler dispatches a layer by pulsing this with all config registers already written. |
| engine_busy | output | 1 | High from LOAD_CONFIG through DRAIN; low in IDLE and DONE. |
| engine_done | output | 1 | High in DONE state, sampled by scheduler before deasserting engine_start. |
| act_in_rd_addr | output | 16 | Engine's read address into the scheduler-owned activation BRAM. |
| act_in_rd_en | output | 1 | Read-enable for the activation BRAM read port. |
| act_in_rd_data | input | 2048 | One packed beat of 256 channels × 8 bits from the activation BRAM (registered 1 cycle after rd_en). |
| act_out_wr_addr | output | 16 | Engine's write address into the scheduler-owned output BRAM. |
| act_out_wr_en | output | 1 | Write-enable for the output BRAM. |
| act_out_wr_data | output | 2048 | One packed beat of 256 output channels × 8 bits. |
| weight_rd_addr | output | 22 | URAM word address (one 256-bit word per address). |
| weight_rd_en | output | 1 | URAM read enable. |
| weight_rd_data | input | 2048 | One URAM beat (256 INT8 weight bytes), used by mac_array. |
| bias_rd_addr | output | 22 | Bias-memory word address; one wide bias word per oc_pass (task 13a Bundle A / Fix 5). |
| bias_rd_en | output | 1 | Bias-memory read enable; pulses once per oc_pass at start of ST_REQUANT. |
| bias_rd_data | input | 8192 | One wide bias word = 256 × INT32, consumed by requant_pipeline as `bias_in`. |

## SUBBLOCK: mac_array

256 INT8×INT8 multiply-accumulate lanes, output-channel-parallel. All
lanes share the broadcast `act_byte`; each lane mults it against its
own slot of `weight_bus`. Accumulators clear on `mac_clear`, advance
one MAC step per `mac_valid_in` pulse.

| Port | Direction | Width | Owning sub-block |
| --- | --- | --- | --- |
| clk | input | 1 | mac_array |
| rst_n | input | 1 | mac_array |
| mac_clear | input | 1 | mac_array |
| mac_valid_in | input | 1 | mac_array |
| act_byte | input | 8 | mac_array |
| weight_bus | input | 2048 | mac_array |
| acc_out | output | 8192 | mac_array |
| mac_busy | output | 1 | mac_array |

## SUBBLOCK: requant_pipeline

Three-stage parallel-256 requantisation pipeline:

1. `biased[lane] = acc_in[lane] + bias_in[lane]`
2. `scaled[lane] = biased[lane] * SCALE_MULT_CONST`
3. `data_out[lane] = saturate_int8((scaled[lane] + sign_aware_round) >>> scale_shift)`

Sign-aware rounding bias MUST be the canonical
`scaled[MSB] ? (HALF-1) : HALF` form from
[01_context.md §"Scale-shift rounding — MANDATORY"](../../knowledge/patterns/protected/01_context.md).
`valid_out` fires exactly 3 cycles after `valid_in`.

| Port | Direction | Width | Owning sub-block |
| --- | --- | --- | --- |
| clk | input | 1 | requant_pipeline |
| rst_n | input | 1 | requant_pipeline |
| valid_in | input | 1 | requant_pipeline |
| acc_in | input | 8192 | requant_pipeline |
| bias_in | input | 8192 | requant_pipeline |
| scale_mult | input | 32 | requant_pipeline |
| scale_shift | input | 6 | requant_pipeline |
| valid_out | output | 1 | requant_pipeline |
| data_out | output | 2048 | requant_pipeline |

## SUBBLOCK: address_generator

Walks the layer's K_TOTAL = IC × KH × KW dimension during ST_RUN and
emits URAM weight/bias and BRAM activation read/write addresses.
Cf. counters in the seed [node_conv_288.v](../../output/rtl/node_conv_288.v)
(ar_pass_target, k_counter, lane_counter, in_pixel_counter) for the
shape of the address-walk loop, with weight stream now from URAM rather
than from a prefetched DRAM cache. Asserts `mac_done` when the current
OC pass has consumed all K_TOTAL weights; asserts `pixel_done` when
all OH×OW output pixels have been emitted.

| Port | Direction | Width | Owning sub-block |
| --- | --- | --- | --- |
| clk | input | 1 | address_generator |
| rst_n | input | 1 | address_generator |
| run_active | input | 1 | address_generator |
| cfg_ic | input | 12 | address_generator |
| cfg_oc | input | 12 | address_generator |
| cfg_kh | input | 3 | address_generator |
| cfg_kw | input | 3 | address_generator |
| cfg_ih | input | 8 | address_generator |
| cfg_iw | input | 8 | address_generator |
| cfg_oh | input | 8 | address_generator |
| cfg_ow | input | 8 | address_generator |
| cfg_stride_h | input | 3 | address_generator |
| cfg_stride_w | input | 3 | address_generator |
| cfg_pad_h | input | 3 | address_generator |
| cfg_pad_w | input | 3 | address_generator |
| cfg_weight_uram_base | input | 22 | address_generator |
| cfg_bias_uram_base | input | 22 | address_generator |
| cfg_act_in_bram_base | input | 16 | address_generator |
| cfg_act_out_bram_base | input | 16 | address_generator |
| oc_pass_idx | input | 3 | address_generator |
| pixel_h | input | 8 | address_generator |
| pixel_w | input | 8 | address_generator |
| weight_rd_addr | output | 22 | address_generator |
| weight_rd_en | output | 1 | address_generator |
| bias_rd_addr | output | 22 | address_generator |
| bias_rd_en | output | 1 | address_generator |
| act_in_rd_addr | output | 16 | address_generator |
| act_in_rd_en | output | 1 | address_generator |
| act_in_ic_byte_idx | output | 8 | address_generator |
| act_out_wr_addr | output | 16 | address_generator |
| k_index | output | 16 | address_generator |
| mac_done | output | 1 | address_generator |
| pixel_done | output | 1 | address_generator |

## SUBBLOCK: config_register_block

AXI4-Lite slave decoding per-layer configuration writes from the
scheduler and exposing them as the `cfg_*` wires consumed by
`address_generator` and `requant_pipeline`. Also samples the external
`engine_start` pin, produces the one-cycle `engine_start_pulse` that
kicks the FSM out of IDLE, and synthesises external `engine_busy` /
`engine_done` outputs from the FSM-internal `engine_busy_in` /
`engine_done_in` status wires.

Config-register map — **authoritative source: task 10's implementation in `output/rtl/engine/config_register_block.v`**. This table mirrors that file.

| Byte offset | Register | Fields | Driver (scheduler step) |
| --- | --- | --- | --- |
| 0x00 | INPUT_CHANNELS  | `{16'd0, cfg_ic[15:0]}` | step 0 |
| 0x04 | OUTPUT_CHANNELS | `{16'd0, cfg_oc[15:0]}` | step 1 |
| 0x08 | KERNEL_H_W      | `{24'd0, cfg_kh[3:0], cfg_kw[3:0]}` | step 2 |
| 0x0C | STRIDE_H_W      | `{26'd0, cfg_stride_h[2:0], cfg_stride_w[2:0]}` | step 3 |
| 0x10 | PADDING_H_W     | `{26'd0, cfg_pad_h[2:0], cfg_pad_w[2:0]}` | step 4 |
| 0x14 | INPUT_H_W       | `{7'd0, cfg_ih[8:0], 7'd0, cfg_iw[8:0]}` | step 5 |
| 0x18 | OUTPUT_H_W      | `{7'd0, cfg_oh[8:0], 7'd0, cfg_ow[8:0]}` | step 6 |
| 0x1C | WEIGHT_BASE_WORD | `{12'd0, cfg_weight_uram_base[19:0]}` | step 7 |
| 0x20 | BIAS_BASE_WORD  | `{16'd0, cfg_bias_uram_base[15:0]}` | step 8 |
| 0x24 | SCALE_MULT      | `cfg_scale_mult[31:0]` (full 32 bits — see task 13a fix 2) | step 9 |
| 0x28 | SCALE_SHIFT_AND_ZP | `{18'd0, cfg_zero_point[7:0], cfg_scale_shift[5:0]}` | step 10 |
| 0x2C | CONTROL         | bit 0 = START (write-1 pulses `engine_start`); bit 1 = BUSY (read-only mirror) | step 13 — **must be last** |
| 0x30 | STATUS          | bit 0 = DONE (read-only) | (read-only) |
| 0x34 | ACT_IN_BASE     | `{16'd0, cfg_act_in_bram_base[15:0]}` — added per task 04c | step 11 |
| 0x38 | ACT_OUT_BASE    | `{16'd0, cfg_act_out_bram_base[15:0]}` — added per task 04c | step 12 |

The scheduler issues these 14 writes per dispatch in the order shown (steps 0–12, then CONTROL.START as step 13). If the order or any offset is changed, both `output/rtl/engine/config_register_block.v` and `scripts/build_scheduler.py` must change together.

| Port | Direction | Width | Owning sub-block |
| --- | --- | --- | --- |
| clk | input | 1 | config_register_block |
| rst_n | input | 1 | config_register_block |
| s_axil_awvalid | input | 1 | config_register_block |
| s_axil_awready | output | 1 | config_register_block |
| s_axil_awaddr | input | 8 | config_register_block |
| s_axil_wvalid | input | 1 | config_register_block |
| s_axil_wready | output | 1 | config_register_block |
| s_axil_wdata | input | 32 | config_register_block |
| s_axil_wstrb | input | 4 | config_register_block |
| s_axil_bvalid | output | 1 | config_register_block |
| s_axil_bready | input | 1 | config_register_block |
| s_axil_bresp | output | 2 | config_register_block |
| s_axil_arvalid | input | 1 | config_register_block |
| s_axil_arready | output | 1 | config_register_block |
| s_axil_araddr | input | 8 | config_register_block |
| s_axil_rvalid | output | 1 | config_register_block |
| s_axil_rready | input | 1 | config_register_block |
| s_axil_rdata | output | 32 | config_register_block |
| s_axil_rresp | output | 2 | config_register_block |
| engine_start_ext | input | 1 | config_register_block |
| engine_busy_in | input | 1 | config_register_block |
| engine_done_in | input | 1 | config_register_block |
| engine_busy_ext | output | 1 | config_register_block |
| engine_done_ext | output | 1 | config_register_block |
| engine_start_pulse | output | 1 | config_register_block |
| cfg_ic | output | 12 | config_register_block |
| cfg_oc | output | 12 | config_register_block |
| cfg_kh | output | 3 | config_register_block |
| cfg_kw | output | 3 | config_register_block |
| cfg_ih | output | 8 | config_register_block |
| cfg_iw | output | 8 | config_register_block |
| cfg_oh | output | 8 | config_register_block |
| cfg_ow | output | 8 | config_register_block |
| cfg_stride_h | output | 3 | config_register_block |
| cfg_stride_w | output | 3 | config_register_block |
| cfg_pad_h | output | 3 | config_register_block |
| cfg_pad_w | output | 3 | config_register_block |
| cfg_scale_mult | output | 32 | config_register_block |
| cfg_scale_shift | output | 6 | config_register_block |
| cfg_weight_uram_base | output | 22 | config_register_block |
| cfg_bias_uram_base | output | 22 | config_register_block |
| cfg_act_in_bram_base | output | 16 | config_register_block |
| cfg_act_out_bram_base | output | 16 | config_register_block |

## SUBBLOCK: bram_to_stream_bridge

Two-half bridge between the engine's parallel BRAM ports and its
internal streaming MAC/requant pipeline.

- **Read half** — accepts the 2048-bit `act_in_rd_data` beats returned
  by the activation BRAM port (registered 1 cycle after the
  `address_generator` asserts `act_in_rd_en`) and selects the single
  signed-INT8 channel identified by `act_in_ic_byte_idx`. That byte is
  driven on `mac_act_byte` and pulsed `mac_act_byte_valid` so the
  mac_array advances one MAC step.
- **Write half** — receives the 256-byte packed `requant_data` bus
  from `requant_pipeline` and forwards it as a single 2048-bit
  `act_out_wr_data` beat with `act_out_wr_en` pulsed for one cycle.
  Holds `bridge_busy` high while a write is in flight so the FSM does
  not advance prematurely.

| Port | Direction | Width | Owning sub-block |
| --- | --- | --- | --- |
| clk | input | 1 | bram_to_stream_bridge |
| rst_n | input | 1 | bram_to_stream_bridge |
| act_in_rd_data | input | 2048 | bram_to_stream_bridge |
| act_in_rd_data_valid | input | 1 | bram_to_stream_bridge |
| act_in_ic_byte_idx | input | 8 | bram_to_stream_bridge |
| mac_act_byte | output | 8 | bram_to_stream_bridge |
| mac_act_byte_valid | output | 1 | bram_to_stream_bridge |
| requant_data | input | 2048 | bram_to_stream_bridge |
| requant_valid | input | 1 | bram_to_stream_bridge |
| out_ready | input | 1 | bram_to_stream_bridge |
| act_out_wr_data | output | 2048 | bram_to_stream_bridge |
| act_out_wr_en | output | 1 | bram_to_stream_bridge |
| bridge_busy | output | 1 | bram_to_stream_bridge |
