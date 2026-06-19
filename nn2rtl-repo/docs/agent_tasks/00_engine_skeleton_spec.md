---
task_id: 00
title: Engine skeleton specification
type: design document + author-written Verilog skeleton
status: review
depends_on: []
unblocks: [07, 08, 09, 10, 11]
---

# Task 00 — Engine skeleton specification

## Goal

Produce the human-designed top-level Verilog skeleton for the shared compute engine, plus a written port specification that the Wave 2 sub-block tasks consume.

The engine is the central artefact of Phase 2 of the deployment plan ([../nn2rtl_u250_deployment_plan.md §6.1-6.3](../nn2rtl_u250_deployment_plan.md)). It is a parameterised multi-shape compute datapath used to run the ~10 heaviest modules of ResNet-50 sequentially. Sub-blocks (MAC array, requantisation pipeline, address generator, config register block) are LLM-generated; the skeleton that wires them together is author-designed.

## Deliverables

1. **`output/rtl/shared_engine_skeleton.v`** — a Verilog file containing:
   - The top-level `module shared_engine (...)` declaration with the full port list.
   - `parameter` declarations for compile-time configuration (MAC count, bus widths, scale-shift bit widths).
   - Internal wire declarations connecting future sub-block stubs.
   - Empty `// SUBBLOCK: <name>` instantiation blocks that the Wave 2 tasks will fill in.
   - A simple FSM outline (state encoding + state transitions) but without the per-state logic that the sub-blocks own.
2. **`docs/agent_tasks/00_engine_skeleton_spec_PORTS.md`** — a port specification document containing for every signal in the top-level interface:
   - Direction (input / output / inout)
   - Width (in bits)
   - Role / semantics (one line)
   - Owning sub-block (which Wave 2 task is responsible for driving / sampling this signal)
3. **`docs/agent_tasks/00_engine_skeleton_spec_FSM.md`** — a state-machine document containing:
   - State list (IDLE, LOAD_CONFIG, RUN, REQUANT, DRAIN, DONE)
   - Transition conditions (what events cause each transition)
   - Which sub-block is active in each state

## Design commitments to honour

From the deployment plan §6.1:
- **MAC array shape**: 256 MACs (tentative, reviewable after first synthesis).
- **Parallelism axis**: output-channel-parallel (process 256 output channels at one spatial position simultaneously).
- **Requantisation pipeline depth**: 3 stages (bias-add → scale-multiply → scale-shift + saturate).
- **Weight port**: reads from on-chip UltraRAM. No AXI4-MM to DDR. URAM base address is a parameter loaded per dispatched layer.

From the deployment plan §3:
- Weights live in UltraRAM (pre-loaded at bitfile load via `.mem` files).
- Activations live in BRAM (ping-pong buffers managed by the scheduler).
- Engine has no external memory interface — it reads URAM, writes BRAM, exposes config via AXI4-Lite.

## Required public interface (must be present in the skeleton)

The top-level module's port list must include at minimum these groups:

- **Clock + reset**: `clk`, `rst_n`
- **AXI4-Lite control slave** (so the scheduler can write the per-layer config registers):
  - `s_axil_awvalid/awready/awaddr`, `s_axil_wvalid/wready/wdata/wstrb`, `s_axil_bvalid/bready/bresp`
  - `s_axil_arvalid/arready/araddr`, `s_axil_rvalid/rready/rdata/rresp`
- **Engine status**: `engine_busy`, `engine_done` (output handshake to the scheduler).
- **Engine start**: `engine_start` (input handshake from the scheduler).
- **BRAM activation input port** (the engine reads activations from a BRAM bank owned by the scheduler):
  - `act_in_rd_addr` (output), `act_in_rd_data` (input, packed width = `BUS_W`), `act_in_rd_en` (output)
- **BRAM activation output port** (the engine writes to a different BRAM bank):
  - `act_out_wr_addr` (output), `act_out_wr_data` (output, packed width = `BUS_W`), `act_out_wr_en` (output)
- **URAM weight read port**:
  - `weight_rd_addr` (output, width sized for the URAM region), `weight_rd_data` (input, packed width chosen by the address-generator sub-block)

Choose widths and packing rules that match the largest layer the engine needs to handle (`node_conv_298` at 512×512×3×3 = 2.36 MB of weights).

## Sub-block boundaries

The skeleton declares (but does not implement) these sub-block instantiations:

- `mac_array u_mac_array (...)` — task 07
- `requant_pipeline u_requant_pipeline (...)` — task 08
- `address_generator u_address_generator (...)` — task 09
- `config_register_block u_config_register_block (...)` — task 10

Each sub-block instantiation block in the skeleton must have:
- A clear `// SUBBLOCK: <name>` comment marker.
- The full port list of that sub-block (so the Wave 2 task knows exactly which signals to drive/sample).
- A short comment naming the task file responsible (e.g. `// see docs/agent_tasks/07_engine_mac_array.md`).

## Reference material

- The existing `output/rtl/node_conv_288.v` is the structural seed. It implements a single 1024-channel 1×1 conv with DRAM-backed weights. Read it before writing the skeleton. The skeleton is functionally a *generalised* version of this module, with weights in URAM rather than streamed from DDR.
- The deployment plan §6.1 lists tentative commitments for MAC count, parallelism axis, requantisation depth. Use those unless you have a documented reason to change them.
- The largest heavy module is `node_conv_298`. Its LayerIR entry (in `output/layer_ir.json`) tells you the maximum input/output channel counts and kernel sizes the engine must support.

## Verification

The skeleton itself does not simulate (it has empty sub-blocks). The verification at this stage is:

1. The skeleton compiles cleanly under `iverilog -t null shared_engine_skeleton.v` (no missing declarations, balanced begin/end).
2. The port spec in `00_engine_skeleton_spec_PORTS.md` covers every signal in the top-level module declaration.
3. The FSM document has one transition arc out of every state (no orphan states).
4. The skeleton uses the canonical port names from the protected pattern docs (clock = `clk`, reset = `rst_n`).

## Out of scope for this task

- Do NOT implement sub-block bodies. Those are Wave 2 tasks. Skeleton must only have the empty `// SUBBLOCK: <name>` instantiations.
- Do NOT touch the existing `output/rtl/*.v` files.
- Do NOT modify `contracts/`, `knowledge/`, or `sdk/`.
- Do NOT call any LLM agents from this task. The skeleton is a human-written Verilog file plus two human-written markdown documents.

## Success criteria

- Skeleton file exists at `output/rtl/shared_engine_skeleton.v`.
- Port-spec markdown exists at `docs/agent_tasks/00_engine_skeleton_spec_PORTS.md`.
- FSM markdown exists at `docs/agent_tasks/00_engine_skeleton_spec_FSM.md`.
- `iverilog -t null output/rtl/shared_engine_skeleton.v` compiles without error.
- The Wave 2 task agents can read the spec documents and produce sub-block RTL that drops into the skeleton's empty stubs without further design discussion.
