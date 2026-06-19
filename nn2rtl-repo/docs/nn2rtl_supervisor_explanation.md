# nn2rtl: End-to-End Explanation For A Supervisor

This document explains the current nn2rtl system in plain language.

The goal is simple. A reader should be able to explain to someone else how nn2rtl turns a neural network into hardware modules, how it checks them, how it repairs them, and what the current results show.

## Short Vocabulary

**What it is:** This section defines the few words that appear many times.

**Why it works this way:** The system crosses two worlds: neural networks and hardware. A small shared vocabulary avoids confusion.

**Current project details:**

- **Neural network:** A model made of layers. Each layer changes a block of numbers into another block of numbers.
- **Quantisation:** The step that turns floating point numbers into small integers. This project uses signed 8-bit integers, often written as INT8. An INT8 value fits in one byte and usually ranges from -128 to 127.
- **RTL:** Register Transfer Level. This is a hardware description that says what happens on each clock cycle.
- **Verilog:** The hardware language used for generated RTL files.
- **FPGA:** A reconfigurable chip. It can be programmed to act like custom hardware.
- **Vivado:** AMD/Xilinx hardware tool used here to check that generated Verilog can be synthesised for an FPGA.
- **Verilator:** A simulator used here to run the generated Verilog and compare its outputs against expected outputs.
- **LayerIR:** The project format for one network layer. IR means intermediate representation. It is the bridge between the neural network frontend and the hardware generator.
- **Golden vector:** A saved expected input or output. It is the answer that the generated hardware must match.
- **Contract:** A small rule book for how a hardware module must connect to the outside world.
- **Agent:** A named LLM role. It receives a task and returns structured output. The TypeScript orchestrator decides when to call each agent.

## System Overview

**What it is:** nn2rtl is a system that takes a quantised neural network and produces Verilog hardware modules for an FPGA.

**Why it works this way:** A neural network is too large and too detailed to ask an LLM to handle as one single prompt. nn2rtl breaks it into layer-sized jobs. Each job has a precise input, a precise expected output, and a deterministic check.

**Current project details:**

- The input is a quantised network checkpoint or an ONNX model.
- ResNet-50 uses the legacy checkpoint path at `output/`.
- MobileNetV2 uses the ONNX frontend at `output/mobilenet-v2/`.
- The shared registry is `networks.json`. It currently registers `resnet-50` and `mobilenet-v2`.
- The output is one Verilog module per runnable layer, plus reports, testbench data, and synthesis results.
- The target hardware is FPGA hardware.
- The main Vivado target in the current reports is `xczu9eg-ffvb1156-2-e`, the ZCU102 board class.
- The system does not yet build one complete FPGA image for the whole network. It builds and checks layer modules.
- The main proof is per-layer correctness and per-layer synthesis.

The high-level flow is:

1. Load or import the network.
2. Quantise it or read the already quantised model.
3. Write LayerIR and golden vectors.
4. Choose a hardware contract for each layer.
5. Ask an LLM agent to generate Verilog for one layer.
6. Run deterministic checks.
7. If it fails, classify and repair the failure.
8. If it passes, store the result and optionally store what was learned.

## Multi-Network Layout

**What it is:** nn2rtl now treats each network as a named project with its own output folder.

**Why it works this way:** ResNet-50 and MobileNetV2 should not overwrite each other's files. A result from one network should also not be silently used as if it came from another network.

**Current project details:**

- The shared registry is `networks.json`.
- The default network is `resnet-50`.
- ResNet-50 keeps the older layout and writes to `output/`.
- MobileNetV2 writes to `output/mobilenet-v2/`.
- SDK commands, MCP tools, scripts, and dashboard jobs read the same registry.
- The dashboard does not keep a separate network list.
- Network-local files include LayerIR, golden vectors, pipeline state, reports, RTL, testbenches, weights, goldens, debug files, and the failure corpus.
- The `import_network` command prepares a network before an expensive generation run.
- The import report says which layers look runnable, blocked, or risky.
- The current MobileNetV2 import report marks most layers as preflight risk, but the later pipeline run did pass 97 generated modules.
- The current MobileNetV2 throughput still skips `node_mean` and `node_linear`, the GAP and Gemm head.

## Tool Bridge And Dashboard

**What it is:** The MCP server and dashboard are the operating layer around the core pipeline.

**Why it works this way:** LLM agents should not run arbitrary hardware commands by guesswork. They use typed tools. Humans also need a way to inspect state without reading raw JSON every time.

**Current project details:**

- The MCP server lives under `mcp/`.
- MCP means Model Context Protocol. In this project it is the tool bridge between agents and local commands.
- The tools include writing Verilog, running Verilator, running Vivado, reading weights, fetching RTL patterns, and computing reference values.
- Inputs and outputs are checked with schemas. This makes tool calls easier to audit.
- The dashboard lives under `dashboard/`.
- It reads the selected network output root.
- It shows pipeline state, reports, coverage, and actions for the selected network.
- The dashboard selector uses the shared registry, so it should not drift away from SDK or script behaviour.

## Frontend: LayerIR And Goldens

**What it is:** The frontend turns a quantised model into LayerIR entries and golden vector files.

**Why it works this way:** The hardware generator should not guess what a neural layer means. It receives a complete layer description and exact expected data. This makes each hardware job small, testable, and repeatable.

**Current project details:**

- ResNet-50 uses `scripts/quantize_model.py` and `scripts/generate_golden.py` on the legacy PyTorch checkpoint path.
- MobileNetV2 uses `scripts/onnx_frontend.py` through `scripts/generate_golden.py`.
- The ONNX frontend currently supports `conv2d`, `relu`, `add`, `maxpool`, `global_avg_pool`, and `gemm`.
- It also folds simple tensor-only shape operators such as Flatten, Reshape, Squeeze, Unsqueeze, and Identity.
- MobileNetV2 uses ReLU6. This appears in LayerIR as `relu` with `clip_max`.
- MobileNetV2 depthwise convolutions are detected when `groups == input_channels == output_channels`.
- The project writes weights and biases as hex files under the selected output root.
- Golden vectors are stored as binary `.goldin` and `.goldout` files under the selected output root.
- These files use the `NN2V` format. It stores the number of vectors, samples per vector, and bytes per sample.
- LayerIR stores paths to these files. It does not inline huge tensors into JSON.

Generating goldens from the quantised model is the correct reference because that is the model being compiled.

The generated hardware is not asked to reproduce the original floating point model. It is asked to reproduce the quantised model. Quantisation changes weights, biases, scales, and sometimes output values. So the only correct reference for the hardware is the exact quantised computation after those changes.

This is correct in the strict engineering sense: for the chosen test vectors, the LayerIR, weights, scales, and goldens all come from the same quantised source. If the Verilog matches those goldens, it implements the compiled model for those vectors. It does not prove the model is accurate on all images. It proves the hardware agrees with the quantised model that nn2rtl compiled.

## What "Bit-Exact" Means

**What it is:** Bit-exact means the hardware output bits are identical to the expected output bits.

**Why it works this way:** Hardware works with fixed-width numbers. A difference of one bit can mean a different integer. So exact comparison is the clearest possible correctness test.

**Current project details:**

- The testbench records exact matches and mismatches for every output sample.
- Strict bit-exact means every output sample is equal and `mismatch_count` is zero.
- The current pass gate is slightly more permissive. It accepts a numerical pass when the largest absolute error is at most 3.
- Timing is stricter. The first `valid_out` must appear at the exact expected cycle.
- In the current ResNet result files, 108 passing modules are strictly exact and 9 passing modules are accepted by tolerance. Another 4 modules are recorded as pass in the pipeline state but do not have fresh per-module evidence on disk (2 have a stale `status:fail` results file from a superseded earlier attempt; 2 have no `.results.json` at all).
- In the current MobileNetV2 result files, 82 passing modules are strictly exact and 15 passing modules are accepted by tolerance.

So the phrase needs care. In a presentation, say:

**Strict bit-exact** means no output differs at all.  
**Current nn2rtl pass** means exact public timing plus maximum numerical error no larger than 3.

## Contracts

**What it is:** A contract is a rule book for the shape, ports, timing, and limits of a generated hardware module.

**Why it works this way:** Neural network layers have many shapes. A small layer can send all channels at once. A large layer may need a tiled stream or another memory plan. Contracts let one orchestrator handle several hardware styles without changing the main control flow.

**Current project details:**

The current contract set is:

| Contract | Main idea | Supported ops in metadata |
| --- | --- | --- |
| `flat-bus` | One packed pixel per cycle | conv2d, relu, add, maxpool, global_avg_pool, gemm |
| `tiled-streaming` | Send a fixed-width group of channels per beat | conv2d, relu, add, maxpool |
| `depthwise-conv` | Special case for MobileNet-style depthwise convolution | conv2d |
| `dram-backed-weights` | Read weights from external memory style ports | conv2d |
| `activation-double-buffering` | Use two activation buffers | conv2d, maxpool |
| `weight-tiling` | Split weight work into tiles | conv2d |

The base stream ports are the same across normal contracts: clock, reset, input valid, input ready, input data, output valid, and output data.

The contract also says:

- how wide the input and output buses may be,
- whether weights live on chip or are streamed,
- which extra ports are needed,
- which operations are legal,
- which testbench template to use.

The escalation path on failure is:

1. Try the selected contract.
2. If the module fails, run deterministic checks and classify the failure.
3. If it looks like a code bug, call Surgeon to repair it.
4. If retry attempts are exhausted, call Retrospector for advice.
5. If the contract itself is the likely problem, mark that contract variant as needing manual correction.
6. Try the next suitable contract when one exists.
7. Keep other layers moving rather than stopping the whole network.

## Main Pipeline

**What it is:** The main pipeline is the layer-by-layer loop that turns LayerIR into checked Verilog.

**Why it works this way:** Each layer can be generated and checked alone. This makes failures easier to find. It also means a later layer does not hide an earlier bug.

**Current project details:**

A single layer goes through this lifecycle:

1. The orchestrator reads one LayerIR entry.
2. It chooses the current contract for that layer.
3. It gathers relevant pattern documents and references.
4. It calls Foundry to write a Verilog module.
5. It runs a structural preflight check before simulation.
6. It runs Icarus Verilog for a fast compile-style check.
7. It runs Verilator with the static testbench.
8. The testbench compares hardware output to the golden output.
9. If simulation passes, Vivado synthesis runs.
10. If Vivado timing passes, the module is marked as passed.
11. If any step fails, the failure is classified and repair begins.

The orchestrator is deterministic TypeScript. This matters. The LLM does not decide the pipeline state. The code does. The LLM only receives bounded tasks, such as "generate this module" or "repair this failing module".

The main state files are:

- `pipeline_state.json`
- `reports/pipeline_summary.json`
- `reports/run_log.jsonl`
- per-module `.results.json`
- per-module `.vivado.json`

For ResNet-50 these files live under `output/`.  
For MobileNetV2 they live under `output/mobilenet-v2/`.

## Agents

**What it is:** Agents are named LLM jobs with narrow responsibilities.

**Why it works this way:** One broad prompt would be hard to test. Separate roles make the system easier to control, cheaper to debug, and easier to explain.

**Current project details:**

The orchestrator is deterministic TypeScript. It calls stateless LLM turns through the Claude Agent SDK. The LLM calls do not own the state machine.

| Agent or call | Job | Reads | Produces |
| --- | --- | --- | --- |
| Cartographer | Extract the model into LayerIR on the legacy path | Quantised checkpoint and frontend tools | PipelineIR and paths to weights and goldens |
| Foundry | Generate first Verilog for one layer | One LayerIR, contract metadata, pattern docs, references, failure context if any | A Verilog module on disk and metadata |
| Surgeon | Repair a failing Verilog module | Broken Verilog, LayerIR, verification result, prior attempts, pattern docs | A repaired Verilog module |
| Failure Classifier | Decide what kind of failure happened | Verifier result, logs, LayerIR, contract summary | A category such as code bug or architectural fit |
| Retrospector | Analyse repeated failure and suggest the next move | Attempt history, failed RTL versions, docs used, contract state | Advice for one final repair or generation attempt |
| Improve Foundry | Rewrite an already passing module to improve one metric | Passing RTL, LayerIR summary, baseline metrics, target rule | Improved RTL variant if it still passes |

Cartographer, Foundry, Surgeon, and Improve Foundry are plugin agent prompts. Failure Classifier and Retrospector are LLM calls driven directly by the orchestrator.

## Testbench

**What it is:** The testbench is a fixed C++ Verilator program that drives one generated Verilog module and checks its output.

**Why it works this way:** The generated hardware is allowed to change. The checker should not change with it. A static testbench avoids the risk that an LLM creates both a wrong design and a wrong test.

**Current project details:**

- The base testbench is `tb/static_verilator_tb.cpp`.
- Each contract has a small template under `contracts/<contract>/testbench.cpp`.
- The orchestrator writes a sidecar JSON file for each module.
- The sidecar tells the testbench the module name, port widths, golden input path, golden output path, and expected latency.
- The testbench drives `valid_in` and waits for `ready_in`.
- It samples `data_out` when `valid_out` is high.
- It checks the first output cycle against `pipeline_latency_cycles`.
- It writes a structured result JSON.

Per-module isolation matters because it gives a clean question:

"Does this one hardware module produce the expected outputs for this one layer?"

That is much easier to debug than asking why a full network output is wrong after many layers.

## Storing Wrong RTL

**What it is:** nn2rtl keeps failed Verilog files and failure records on purpose.

**Why it works this way:** A failed design is useful evidence. It shows what the LLM tried, what went wrong, and which layer shape caused the issue. That evidence helps later repair attempts and future generations.

**Current project details:**

- Failed attempts are stored in the output tree.
- The failure corpus lives under each network output root.
- For ResNet-50 it is under `output/failure_corpus/`.
- For MobileNetV2 it is under `output/mobilenet-v2/failure_corpus/`.
- Visible failure records are indexed in `failure_corpus/visible/index.jsonl`.
- Old or moved failures go into `failure_corpus/archive/`.
- Failure records include network id, model name, module id, signature data, applicability data, and contraindications.

The failure corpus feeds future work in two ways:

1. Surgeon can see what failed before and avoid repeating the same repair.
2. Foundry can be warned about similar layer shapes before it generates new RTL.

This turns failure into training evidence for the local system. It does not retrain the LLM. It improves the prompts and local memory used around the LLM.

## File Lifecycle For References And Improvement

**What it is:** nn2rtl stores reusable pattern documents and reference Verilog in controlled tiers.

**Why it works this way:** A useful generated pattern should not become trusted forever after one pass. It first needs a probation period. Bad knowledge must be easy to archive.

**Current project details:**

The knowledge tree has these main areas:

- `knowledge/patterns/protected/`
- `knowledge/patterns/probationary/`
- `knowledge/patterns/active/`
- `knowledge/patterns/archive/`
- `knowledge/patterns/improved/`
- matching reference folders under `knowledge/references/`

The lifecycle is:

1. **Protected:** Hand-written, trusted starting material. The pipeline does not edit it.
2. **Probationary:** New generated knowledge. It passed at least once, but it is not fully trusted yet.
3. **Active:** Generated knowledge that has enough successful use to be preferred.
4. **Archive:** Knowledge moved out of use after failure or retirement.
5. **Improved:** Passing variants made by the improve command.

Current document counts:

| Type | Count | Notes |
| --- | ---: | --- |
| Hand-written protected pattern docs | 12 | `01_context.md` to `12_depthwise_conv.md` |
| Protected reference Verilog files | 4 | Conv references from ResNet work |
| Generated lifecycle docs | 13 | 12 probationary, 1 active |
| Generated docs inferred from ResNet-50 | 5 | Includes tiled and improved conv work |
| Generated docs inferred from MobileNetV2 | 8 | All depthwise-conv probationary docs |
| Improved pattern docs | 3 | One throughput variant and two node_conv_248 variants |

ResNet-50 came first. Of the 12 hand-written protected pattern docs, 9 were already in place when MobileNetV2 began (shared context, conv1x1, conv3x3, conv7x7, add, ReLU, maxpool, common bugs, DRAM-backed weights). All 4 hand-written protected reference Verilog files were also from the ResNet era. MobileNetV2 added 3 new protected prose docs for new operations (`10_global_avg_pool.md`, `11_gemm.md`, `12_depthwise_conv.md`) and 0 new protected reference Verilog files. So roughly three quarters of the protected starting knowledge MobileNetV2 used was inherited from ResNet, and the new MobileNet-specific work was prose-only.

## Improve Command

**What it is:** The improve command takes a module that already works and asks for a better version along one optimisation target.

**Why it works this way:** The main pipeline asks, "make it work." The improve command asks, "keep it working, but make it better in this specific way." These are different tasks.

**Current project details:**

The current optimisation targets are:

- `use-dsp`: use DSP blocks for real multiplication work.
- `use-bram`: move useful memory into block RAM.
- `reduce-lut`: reduce LUT use.
- `reduce-ff`: reduce flip-flop use.
- `improve-fmax`: raise the maximum clock frequency.
- `reduce-latency`: reduce latency when the flow allows it.
- `increase-throughput`: increase frames per second, usually by doing more work in parallel.

The improve flow is stricter than normal generation:

1. It starts from a passing module.
2. Improve Foundry makes one attempted rewrite.
3. Verilator checks the same golden vectors.
4. Vivado checks synthesis and timing.
5. A deterministic target checker decides if the requested metric truly improved.
6. If the attempt fails, a later attempt gets the failure evidence.
7. On the third attempt, Retrospector advice may be added.
8. A successful result can replace the canonical RTL or be kept as a variant.

For multiple targets, the current logic prefers a sequence. It tries one target at a time. After each accepted step, the next step must preserve the earlier wins.

## Self-Improving Mechanics

**What it is:** Self-improvement means the system stores useful lessons from runs and uses them later.

**Why it works this way:** LLM calls are stateless. They do not remember prior project failures unless the system gives that evidence back to them. nn2rtl builds its own local memory through reports, failure records, and reference files.

**Current project details:**

There are two kinds of improvement.

**In-run improvement** happens during one layer attempt.

- Foundry creates RTL.
- The testbench finds a failure.
- Failure Classifier labels the failure.
- Surgeon repairs the module.
- Retrospector may advise after repeated failure.
- The same layer gets another chance.

**Cross-run improvement** happens after a run.

- Passing modules can create probationary docs and references.
- Failed modules add to the failure corpus.
- Future Foundry and Surgeon calls can retrieve relevant docs, references, and failure lessons.
- Generated knowledge has applicability metadata. It should only be reused where it fits.

This is not automatic magic. It is structured reuse of local evidence.

## Signatures And Retrieval

**What it is:** A signature is a content-based summary of a layer after its contract has been chosen.

**Why it works this way:** A lesson from one layer should only be reused on a layer that is truly similar. Signatures stop the system from treating every convolution as the same problem.

**Current project details:**

- The project stores a base layer signature for diagnostics.
- It stores a runtime layer signature and a `signature_hash` for retrieval.
- The exact reference key includes operation type, contract id, kernel, stride, dilation, groups, bus width, channel tile, channel counts, and quantisation family.
- Padding is stored and used for ranking.
- `mixed_or_unknown` quantisation is not allowed to make an exact reference match.
- Retrieval starts strict and then relaxes:
  - exact signature hash,
  - exact reference key,
  - operation plus contract plus kernel and shape details,
  - operation plus contract plus kernel,
  - operation only.
- Contraindications can veto a candidate at every level.

This matters for MobileNetV2. Depthwise convolution looks like convolution, but it has no cross-channel reduction. A normal convolution reference must not be reused as if it were depthwise-safe.

## Results: Main Pipeline

**What it is:** This section summarises the current measured project outputs.

**Why it works this way:** The supervisor needs totals and averages, but also needs to know which numbers are complete and which are partial.

**Current project details:**

### ResNet-50

| Metric | Current value |
| --- | ---: |
| LayerIR layers | 119 |
| Ops | 53 conv2d, 49 relu, 16 add, 1 maxpool |
| Pipeline state | 119 pass, 0 fail_abort |
| Result files (fresh) | 117 pass with fresh evidence (108 strict + 9 tolerance); 2 stale `status:fail` from earlier attempts; 2 missing |
| Throughput measured | 117 / 119 layers |
| Throughput skipped | `node_relu_24`, `node_conv_252` (no fresh `.results.json`) |
| Bottleneck | `node_conv_196`, the 7x7 stem conv |
| Steady-state throughput | 1.5702 fps |
| One-image end-to-end latency estimate | 0.7196 s |
| Total LLM cost | $170.61 |
| Average cost per passing module | $1.43 |
| Average attempts per state entry | 1.09 |

Cost split:

| Model family | Cost |
| --- | ---: |
| Opus | $163.57 |
| Sonnet | $3.49 |
| Haiku | $3.55 |

For ResNet-50, the bottleneck is the first large spatial convolution. The smaller ReLU and add modules are much faster and do not set network throughput.

### MobileNetV2

| Metric | Current value |
| --- | ---: |
| LayerIR layers | 99 |
| Ops | 52 conv2d, 35 relu, 10 add, 1 global_avg_pool, 1 gemm |
| ReLU6 layers | 35 |
| Depthwise conv layers | 17 |
| Pipeline state | 97 pass (GAP and Gemm head excluded from this run) |
| Result files (fresh) | 97 pass with fresh evidence (82 strict + 15 tolerance); 0 fail |
| Throughput measured | 97 / 99 layers |
| Throughput skipped | `node_mean`, `node_linear` (GAP and Gemm head, deliberately deferred) |
| Bottleneck | `node_conv_912` |
| Steady-state throughput | 10.1424 fps |
| One-image end-to-end latency estimate | 0.1079 s |
| Total LLM cost | $196.39 |
| Average cost per passing module | $2.02 |
| Average attempts per state entry | 1.30 |

Cost split:

| Model family | Cost |
| --- | ---: |
| Opus | $186.75 |
| Sonnet | $3.65 |
| Haiku | $5.99 |

MobileNetV2 has no FINN or hls4ml comparison in the current project. Its reported throughput is nn2rtl-only. It also skips the final GAP and Gemm head in the throughput roll-up, so the current MobileNetV2 result is best described as "97 of 99 hardware layers measured".

### Whole-Network Area And Frequency Summary

**What it is:** The aggregated post-synthesis numbers across every passing module of each network.

**Why it works this way:** Per-layer throughput is the most important headline number, but a supervisor will also want to know how much hardware the system is producing in total, what an average module looks like, and where the outliers sit. Mean and median diverge for several metrics because a small number of large modules dominate the means.

**Method:** Each row is computed across all passing modules of the network (119 for ResNet-50, 97 for MobileNetV2). BRAM18-equivalent is computed as `bram18_count + 2 x bram36_count` because each BRAM36 contains two BRAM18 primitives. Fmax is only counted from modules whose Vivado run actually reported a setup-timing slack; modules where the Vivado report came back with `fmax = 0` (a tool-side quirk on some synthesis runs) are excluded from the Fmax row but are still counted in the area rows.

| Metric | ResNet-50 (119 modules) | MobileNetV2 (97 modules) |
| --- | ---: | ---: |
| LUT, sum | 2,917,729 | 2,042,651 |
| LUT, mean per module | 24,519 | 21,058 |
| LUT, median per module | 5,687 | 6,088 |
| LUT, max single module | 188,568 (`node_conv_296`) | 336,522 (`node_conv_818`, depthwise stem) |
| FF, sum | 1,062,205 | 366,722 |
| FF, mean per module | 8,926 | 3,781 |
| FF, median per module | 5,667 | 3,073 |
| FF, max single module | 49,509 | 13,683 |
| DSP, sum | 219 | 11,397 |
| DSP, mean per module | 1.84 | 117.5 |
| DSP, median per module | 1 | 2 |
| DSP, max single module | 25 (`node_conv_256`) | 768 (`n4_15`, a 384-channel ReLU6) |
| BRAM18-equivalent, sum | 2,279 | 13,863 |
| BRAM18-equivalent, mean per module | 19.2 | 142.9 |
| BRAM18-equivalent, median per module | 0 | 0 |
| BRAM18-equivalent, max single module | 171 | 3,072 |
| Fmax MHz, modules with timing reported | 79 / 119 | 75 / 97 |
| Fmax MHz, mean | 327.7 | 264.1 |
| Fmax MHz, median | 313.8 | 268.9 |
| Fmax MHz, max | 621.9 | 420.5 |
| Fmax MHz, min | 166.9 | 135.6 |

**What this shows:**

- The **median module is small** on both networks (5,700-6,100 LUTs, 3,000-5,700 FFs, 1-2 DSPs, 0 BRAMs). This is the "typical" nn2rtl module shape: a compact, DSP-packed datapath with weights and biases in BRAM and a short pipeline.
- The **means are much larger** because a handful of layers carry most of the area. ResNet-50's largest module (`node_conv_296`) alone uses 188k LUTs, and MobileNetV2's largest (`node_conv_818`) uses 336k LUTs. These are outliers — almost all other modules are an order of magnitude smaller.
- MobileNetV2 uses **roughly 50x more DSPs in total** than ResNet-50 (11,397 vs 219). This is because the depthwise contract and some ReLU6 implementations pack many parallel multipliers; the requantising ReLU6 on a 384-channel feature map (`n4_15`) consumes 768 DSPs by itself.
- Median Fmax is comfortably above 250 MHz on both networks (314 MHz ResNet, 269 MHz MobileNet), suggesting the typical module is timing-comfortable on the ZCU102.

**Honest caveat — these are per-module numbers, not a whole-FPGA budget.**

Adding up modules is not the same as saying "these all fit on one chip at the same time." The ZCU102 (xczu9eg) has approximately 274k LUTs, 548k FFs, 2,520 DSPs, and 1,824 BRAM36 (3,648 BRAM18) total. By those budgets:

- ResNet-50's per-module sum exceeds the chip in LUT (~11x), FF (~2x), and BRAM18-equivalent (negligible budget impact at 2,279 / 3,648). DSP fits comfortably.
- MobileNetV2's per-module sum exceeds the chip in LUT (~7x), DSP (~4.5x over budget), and BRAM18-equivalent (~3.8x).

This is **expected** given that nn2rtl currently produces and verifies modules independently. A real deployment would either (a) time-multiplex layers onto a smaller shared accelerator, or (b) build the largest few layers, accept that they own the chip, and skip a full-network image. The presentation should frame these aggregate numbers as "what nn2rtl has generated and verified," not "what is currently flying on one ZCU102."

## Results: Improve Experiments

**What it is:** These experiments test whether the system can improve already passing RTL.

**Why it works this way:** Correct first-pass RTL is only one goal. A hardware design also needs resource and speed trade-offs. The improve command tests whether LLMs can make controlled quality changes without breaking correctness.

**Current project details:**

The key methodological finding is that one large combined request is weaker than a controlled sequence.

The clearest example is `node_conv_248`:

- A simultaneous `use-dsp,use-bram` request failed after 3 attempts.
- A single `use-bram` variant succeeded.
- A single `use-dsp` variant succeeded.
- The successful variants were stored under `knowledge/patterns/improved/` and `knowledge/references/improved/`.

This led to the current multi-target design:

1. Try one target.
2. Check it with Verilator, Vivado, and the deterministic target rule.
3. If it passes, keep that as the new baseline for the next target.
4. If a later step breaks an earlier win, reject it.

This is a methodological result. It says that controlled step-by-step optimisation is easier to verify than asking one LLM call to optimise everything at once.

Current improve report examples include successful `use-bram`, `reduce-lut`, `reduce-ff`, and kept-as-variant `use-dsp` results. Some `increase-throughput` and `use-bram` attempts still fail, which is expected because these changes often require deeper architecture changes.

### Headline Improve Results

**What it is:** The strongest before-and-after numbers from successful improve runs. Each row is a real, deterministically verified rewrite. Verilator still passes against the same golden vectors, and Vivado still meets timing on the same FPGA target. Only the named metric was the optimisation goal; the others are reported for context.

**Why it works this way:** These numbers show that the improve loop can make large area changes on already-correct hardware without breaking it. The improvements are not toy reductions. Some are full order-of-magnitude swings.

| Module | Target | Baseline | After | Change | Effect |
| --- | --- | ---: | ---: | ---: | --- |
| `node_conv_298` | reduce-ff | 267,918 FFs | 6,254 FFs | **-97.7%** | 43x fewer flip-flops; massive register pressure removed |
| `node_conv_292` | reduce-ff | 267,737 FFs | 10,357 FFs | **-96.1%** | 26x fewer flip-flops |
| `node_conv_298` | reduce-lut | 999,999 LUTs (overflow / unsynthesisable) | 89,499 LUTs | **-91.1%** | Rescued an unsynthesisable baseline into a real, fitting design |
| `node_conv_284` | reduce-lut | 227,703 LUTs | 64,530 LUTs | **-71.7%** | 3.5x fewer LUTs |
| `node_conv_248` | use-dsp | 1 DSP, multiplies in LUTs | 25 DSPs, multiplies in DSP48E2 slices | +24 DSPs | LUTs freed; multiplier datapath moved into hardened DSP blocks |
| `node_conv_282` | use-bram | 0 BRAM18, all storage in LUTs | 8 BRAM18 | +8 BRAM18 | Weights and activations moved off the LUT fabric |
| `node_conv_284` | use-bram | 0 BRAM18 | 26 BRAM18 | +26 BRAM18 | Largest BRAM lift; aligns with the parallel reduce-lut win on the same module |
| `node_conv_290` | use-bram | 0 BRAM18 | 7 BRAM18 | +7 BRAM18 | Moved storage off LUTs |
| `node_conv_292` | use-bram | 0 BRAM18 | 15 BRAM18 | +15 BRAM18 | Companion change to the FF reduction win on the same module |
| `node_conv_298` | use-bram | 7 BRAM18 | 22 BRAM18 | +15 BRAM18 | Built on the LUT reduction; more weight memory pushed into BRAM |

**What this shows in one sentence:** when the improve loop succeeds, the changes are not cosmetic. A 97.7 percent flip-flop reduction and a 91.1 percent LUT reduction on the same family of modules are real engineering wins that human designers would normally produce after careful manual restructuring.

**Caveats:**

- The "999,999 LUTs" baseline for `node_conv_298 reduce-lut` is the project sentinel for "Vivado could not place this design at all." The rewrite turned an unsynthesisable variant into a placed, timing-passing module. This is impressive but should be framed as a recovery rather than a typical 10x compression.
- All rewrites were verified with the same Verilator goldens and the same Vivado post-synth flow as the original.
- The improve run is per-module. The numbers above are for individual layers and do not directly multiply into a whole-network reduction.

## FINN And hls4ml Comparison

**What it is:** This section compares nn2rtl against two existing neural-network-to-hardware tools where current data exists.

**Why it works this way:** A thesis needs a baseline. The comparison is not perfect, because the tools expect different model formats and generate different styles of hardware.

**Current project details:**

### FINN

FINN is a specialised FPGA tool for quantised neural networks.

Current FINN files are under `comparison/results/finn/` and `comparison/tier_a/compare_three_way.csv`.

The current aggregate PPA file `comparison/results/finn_ppa.csv` has throughput columns, but in the checked file it only has two data rows and both rows have `verification_status: failed`. The wider three-way CSV has FINN estimated fps for 11 convolution rows.

For those 11 FINN rows:

- Bottleneck estimated FINN throughput is 7.827 fps.
- Median estimated FINN throughput is 129.85 fps.
- The same stem layer is the bottleneck in the ResNet comparison.

Compared with nn2rtl ResNet-50:

| Metric | nn2rtl ResNet-50 | FINN comparison rows |
| --- | ---: | ---: |
| Bottleneck throughput | 1.5702 fps | 7.827 fps |
| Main bottleneck | stem conv | stem conv |
| FINN conv rows with fps | not applicable | 11 |

The trade-off is clear. FINN is faster in the measured bottleneck because it uses more parallel multiply work. nn2rtl is more conservative. It uses a more serial design in many places. That lowers throughput but can use fewer resources.

The comparison also found structural limits in the FINN path:

- Many FINN runs were not clean verification passes.
- Some failed because the expected batch shape and actual batch shape did not match.
- Some add and ReLU runs had verification disabled or missing required FINN output artefacts.
- FINN expects a tool-friendly quantised ONNX style. The project uses INT8 post-training quantisation, which does not always map cleanly into FINN's assumptions.

This is itself a finding: existing tools can be strong when the model format fits them, but hard to use as a drop-in comparator for this exact quantised flow.

### hls4ml

hls4ml is a high-level synthesis flow. It starts from higher-level model descriptions and uses HLS to generate hardware.

Current hls4ml files are under `comparison/tier_a/hls4ml_out/`.

The project computed throughput by reading each HLS synthesis report's interval cycles and joining that with the post-synthesis Fmax in `compare_three_way.csv`.

Current hls4ml throughput summary:

| Metric | Current value |
| --- | ---: |
| Layers measured | 12 |
| Bottleneck layer | `layer0_0_conv1` |
| Bottleneck throughput | 189.95 fps |
| Median measured layer throughput | 6192.80 fps |

hls4ml is much faster on this small measured subset. It is not a full MobileNet comparison, and it is not a full ResNet-50 whole-network comparison in the current files. It is a layer subset comparison.

### Three-Way Area And Performance Comparison

**What it is:** A like-for-like layer-by-layer comparison of nn2rtl, hls4ml, and FINN on the layers where all three tools produced a build.

**Why it works this way:** Aggregate ResNet-50 numbers are not comparable because hls4ml and FINN did not build the whole network. The only fair comparison is on layers where all three tools have a built module with Vivado-measured area and timing.

**Current project details:**

The source data is `comparison/tier_a/compare_three_way.csv`, joined with per-tool throughput files. Eight layers have area and Fmax data in all three tools. Six of those eight also have a throughput-per-frame measurement in all three tools.

**Per-layer detail (six fully-comparable layers):**

| Layer | Tool | LUT | FF | DSP | BRAM18 | Fmax (MHz) | fps |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| layer0_0_conv1 | nn2rtl | 3618 | 1922 | 1 | 7 | 187.30 | 1.57 |
| layer0_0_conv1 | hls4ml | 75262 | 20311 | 0 | 0 | 70.34 | 189.95 |
| layer0_0_conv1 | FINN | 46080 | 30790 | 0 | 0 | 115.46 | 7.83 |
| layer1_0_conv1 | nn2rtl | 1937 | 1468 | 1 | 0 | 365.76 | 27.82 |
| layer1_0_conv1 | hls4ml | 4969 | 2968 | 0 | 0 | 69.01 | 1833.81 |
| layer1_0_conv1 | FINN | 46625 | 32767 | 0 | 1 | 111.38 | 554.96 |
| layer1_0_conv3 | nn2rtl | 4002 | 3082 | 1 | 0 | 365.76 | 6.96 |
| layer1_0_conv3 | hls4ml | 45037 | 16409 | 0 | 0 | 71.03 | 1887.49 |
| layer1_0_conv3 | FINN | 54301 | 33232 | 0 | 1 | 107.20 | 133.54 |
| layer1_0_downsample | nn2rtl | 4099 | 3086 | 1 | 0 | 365.76 | 6.96 |
| layer1_0_downsample | hls4ml | 16940 | 7791 | 1 | 0 | 73.87 | 1962.96 |
| layer1_0_downsample | FINN | 51866 | 31846 | 0 | 1 | 104.24 | 129.85 |
| layer1_1_conv1 | nn2rtl | 3109 | 3101 | 1 | 0 | 350.63 | 6.78 |
| layer1_1_conv1 | hls4ml | 23001 | 11979 | 0 | 0 | 73.87 | 1177.77 |
| layer1_1_conv1 | FINN | 45370 | 29188 | 0 | 1 | 112.87 | 140.59 |
| layer1_1_conv3 | nn2rtl | 5687 | 3119 | 1 | 0 | 350.63 | 6.91 |
| layer1_1_conv3 | hls4ml | 36705 | 18086 | 0 | 0 | 68.51 | 1391.70 |
| layer1_1_conv3 | FINN | 52731 | 33196 | 0 | 1 | 103.55 | 128.99 |

**Averages across the eight area-comparable layers:**

| Metric | nn2rtl | hls4ml | FINN |
| --- | ---: | ---: | ---: |
| LUT (mean) | 3,992 | 32,174 | 49,079 |
| LUT (median) | 4,002 | 24,063 | 50,771 |
| FF (mean) | 2,748 | 11,956 | 31,288 |
| FF (median) | 3,099 | 11,979 | 31,250 |
| DSP (mean) | 1.0 | 0.13 | 0 |
| BRAM18 (mean) | 0.88 | 0 | 0.88 |
| Fmax MHz (mean) | 339 | 71 | 109 |
| Fmax MHz (median) | 365 | 70 | 111 |

**Throughput across the six fps-comparable layers:**

| Metric | nn2rtl | hls4ml | FINN |
| --- | ---: | ---: | ---: |
| fps (mean) | 9.50 | 1,407 | 183 |
| fps (median) | 6.96 | 1,834 | 134 |
| fps (min, the bottleneck stem layer) | 1.57 | 189.95 | 7.83 |
| fps (max) | 27.82 | 1,963 | 555 |

**What this shows:**

- **Area:** nn2rtl uses roughly **8x fewer LUTs** than hls4ml and **12x fewer LUTs** than FINN on average. Flip-flop use is **4x lower** than hls4ml and **11x lower** than FINN. nn2rtl is the only tool that consistently maps multiplies into DSP blocks on these layers.
- **Clock:** nn2rtl Fmax is **about 5x higher** than hls4ml and **3x higher** than FINN, because the generated datapaths are smaller and shallower.
- **Throughput:** nn2rtl is **about 150x slower** than hls4ml and **about 20x slower** than FINN on these layers. That is the cost of the current single-pixel-per-cycle bus and the modest multiplier count per channel.
- **Single message:** the three tools sit at very different points on the area-versus-throughput curve. nn2rtl currently optimises for tiny, fast-clock, DSP-packed modules. hls4ml optimises for highly parallel HLS pipelines. FINN sits in between but pays a large LUT and FF cost for its dataflow shell.

**Caveats:**

- This compares ResNet-50 stage-0 and stage-1 layers only. MobileNetV2 has no FINN or hls4ml data.
- FINN reports `estimated_ooc_throughput_fps`. hls4ml uses the post-synth interval cycles from the C-synthesis report multiplied by the post-synth Fmax. nn2rtl uses post-synth Fmax times measured Verilator cycles per frame. The methodology is consistent within each tool but not identical across tools.
- Most FINN comparison rows had `verification_status: failed`. Throughput is reported as an estimate by FINN itself rather than a measured pass.

## Why nn2rtl Is Different From FINN And hls4ml

**What it is:** nn2rtl is an LLM-driven RTL generation and repair system, not a fixed compiler template.

**Why it works this way:** The research question is not only "can we make the fastest accelerator?" It is "can LLM agents produce, check, repair, and improve RTL at useful scale?"

**Current project details:**

FINN and hls4ml use established compiler flows. They are faster when the input format fits their assumptions. nn2rtl instead explores a different path:

- It generates human-readable Verilog modules.
- It keeps wrong attempts and learns from them.
- It can add new local pattern documents.
- It can try new contracts for new layer families.
- It exposes failures as research data rather than hiding them.

This makes nn2rtl slower today, but more open as a research system.

## Honest Limitations

**What it is:** This section lists the main limits that should be said clearly.

**Why it works this way:** A supervisor will trust the presentation more if the limits are explicit.

**Current project details:**

- The system checks modules one by one. It does not yet build a full network FPGA design.
- ResNet-50 has 119 of 119 modules passing in pipeline state, but per-module evidence on disk is not uniformly fresh. 117 modules have a passing `.results.json`. 2 modules (`node_conv_266`, `node_conv_294`) still carry a stale `status:fail` results file from an earlier attempt that was later superseded. 2 modules (`node_relu_24`, `node_conv_252`) have no `.results.json` on disk. The throughput summary therefore measures 117 of 119 layers.
- MobileNetV2 throughput skips the final `global_avg_pool` and `gemm` layers. These are extracted in LayerIR but were deliberately deferred from this generation run.
- The import report for MobileNetV2 still says `e2e_comparison_reliable: true` based only on blocked cycle ratio. That is too weak, because skipped terminal classifier layers still make full classification output incomplete.
- Current verifier pass is not always strict bit-exact. It allows maximum error up to 3. The reports still record exact mismatch counts.
- The ONNX frontend supports the current MobileNetV2 export, but it is not a universal ONNX compiler. More ONNX patterns would need more frontend work.
- The system depends on LLM calls. This means cost, latency, and occasional unstable generation remain real issues.
- Generated knowledge can overfit. The project now stores applicability metadata, but human review is still important before treating generated docs as universal.
- GAP and Gemm have protected prose docs but no protected reference Verilog yet.
- FINN and hls4ml comparisons are partial and not perfectly apples-to-apples.

## What A Supervisor Should Remember

**What it is:** nn2rtl is a controlled experiment in using LLM agents to build hardware from neural networks.

**Why it works this way:** The system combines creative LLM generation with deterministic checking. The LLM writes and repairs RTL. The TypeScript orchestrator, testbench, and synthesis tools decide whether it is accepted.

**Current project details:**

The core story is:

1. A quantised model becomes LayerIR and goldens.
2. Each LayerIR entry becomes one hardware task.
3. Contracts define the legal hardware interface.
4. Foundry generates Verilog.
5. The static testbench and Vivado judge the result.
6. Surgeon and Retrospector repair failures.
7. Failed RTL is kept as useful evidence.
8. Passing patterns and references become local knowledge.
9. Improve mode tests whether working RTL can be made better without breaking it.

The current results show that this works at meaningful scale:

- ResNet-50 has 119 LayerIR layers and 117 measured throughput layers.
- MobileNetV2 has 99 LayerIR layers and 97 measured throughput layers.
- The system has generated and checked many real Verilog modules.
- The system has extended from ResNet-style convolutions to MobileNet-style ReLU6 and depthwise convolution.

The current results also show the limits clearly:

- Full-network integration is not done.
- MobileNetV2 still lacks measured GAP and Gemm head modules.
- Throughput is behind FINN and hls4ml on the measured comparison rows.
- The main research value is the autonomous generate-check-repair loop and the way local knowledge accumulates over time.
