---
task_id: 01
title: Weight memory map generator
type: Python tooling
status: review
depends_on: []
unblocks: [02, 03, 07, 09]
---

# Task 01 — Weight memory map generator

## Goal

Write a Python script that reads ResNet-50's `output/layer_ir.json` plus the existing per-layer `output/weights/*.hex` files, computes a deterministic per-layer URAM base-address layout, and emits:

1. A single combined `.mem` file (Verilog `$readmemh` format) representing the full URAM image.
2. A generated Verilog header (`weight_memory_map.vh`) of `localparam` declarations giving each layer's base address and size, indexable by module ID at the synthesis stage.
3. A JSON sidecar (`weight_memory_map.json`) holding the same layout in machine-readable form, for the scheduler and other downstream tools.

This is pure deterministic Python tooling. No LLM involved. The output feeds the engine (task 09 reads the addresses) and the top-level wrapper (task 02 references the URAM region).

## Deliverable

A new script at `scripts/build_weight_memory_map.py` plus the three output artefacts described above.

### Script behaviour

- Reads `output/layer_ir.json` (path overridable via `--layer-ir`).
- For each layer where `op_type == "conv2d"` and `weights_path` is set:
  - Reads the existing `.hex` file from `weights_path`.
  - Aligns the layer's weight region to a URAM word boundary (288 bits = 36 bytes). Pad with zero bytes if the layer's weight size is not a multiple of 36.
  - Assigns a base address (in URAM-word units) starting from 0 and growing sequentially.
- Concatenates all aligned weight regions into a single `.mem` file.
- Emits the Verilog header with one `localparam` per conv layer:
  ```verilog
  localparam WBASE_node_conv_196_WORDS = 0;
  localparam WSIZE_node_conv_196_WORDS = 261;
  localparam WBASE_node_conv_198_WORDS = 261;
  ...
  ```
- Emits the JSON sidecar with the same data plus the total URAM word count used:
  ```json
  {
    "uram_word_bits": 288,
    "total_words_used": NNNN,
    "total_uram_blocks_required": MMM,
    "uram_capacity_blocks": 1280,
    "utilisation_pct": XX.X,
    "layers": [
      { "module_id": "node_conv_196", "base_word": 0, "size_words": 261, "size_bytes": 9408, "padded_to_words": 261 },
      ...
    ]
  }
  ```

### CLI

```
python scripts/build_weight_memory_map.py \
    --network=resnet-50 \
    [--layer-ir=output/layer_ir.json] \
    [--out-mem=output/weights/uram_weights.mem] \
    [--out-header=output/weights/weight_memory_map.vh] \
    [--out-json=output/weights/weight_memory_map.json]
```

Use `networks.json` to resolve the default output root per network.

## Context (read this before starting)

- The deployment plan §3 fixes the architecture: all 22.4 MB of weights pre-loaded into URAM at bitfile load. No DDR.
- The deployment plan §6.7 specifies that the weight memory layout is deterministic and author-controlled (not LLM-generated): "About 100 lines of Python."
- Existing `.hex` files were written by the nn2rtl frontend using INT8 byte-per-line format. One hex byte per line, no headers. Per-layer file paths are stored in LayerIR under `weights_path`.
- The U250 has 1,280 UltraRAM blocks of 288 Kbit (= 36 KB) each. Each URAM block is 4,096 words of 288 bits.
- ResNet-50 weight footprint (from a previous audit): 22.4 MB across 53 conv2d layers. Largest single layer: `node_conv_298` at 2,359,296 bytes.

## `.mem` line format (pinned)

`$readmemh` in IEEE 1364 / IEEE 1800 accepts whitespace-separated hex values; line breaks are treated as whitespace. **This task uses one hex value per line**, where one "value" is one full 288-bit URAM word, rendered as exactly 72 hexadecimal characters (no `0x` prefix, no leading zero stripping), followed by a single `\n`. No blank lines, no comments, no addresses.

Example (one URAM word = 288 bits = 72 hex chars):

```
0001020304...7f808182838485 ...                          (72 chars)\n
0102030405...807e8a8b8c8d8e ...                          (72 chars)\n
```

Each URAM word holds 36 INT8 weight bytes. The script packs them little-endian within the word: byte 0 (the first weight) occupies bits [7:0], byte 1 occupies bits [15:8], …, byte 35 occupies bits [287:280]. The hex rendering emits the most-significant byte first (so the leftmost two hex chars of the line are byte 35, the rightmost two are byte 0). This matches Verilog's natural `$readmemh` interpretation of a packed register.

This format is verified by `$readmemh` in iverilog, Verilator, and Vivado XSIM — all three accept it.

## How to verify

1. Run the script. It should print:
   - Total bytes written
   - Total URAM words used
   - URAM blocks required (= ceil(words / 4096))
   - Utilisation percentage against the 1,280-block budget
2. Manually check one layer's address calculation: for `node_conv_196` (9,408 bytes of weights), the size in 288-bit words should be `ceil(9408 / 36) = 262 words`. Base address starts at 0.
3. The script must be deterministic: running it twice on the same input produces byte-identical `.mem`, `.vh`, and `.json` files.
4. Total URAM blocks required must be ≤ 1,280 (the U250 budget). If exceeded, the script must exit non-zero and print which layer pushed it over.
5. The `.mem` file is `$readmemh`-loadable in the pinned format above. A small test harness in `output/rtl/test_uram_load.v` (instantiates a `reg [287:0] mem [0:N-1]`, runs `$readmemh("path/to/.mem", mem)`, prints `mem[0]` and `mem[N-1]`) compiles under iverilog and prints the expected first / last URAM words. Compare against a Python script that reads the original `.hex` files and computes the same expected first / last word values.

## Out of scope

- Do NOT generate Verilog beyond the header file. The engine's actual URAM instantiation lives in task 00's skeleton.
- Do NOT modify `output/weights/*.hex` source files. The script only reads them.
- Do NOT touch LayerIR. The script consumes it read-only.
- Do NOT add bias data to this memory map. Biases (much smaller, per-output-channel INT32) are stored separately by the existing per-layer mechanism. This task is for the INT8 weight tensors only.

## Success criteria

- Script runs in under 5 seconds end-to-end.
- Output `.mem` file is exactly `total_words_used × 36` bytes (hex-encoded as `(36 × 2 + 1)` characters per line including newline).
- Generated Verilog header is `$include`-able with no syntax errors (`iverilog -t null` against a tiny wrapper succeeds).
- JSON sidecar validates against a simple schema (the four required top-level fields all present and consistent).
- Final reported URAM utilisation matches the deployment plan §3 estimate (~50% URAM with the deployment plan's headroom).
