# Reference-file provenance

Three reference files are load-bearing in the pattern library today.
All written in this repo, same license as the rest of the codebase.

## `conv1x1_passing_reference.v`

- **Shape**: `layer1_0_conv1` (IC=OC=64, IH=IW=112, KH=KW=1, MP=4) in
  concrete form, with the registered-`mul_q` DSP48E1 MAC pattern.
- **Used by**: `02_conv1x1.md` and the `get_rtl_patterns` MCP tool tell
  Foundry to adapt this file for any pointwise (1×1) `conv2d`. Treat
  its control structure as a proven template; adapt the localparam
  block (IC, OC, IH, IW, MP, SCALE_MULT, SCALE_SHIFT) and the two
  `$readmemh` paths to the new LayerIR.

## `conv3x3_passing_reference.v`

- **Shape**: `layer1_0_conv2` (IC=OC=64, IH=IW=112, KH=KW=3, stride=1,
  padding=1, MP=4) in concrete form. The whole body is library-module
  instantiation (`coord_scheduler` + `line_buf_window` + `conv_datapath`)
  per the split spatial-conv architecture documented in
  `rtl_library/SPLIT_ARCHITECTURE.md`.
- **Used by**: `03_conv3x3_pad1.md` and the `get_rtl_patterns` MCP
  tool tell Foundry to adapt this file for any 3×3 spatial `conv2d`.
  Adapt the localparam block (IC, OC, IH, IW, OH, OW, SH, SW, PH, PW,
  MP, SCALE_MULT, SCALE_SHIFT) and the two `WEIGHTS_PATH` /
  `BIAS_PATH` parameters on the `conv_datapath` instantiation. Do
  NOT add extra `always` blocks beyond the single `start_pulse`
  one shown — the rest of the FSM lives in the library modules.

## `conv7x7_passing_reference.v`

- **Shape**: `layer0_0_conv1` (IC=3, OC=64, IH=IW=224, OH=OW=112,
  KH=KW=7, stride=2, padding=3, MP=4) in concrete form. Same split-
  architecture skeleton as `conv3x3_passing_reference.v` (the library
  modules are kernel-agnostic); only the localparam block and the
  asymmetric bus widths (`data_in [23:0]`, `data_out [511:0]`) differ.
- **Used by**: `04_conv7x7_pad3.md` and the `get_rtl_patterns` MCP
  tool tell Foundry to adapt this file for any 7×7 stem-shaped
  `conv2d`. Adapt the localparam block (IC, OC, IH, IW, OH, OW, SH,
  SW, PH, PW, MP, SCALE_MULT, SCALE_SHIFT) and the two `WEIGHTS_PATH`
  / `BIAS_PATH` parameters on the `conv_datapath` instantiation. Do
  NOT add extra `always` blocks beyond the single `start_pulse`
  one shown.

## Rule for adding future references

Every file in `knowledge/references/{protected,active,probationary}/`
must be actually read by `get_rtl_patterns` / `foundry.md`'s catalog and
actually referenced by a `knowledge/patterns/{protected,active,probationary}/*.md`
file for adaptation. If neither is true, archive or delete the file —
provenance-only references are noise that mislead both reviewers and the
agent.

When adding a new reference, record here: source repo / shape it
embodies / license, and which pattern file + MCP-tool branch consumes
it.
