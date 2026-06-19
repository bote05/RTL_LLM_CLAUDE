# RQ2 Leg C -- hls4ml -> Vitis HLS flow for QKeras ResNet-8 (ZCU104 / xczu7ev)

## Status: FLOW VALIDATED. csynth COMPLETE. P&R LAUNCHED.

Date: 2026-06-13. Tools: hls4ml 1.3.0 (Vitis backend), Vitis HLS 2024.2,
Vivado 2024.2. Part xczu7ev-ffvc1156-2-e, clock 10 ns. io_stream + Strategy
Resource + ReuseFactor 72.

## Model used (CURRENT, for flow/resource/timing validation only)
`/root/rq2_training/qkeras/resnet8_qkeras8_best_nosoftmax.h5` (78,666 params),
the first 300-epoch QKeras-8bit QAT run: best val_acc 0.8267, final test 0.8238.
A higher-accuracy retrain (target >=85% MLPerf Tiny closed floor) is planned;
**FINAL ACCURACY NUMBERS WILL COME FROM RE-RUNNING THIS EXACT FLOW ON THE
RETRAINED QKeras MODEL.** Swap the `--model` path in convert_resnet8.py /
run_csynth.py / run_export_pnr.py and re-run; nothing else changes.

## csim numerical check (QKeras vs hls4ml C++ bit-accurate sim)
N=64 CIFAR-10 test images, /256.0 preprocessing (matches QAT training):
  argmax agreement = 64/64
  max |logit diff| = 0.25
  hls4ml top-1 = QKeras top-1 = 0.828 (on the 64-image slice)

## csynth results (Vitis HLS C synthesis, 25:38 elapsed)
  Clock target 10 ns, ESTIMATED 7.282 ns -> Estimated Fmax 137.32 MHz
  Latency: 90,146-90,195 cycles  = 0.901-0.902 ms  (dataflow)
  Interval (II): 4,626 (min) .. 90,170 (max)  -> dataflow pipeline
  Resource ESTIMATE (xczu7ev):
    LUT  339,196 / 230,400 = 147 %   (OVER budget)
    FF   169,908 / 460,800 =  36 %
    DSP     831 /   1,728  =  48 %
    BRAM_18K 1,465 /   624 = 234 %   (OVER budget; pre-FIFO-opt)
    URAM      0

## Two real bugs found + fixed (both are in convert_resnet8.py)
1. hls4ml 1.3.0 codegen bug: the Resource dense kernel selector emits
   `nnet::DenseResource_rf_gt_nin<>` when reuse_factor > n_in and
   reuse_factor % n_in != 0, but that CamelCase template is NOT defined in
   nnet_dense.h (only _rf_leq_nin and _rf_gt_nin_rem0 exist) -> hard C++ compile
   error. Bit the two 1x1 projection convs (s2_proj n_in=16, s3_proj n_in=32).
   FIX: clamp those layers' ReuseFactor to n_in -> takes the working
   rf_leq_nin path (full reuse, no DSP penalty, numerically identical).
2. Dropped adder-alignment quantizers (correctness): the QKeras
   QActivation(quantized_bits(8,2,alpha=1)) layers on the residual-add operands
   (s*_branch_q, s*_proj_q) SATURATE to [-4,4). hls4ml fuses these standalone
   *linear* QKeras quantizers away, so unsaturated BN/proj outputs (up to +-110)
   fed straight into the Add -> residual sum exploded (traced: s2_add maxdiff 34,
   s3_add maxdiff 107, csim 6/16 argmax, top-1 0.44). FIX: pin the result
   precision of the layers feeding each Add (s1_bn2, s2_bn2, s2_proj, s3_bn2,
   s3_proj) to fixed<8,3,RND_CONV,SAT,0> (hls4ml's canonical map of
   quantized_bits(8,2)). After fix: csim 64/64, max logit diff 0.25.

## Resource over-budget -- known, expected for this validation run
LUT 147% is dominated by `pooling2d_cl` (avg_pool 8x8 GAP) at 117,913 LUT /
65,272 FF, plus default array partitioning (the hls4ml-emitted tcl uses
`config_array_partition -maximum_size 4096`, a directive REMOVED in Vitis HLS
2024.2 -> errored non-fatally -> default partitioning). BRAM 234% is pre-FIFO-opt.
The launched P&R run enables fifo_opt=True (cosim-profiled skip-FIFO depths,
controls BRAM). To fit, future tuning levers (apply after the retrained model):
  - serialize / ram_style the avg_pool GAP accumulator (biggest LUT win)
  - raise ReuseFactor (72 -> higher) to time-multiplex conv MACs
  - revisit the array_partition tcl for 2024.2 compatibility
This run's purpose was flow + numbers, not a fitting netlist.

## Files (all mirrored to /mnt/d/RTL_LLM_CLAUDE/rq2_resnet8/hls4ml/)
  convert_resnet8.py   - convert + compile + csim (the validated config)
  run_csynth.py        - re-convert + write + csynth
  run_export_pnr.py    - re-convert + write + fifo_opt + synth + export + Vivado P&R
  watch_then_pnr.sh    - waits for csynth, extracts numbers, auto-launches P&R
  trace_divergence.py  - per-layer QKeras-vs-hls4ml profiling (localized bug #2)
  analyze_rf.py        - reuse-factor / kernel-selection audit (localized bug #1)
  CSYNTH_SUMMARY.txt   - extracted csynth numbers
  myproject_csynth.rpt - full Vitis HLS csynth report
HLS project: /root/rq2_training/hls4ml_resnet8/prj (WSL)
