"""
RQ2 Leg C FINAL -- hls4ml -> Vitis HLS flow for the RETRAINED 89.11% QKeras
ResNet-8 (ZCU104, xczu7ev), tuned to FIT (<100% all resources, csynth estimate).

Input model: /root/rq2_training/qkeras_gpu_fixed_full/resnet8_qkeras8_best_nosoftmax.h5
  (epoch 488, val_accuracy 0.8911; strip-softmax export copy of resnet8_qkeras8_best.h5)

WHAT CHANGED vs the flow-validation config (/root/rq2_training/hls4ml_resnet8):
  KEPT (proven fixes):
    (a) 1x1 s2_proj/s3_proj ReuseFactor clamp to n_chan  (rf_gt_nin codegen bug guard)
    (b) residual-add operand result-precision pin (adder-align quantizer)
  UPDATED for the retrained model:
    * The add-operand QKeras quantizers in THIS model are quantized_bits(8,5,alpha=1)
      (range +/-16), NOT (8,2). hls4ml fuses those standalone QActivation layers away,
      so the Add operands come straight from the bn2/proj feeders. Pin their result
      precision to fixed<8,6,RND_CONV,SAT,0> = the canonical hls4ml mapping of
      quantized_bits(8,5,alpha=1) (int bits = 5 + 1 sign). This re-imposes the exact
      saturation/rounding QKeras applied, so csim stays bit-faithful.
  FIT TUNING (binding constraints in the flow-validation csynth were BRAM_18K 234%
  and LUT 147%; real Vivado logic-synth was LUT 105.6% / BRAM 102.9%):
    * GAP (avg_pool 8x8) was the #1 LUT/FF hog: 117,913 LUT + 65,272 FF with accum=auto.
      -> pin avg_pool accum_t to a TIGHT ufixed<16,10> and result to ufixed<8,4>.
         The 8x8 sum of ufixed<8,4> values (range [0,16)) maxes at ~1024 -> fits
         <16,10> exactly with no overflow; /64 is an exact power-of-2 shift. This
         collapses the wide auto-accum adder tree + divider that drove the blowup.
    * Raise global ReuseFactor 72 -> 128. ResNet-8 is tiny and latency is not on the
      critical path for the comparison; larger RF time-multiplexes the conv MAC arrays
      -> fewer parallel datapaths -> less LUT/FF/DSP per conv. (RF must divide n_in
      where possible; the rf_gt_nin guard handles the 1x1 projections.)
    * Constrain inter-layer / clone FIFO depths. The csynth BRAM estimate is dominated
      by default depth-1024 dataflow FIFOs (1229 of 1465 BRAM_18K). fifo_opt in the
      full build collapses these via cosim profiling; for the csynth ESTIMATE to fit we
      cap the early-stage stream FIFOs explicitly (the 32x32x16 = 16384-element streams
      do not need depth 1024 -- a few rows suffice for the line-buffer conv pipeline).
"""

import argparse
import os
import pickle
import sys

import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--model",
        default="/root/rq2_training/qkeras_gpu_fixed_full/resnet8_qkeras8_best_nosoftmax.h5",
    )
    p.add_argument("--out-dir", default="/root/rq2_training/hls4ml_resnet8_final/prj")
    p.add_argument("--part", default="xczu7ev-ffvc1156-2-e")
    p.add_argument("--clock", type=float, default=10.0)
    p.add_argument("--reuse", type=int, default=128)
    p.add_argument(
        "--data-dir",
        default="/mnt/d/RTL_LLM_CLAUDE/rq2_resnet8/data/cifar-10-batches-py",
    )
    p.add_argument("--n-csim", type=int, default=32)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--fifo-cap", type=int, default=0,
                   help="if >0, cap inter-layer stream FIFO depths to this many beats")
    p.add_argument("--do-csim", action="store_true")
    p.add_argument("--do-csynth", action="store_true")
    return p.parse_args()


def load_cifar10_test(data_dir, n):
    def unpickle(f):
        with open(f, "rb") as fo:
            return pickle.load(fo, encoding="bytes")

    d = unpickle(os.path.join(data_dir, "test_batch"))
    x_test = d[b"data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    y_test = np.array(d[b"labels"])
    # /256.0 EXACTLY as the QAT training did (NOT /255).
    x_test = (x_test / 256.0).astype("float32")
    return x_test[:n], y_test[:n]


def build_config(model, reuse, fifo_cap):
    import hls4ml

    config = hls4ml.utils.config_from_keras_model(
        model,
        granularity="name",
        backend="Vitis",
        default_precision="ap_fixed<16,6>",
        default_reuse_factor=reuse,
    )

    config["Model"]["Strategy"] = "Resource"
    config["Model"]["ReuseFactor"] = reuse
    for lname, lcfg in config["LayerName"].items():
        lcfg["ReuseFactor"] = reuse
        lcfg["Strategy"] = "Resource"

    # ---- (a) 1x1 projection RF clamp (rf_gt_nin codegen bug guard) ----------
    n_chan_by_layer = {}
    for L in model.layers:
        ksz = getattr(L, "kernel_size", None)
        if ksz == (1, 1):
            try:
                n_chan_by_layer[L.name] = int(L.input_shape[-1])
            except Exception:
                n_chan_by_layer[L.name] = int(L.get_input_shape_at(0)[-1])
    for lname, n_in in n_chan_by_layer.items():
        if lname in config["LayerName"] and reuse > n_in and reuse % n_in != 0:
            config["LayerName"][lname]["ReuseFactor"] = n_in
            print("[cfg] clamped %s ReuseFactor %d -> %d (n_in=%d, rf_gt_nin guard)"
                  % (lname, reuse, n_in, n_in), flush=True)

    # ---- (b) adder-align result precision pin (UPDATED for the 8,5 model) ---
    # The retrained model's add-operand QActivations are quantized_bits(8,5,alpha=1)
    # -> canonical hls4ml ap_fixed = fixed<8,6,RND_CONV,SAT,0>. hls4ml fuses those
    # standalone linear quantizers away, so pin the feeders that drive each Add.
    ADD_OPERAND_PREC = "fixed<8,6,RND_CONV,SAT,0>"
    add_feeders = ["s1_bn2", "s2_bn2", "s2_proj", "s3_bn2", "s3_proj"]
    for lname in add_feeders:
        if lname in config["LayerName"]:
            prec = config["LayerName"][lname].setdefault("Precision", {})
            if not isinstance(prec, dict):
                prec = {"result": prec}
                config["LayerName"][lname]["Precision"] = prec
            prec["result"] = ADD_OPERAND_PREC
            print("[cfg] pinned %s result -> %s (adder-align, 8,5 model)"
                  % (lname, ADD_OPERAND_PREC), flush=True)

    # ---- GAP fit: tight accum/result on avg_pool (#1 LUT lever) -------------
    # 8x8 average of ufixed<8,4> values (range [0,16)). sum of 64 <= ~1024 -> ufixed<16,10>
    # holds it exactly; /64 is an exact >>6. Pinning the accum away from `auto` collapses
    # the wide adder tree + divider that produced 117,913 LUT / 65,272 FF.
    if "avg_pool" in config["LayerName"]:
        pp = config["LayerName"]["avg_pool"].setdefault("Precision", {})
        if not isinstance(pp, dict):
            pp = {}
            config["LayerName"]["avg_pool"]["Precision"] = pp
        pp["accum"] = "ufixed<16,10>"
        pp["result"] = "ufixed<8,4>"
        config["LayerName"]["avg_pool"]["ReuseFactor"] = reuse
        print("[cfg] avg_pool accum->ufixed<16,10> result->ufixed<8,4> (GAP fit)",
              flush=True)

    return config


def maybe_cap_fifos(config, fifo_cap):
    """Cap the default deep inter-layer dataflow FIFOs in the csynth estimate.
    The full build's fifo_opt does this exactly from cosim; this is the static
    fallback so the *estimate* fits. Applied per-layer where supported."""
    if fifo_cap <= 0:
        return
    # hls4ml exposes per-LayerName 'fifo_depth' attr that lands on the layer's
    # output stream pragma in io_stream.
    for lname, lcfg in config.get("LayerName", {}).items():
        lcfg.setdefault("fifo_depth", fifo_cap)
    print("[cfg] capped per-layer fifo_depth -> %d beats" % fifo_cap, flush=True)


def main():
    args = parse_args()
    os.environ.setdefault("OMP_NUM_THREADS", str(args.threads))
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

    import tensorflow as tf

    tf.config.threading.set_intra_op_parallelism_threads(args.threads)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)

    sys.path.insert(0, "/root/rq2_training/qkeras")
    from resnet8_qkeras8 import load_qkeras_h5

    import hls4ml

    print("[hls4ml] version %s" % hls4ml.__version__, flush=True)

    model = load_qkeras_h5(args.model)
    print("[hls4ml] loaded %s (%d params)" % (args.model, model.count_params()), flush=True)

    config = build_config(model, args.reuse, args.fifo_cap)
    maybe_cap_fifos(config, args.fifo_cap)

    print("[hls4ml] config built: Strategy=Resource ReuseFactor=%d (%d layer entries)"
          % (args.reuse, len(config.get("LayerName", {}))), flush=True)

    hls_model = hls4ml.converters.convert_from_keras_model(
        model,
        hls_config=config,
        output_dir=args.out_dir,
        backend="Vitis",
        io_type="io_stream",
        part=args.part,
        clock_period=args.clock,
    )
    print("[hls4ml] convert_from_keras_model OK (clock=%.1fns part=%s reuse=%d)"
          % (args.clock, args.part, args.reuse), flush=True)

    hls_model.compile()
    print("[hls4ml] hls_model.compile() OK", flush=True)

    if args.do_csim:
        x, y = load_cifar10_test(args.data_dir, args.n_csim)
        qk_logits = model.predict(x, verbose=0)
        qk_pred = qk_logits.argmax(axis=1)
        hls_logits = hls_model.predict(np.ascontiguousarray(x))
        hls_logits = np.asarray(hls_logits).reshape(qk_logits.shape)
        hls_pred = hls_logits.argmax(axis=1)

        agree = int((qk_pred == hls_pred).sum())
        qk_acc = float((qk_pred == y).mean())
        hls_acc = float((hls_pred == y).mean())
        max_abs = float(np.max(np.abs(qk_logits - hls_logits)))
        print("[csim] N=%d  QKeras-vs-hls4ml argmax agreement=%d/%d  max|logit diff|=%.4f"
              % (args.n_csim, agree, args.n_csim, max_abs), flush=True)
        print("[csim] QKeras top-1 on these=%.3f  hls4ml top-1 on these=%.3f"
              % (qk_acc, hls_acc), flush=True)
        print("CSIM_AGREE=%d/%d MAXDIFF=%.4f QK_ACC=%.4f HLS_ACC=%.4f"
              % (agree, args.n_csim, max_abs, qk_acc, hls_acc), flush=True)

    if args.do_csynth:
        print("[csynth] launching C synthesis...", flush=True)
        report = hls_model.build(
            reset=True, csim=False, synth=True, cosim=False,
            export=False, vsynth=False, fifo_opt=False, log_to_stdout=True,
        )
        print("[csynth] DONE", flush=True)
        try:
            import pprint
            pprint.pprint(report)
        except Exception:
            print(report)
        print("CSYNTH_OK", flush=True)

    print("CONVERT_DONE", flush=True)


if __name__ == "__main__":
    main()
