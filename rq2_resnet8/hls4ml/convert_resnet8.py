"""
RQ2 Leg C -- hls4ml -> Vitis HLS flow for the QKeras 8-bit ResNet-8 (ZCU104, xczu7ev).

This script proves the QKeras -> hls4ml -> Vitis HLS flow end-to-end on the CURRENT
QKeras model and produces resource/timing numbers. The model will be swapped for the
higher-accuracy retrain afterward; this run is for flow + resource/timing validation.

Recipe (verified precedent for ResNet-8, Tailor / official MLPerf Tiny RN08):
  * io_stream + Strategy Resource
  * ReuseFactor start ~72
  * ap_fixed precision matching the QKeras 8-bit quantizers (granularity=name auto-derives
    per-layer precision from the QKeras kernel/bias/activation quantizers)
  * skip-connection Add implemented via Clone + skip-FIFO + Add merge; FIFO-depth
    optimizer enabled in the full build (avoids skip-FIFO deadlock + BRAM blowup)

Steps: load model -> config_from_keras_model -> convert_from_keras_model (Vitis,
xczu7ev-ffvc1156-2-e, clock 10ns) -> write -> csim (numerical vs QKeras) -> csynth.
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
        default="/root/rq2_training/qkeras/resnet8_qkeras8_best_nosoftmax.h5",
    )
    p.add_argument("--out-dir", default="/root/rq2_training/hls4ml_resnet8/prj")
    p.add_argument("--part", default="xczu7ev-ffvc1156-2-e")
    p.add_argument("--clock", type=float, default=10.0, help="clock period ns")
    p.add_argument("--reuse", type=int, default=72)
    p.add_argument(
        "--data-dir",
        default="/mnt/d/RTL_LLM_CLAUDE/rq2_resnet8/data/cifar-10-batches-py",
    )
    p.add_argument("--n-csim", type=int, default=16, help="num CIFAR images for csim")
    p.add_argument("--threads", type=int, default=6)
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

    # granularity=name -> hls4ml reads the QKeras quantizers per layer and derives the
    # matching ap_fixed precision for weights/biases/activations automatically.
    config = hls4ml.utils.config_from_keras_model(
        model,
        granularity="name",
        backend="Vitis",
        default_precision="ap_fixed<16,6>",
        default_reuse_factor=args.reuse,
    )

    # Strategy Resource at the model level + per-layer reuse factor (the recipe).
    config["Model"]["Strategy"] = "Resource"
    config["Model"]["ReuseFactor"] = args.reuse
    for lname, lcfg in config["LayerName"].items():
        # Only the weight-bearing layers honour ReuseFactor; harmless to set broadly.
        lcfg["ReuseFactor"] = args.reuse
        lcfg["Strategy"] = "Resource"

    # ---- per-layer reuse-factor clamp (hls4ml 1.3.0 codegen bug guard) -----
    # The Resource dense kernel selector emits nnet::DenseResource_rf_gt_nin<>
    # when reuse_factor > n_in AND reuse_factor % n_in != 0. That CamelCase
    # template is NOT defined in this nnet_dense.h (only _rf_leq_nin and
    # _rf_gt_nin_rem0 exist) -> hard C++ compile error. It bites the two 1x1
    # projection convs (s2_proj n_chan=16, s3_proj n_chan=32) where n_in is the
    # channel count. Clamp those layers' RF to n_chan so they take the working
    # rf_leq_nin path (full reuse, no DSP penalty, numerically identical).
    n_chan_by_layer = {}
    for L in model.layers:
        ksz = getattr(L, "kernel_size", None)
        if ksz == (1, 1):  # 1x1 conv -> n_in == input channels
            try:
                n_chan_by_layer[L.name] = int(L.input_shape[-1])
            except Exception:
                n_chan_by_layer[L.name] = int(L.get_input_shape_at(0)[-1])
    for lname, n_in in n_chan_by_layer.items():
        if lname in config["LayerName"] and args.reuse > n_in and args.reuse % n_in != 0:
            config["LayerName"][lname]["ReuseFactor"] = n_in
            print("[hls4ml] clamped %s ReuseFactor %d -> %d (n_in=%d, rf_gt_nin bug guard)"
                  % (lname, args.reuse, n_in, n_in), flush=True)

    # ---- restore the adder-alignment quantizers (correctness fix) ---------
    # The QKeras QActivation(quantized_bits(8,2,alpha=1)) layers on the two add
    # operands (s*_branch_q on the conv branch, s*_proj_q on the projection
    # skip) SATURATE to [-4, 4). hls4ml fuses these standalone *linear* QKeras
    # quantizers away, so the unsaturated BN/proj outputs (range up to +-110)
    # feed straight into the Add -> the residual sum explodes and accuracy
    # collapses (traced: s2_add maxdiff 34, s3_add maxdiff 107 vs QKeras).
    # Re-impose the saturation by pinning the result precision of the layers
    # that feed each Add to the QKeras quantizer's fixed<8,3,RND_CONV,SAT>
    # format (hls4ml's canonical mapping of quantized_bits(8,2)).
    ADD_OPERAND_PREC = "fixed<8,3,RND_CONV,SAT,0>"
    add_feeders = ["s1_bn2", "s2_bn2", "s2_proj", "s3_bn2", "s3_proj"]
    for lname in add_feeders:
        if lname in config["LayerName"]:
            prec = config["LayerName"][lname].setdefault("Precision", {})
            if not isinstance(prec, dict):
                prec = {"result": prec}
                config["LayerName"][lname]["Precision"] = prec
            prec["result"] = ADD_OPERAND_PREC
            print("[hls4ml] pinned %s result precision -> %s (adder-align quantizer)"
                  % (lname, ADD_OPERAND_PREC), flush=True)
    # The identity skip into s1_add is the stem/relu_out path -- already on the
    # ufixed<8,2> quantized_relu grid (range [0,4)), so it needs no extra pin.

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
    print("[hls4ml] convert_from_keras_model OK (clock=%.1fns part=%s)"
          % (args.clock, args.part), flush=True)

    hls_model.compile()
    print("[hls4ml] hls_model.compile() OK", flush=True)

    # ---- csim numerical check vs QKeras predictions -----------------------
    if args.do_csim:
        x, y = load_cifar10_test(args.data_dir, args.n_csim)
        # QKeras (software) reference logits.
        qk_logits = model.predict(x, verbose=0)
        qk_pred = qk_logits.argmax(axis=1)
        # hls4ml (bit-accurate C++ sim via the compiled bridge).
        hls_logits = hls_model.predict(np.ascontiguousarray(x))
        hls_logits = np.asarray(hls_logits).reshape(qk_logits.shape)
        hls_pred = hls_logits.argmax(axis=1)

        agree = int((qk_pred == hls_pred).sum())
        qk_acc = float((qk_pred == y).mean())
        hls_acc = float((hls_pred == y).mean())
        max_abs = float(np.max(np.abs(qk_logits - hls_logits)))
        print("[csim] N=%d  QKeras-vs-hls4ml argmax agreement=%d/%d  "
              "max|logit diff|=%.4f" % (args.n_csim, agree, args.n_csim, max_abs),
              flush=True)
        print("[csim] QKeras top-1 on these=%.3f  hls4ml top-1 on these=%.3f"
              % (qk_acc, hls_acc), flush=True)
        print("CSIM_AGREE=%d/%d MAXDIFF=%.4f" % (agree, args.n_csim, max_abs), flush=True)

    # ---- csynth (C synthesis: latency/II + resource estimate) -------------
    if args.do_csynth:
        print("[csynth] launching C synthesis (this can take ~20-60 min)...", flush=True)
        report = hls_model.build(
            reset=True,
            csim=False,      # numerical check already done above via predict()
            synth=True,      # C synthesis -> latency/II + resource estimate
            cosim=False,
            export=False,
            vsynth=False,
            fifo_opt=False,  # csynth pass; fifo_opt (cosim-based) reserved for full build
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
