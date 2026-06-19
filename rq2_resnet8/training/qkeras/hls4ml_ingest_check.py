"""
hls4ml v1.3.0 ingest check for the QKeras ResNet-8 -- RQ2 Leg C.

Loads the strip-softmax .h5, runs hls4ml.utils.config_from_keras_model +
convert_from_keras_model (io_stream, Vitis backend, xczu7ev-ffvc1156-2-e) and
writes the HLS project. NO synthesis is run -- conversion + write completing is
the integration gate this script exists to retire.
"""

import argparse
import os
import sys


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--model",
        default="/root/rq2_training/qkeras/resnet8_qkeras8_best_nosoftmax.h5",
    )
    p.add_argument("--out-dir", default="/root/rq2_training/qkeras/hls4ml_ingest")
    p.add_argument("--part", default="xczu7ev-ffvc1156-2-e")
    p.add_argument("--threads", type=int, default=4)
    args = p.parse_args()

    os.environ.setdefault("OMP_NUM_THREADS", str(args.threads))
    import tensorflow as tf

    tf.config.threading.set_intra_op_parallelism_threads(args.threads)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from resnet8_qkeras8 import load_qkeras_h5

    import hls4ml

    print("[hls4ml] version %s" % hls4ml.__version__, flush=True)

    model = load_qkeras_h5(args.model)
    print("[hls4ml] loaded %s (%d params)" % (args.model, model.count_params()), flush=True)

    config = hls4ml.utils.config_from_keras_model(
        model,
        granularity="name",
        backend="Vitis",
        default_precision="ap_fixed<16,6>",
    )
    print("[hls4ml] config_from_keras_model OK (%d layer entries)"
          % len(config.get("LayerName", {})), flush=True)

    hls_model = hls4ml.converters.convert_from_keras_model(
        model,
        hls_config=config,
        output_dir=args.out_dir,
        backend="Vitis",
        io_type="io_stream",
        part=args.part,
    )
    print("[hls4ml] convert_from_keras_model OK", flush=True)

    hls_model.write()  # emit HLS project files; NO synthesis
    print("[hls4ml] project written to %s" % args.out_dir, flush=True)
    print("HLS4ML_INGEST_OK", flush=True)


if __name__ == "__main__":
    main()
