"""
Train the QKeras 8-bit QAT ResNet-8 on CIFAR-10 -- RQ2 Leg C (hls4ml).

Recipe mirrors the OFFICIAL MLPerf Tiny training reference
(mlcommons/tiny benchmark/training/image_classification/train.py, which reached
~87% float top-1) crossed with the official v1.0 open/hls4ml RN08 QAT submission:
  optimizer    : Adam
  LR schedule  : 0.001 * 0.99^epoch          (official train.py lr_schedule)
  batch size   : 32                          (official)
  augmentation : rotation 15, w/h shift 0.1, horizontal flip (official datagen)
  input scale  : X/256.0                     (official QAT submission; the float
                                              reference trains on raw 0-255, but for
                                              QAT the /256 input maps cleanly to an
                                              8-bit fixed-point input in hls4ml)
  epochs       : 300 default (official float ref used 500; see --epochs)

Artifacts (under --out-dir, default /root/rq2_training/qkeras):
  ckpt/epoch_NNN.h5                    every --ckpt-every epochs (default 25)
  resnet8_qkeras8_best.h5              best test-set accuracy (training head, softmax)
  resnet8_qkeras8_best_nosoftmax.h5    strip-softmax copy for hls4ml ingest
  resnet8_qkeras8_last.h5              final-epoch model
  best.json / train_history.csv        bookkeeping

Test-set eval runs every --eval-every epochs (default 10) via validation_freq.

Thread cap: --threads (default 4) is applied to TF inter/intra op BEFORE any op
runs. Keep OMP_NUM_THREADS=4 in the environment as well (host Vivado route in
flight -- do not oversubscribe).
"""

import argparse
import json
import os
import pickle
import random
import sys
import time

import numpy as np

DEFAULT_DATA_DIR = "/mnt/d/RTL_LLM_CLAUDE/rq2_resnet8/data/cifar-10-batches-py"
DEFAULT_OUT_DIR = "/root/rq2_training/qkeras"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    p.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--lr-decay", type=float, default=0.99)
    p.add_argument("--eval-every", type=int, default=10)
    p.add_argument("--ckpt-every", type=int, default=25)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--batch-log-every",
        type=int,
        default=250,
        help="print running loss every N train batches (0 = off)",
    )
    return p.parse_args()


def load_cifar10(data_dir):
    """Load CIFAR-10 python batches exactly like the official train.py."""

    def unpickle(f):
        with open(f, "rb") as fo:
            return pickle.load(fo, encoding="bytes")

    xs, ys = [], []
    for i in range(1, 6):
        d = unpickle(os.path.join(data_dir, "data_batch_%d" % i))
        xs.append(d[b"data"])
        ys += list(d[b"labels"])
    x_train = np.vstack(xs).reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    y_train = np.array(ys)

    d = unpickle(os.path.join(data_dir, "test_batch"))
    x_test = d[b"data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    y_test = np.array(d[b"labels"])
    return x_train, y_train, x_test, y_test


def main():
    args = parse_args()

    # Thread caps BEFORE any TF op executes (Vivado route on the host).
    os.environ.setdefault("OMP_NUM_THREADS", str(args.threads))
    os.environ.setdefault("TF_NUM_INTRAOP_THREADS", str(args.threads))
    os.environ.setdefault("TF_NUM_INTEROP_THREADS", str(args.threads))
    import tensorflow as tf

    tf.config.threading.set_intra_op_parallelism_threads(args.threads)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)

    from tensorflow.keras.callbacks import (
        Callback,
        CSVLogger,
        LearningRateScheduler,
    )
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from tensorflow.keras.utils import to_categorical

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from resnet8_qkeras8 import assert_param_contract, build_resnet8_qkeras

    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    out_dir = args.out_dir
    ckpt_dir = os.path.join(out_dir, "ckpt")
    os.makedirs(ckpt_dir, exist_ok=True)

    print("[cfg] %s" % json.dumps(vars(args)), flush=True)

    # ---- data --------------------------------------------------------------
    x_train, y_train, x_test, y_test = load_cifar10(args.data_dir)
    # /256 like the official QAT submission (NOT /255): clean 8-bit fixed-point.
    x_train = (x_train / 256.0).astype("float32")
    x_test = (x_test / 256.0).astype("float32")
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)
    print(
        "[data] train %s test %s" % (str(x_train.shape), str(x_test.shape)),
        flush=True,
    )

    # ---- model -------------------------------------------------------------
    builder_kwargs = dict(
        num_filters=16, total_bits=8, weight_int_bits=0, act_int_bits=2, alpha=1
    )
    model = build_resnet8_qkeras(final_activation=True, **builder_kwargs)
    n_params = assert_param_contract(model)
    print("[model] resnet8_qkeras8 params=%d (contract OK)" % n_params, flush=True)
    model.summary()

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    # ---- official augmentation ----------------------------------------------
    datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
    )

    # ---- official LR schedule -----------------------------------------------
    def lr_schedule(epoch):
        lr = args.lr * (args.lr_decay ** epoch)
        print("[lr] epoch %d lr=%g" % (epoch + 1, lr), flush=True)
        return lr

    # ---- callbacks -----------------------------------------------------------
    class EpochTimer(Callback):
        def on_epoch_begin(self, epoch, logs=None):
            self.t0 = time.time()

        def on_epoch_end(self, epoch, logs=None):
            print(
                "[time] epoch %d wall %.1fs" % (epoch + 1, time.time() - self.t0),
                flush=True,
            )

    class BatchLossLogger(Callback):
        def on_train_batch_end(self, batch, logs=None):
            ev = args.batch_log_every
            if ev and batch % ev == 0:
                print(
                    "[batch] %d loss %.4f acc %.4f"
                    % (batch, (logs or {}).get("loss", -1), (logs or {}).get("accuracy", -1)),
                    flush=True,
                )

    class PeriodicCheckpoint(Callback):
        def on_epoch_end(self, epoch, logs=None):
            if (epoch + 1) % args.ckpt_every == 0:
                path = os.path.join(ckpt_dir, "epoch_%03d.h5" % (epoch + 1))
                self.model.save(path)
                print("[ckpt] saved %s" % path, flush=True)

    class BestSaver(Callback):
        """Fires on epochs where validation ran (validation_freq)."""

        def __init__(self):
            super().__init__()
            self.best = -1.0

        def save_best(self, epoch, val_acc):
            self.best = val_acc
            best_path = os.path.join(out_dir, "resnet8_qkeras8_best.h5")
            self.model.save(best_path)
            # strip-softmax copy for hls4ml (same weighted layers, exact transfer)
            nosm = build_resnet8_qkeras(final_activation=False, **builder_kwargs)
            nosm.set_weights(self.model.get_weights())
            nosm_path = os.path.join(out_dir, "resnet8_qkeras8_best_nosoftmax.h5")
            nosm.save(nosm_path)
            with open(os.path.join(out_dir, "best.json"), "w") as f:
                json.dump(
                    {
                        "epoch": int(epoch + 1),
                        "val_accuracy": float(val_acc),
                        "best_h5": best_path,
                        "best_nosoftmax_h5": nosm_path,
                    },
                    f,
                    indent=2,
                )
            print(
                "[best] epoch %d val_acc %.4f -> %s (+nosoftmax)"
                % (epoch + 1, val_acc, best_path),
                flush=True,
            )

        def on_epoch_end(self, epoch, logs=None):
            va = (logs or {}).get("val_accuracy")
            if va is None:
                return
            print("[eval] epoch %d val_acc %.4f" % (epoch + 1, va), flush=True)
            if va > self.best:
                self.save_best(epoch, float(va))

    best_saver = BestSaver()
    callbacks = [
        LearningRateScheduler(lr_schedule),
        EpochTimer(),
        BatchLossLogger(),
        PeriodicCheckpoint(),
        best_saver,
        CSVLogger(os.path.join(out_dir, "train_history.csv"), append=True),
    ]

    # ---- train ---------------------------------------------------------------
    steps = int(np.ceil(len(x_train) / float(args.batch_size)))
    val_freq = min(args.eval_every, args.epochs)
    t_start = time.time()
    model.fit(
        datagen.flow(x_train, y_train, batch_size=args.batch_size),
        steps_per_epoch=steps,
        epochs=args.epochs,
        validation_data=(x_test, y_test),
        validation_freq=val_freq,
        callbacks=callbacks,
        verbose=2,
    )
    t_total = time.time() - t_start

    # ---- finalize --------------------------------------------------------------
    last_path = os.path.join(out_dir, "resnet8_qkeras8_last.h5")
    model.save(last_path)
    loss, acc = model.evaluate(x_test, y_test, verbose=0)
    print("[final] test loss %.4f acc %.4f" % (loss, acc), flush=True)
    if acc > best_saver.best:
        best_saver.save_best(args.epochs - 1, float(acc))
    print(
        "[done] %d epochs in %.1fs (%.1fs/epoch) best_val_acc %.4f"
        % (args.epochs, t_total, t_total / max(args.epochs, 1), best_saver.best),
        flush=True,
    )


if __name__ == "__main__":
    main()
