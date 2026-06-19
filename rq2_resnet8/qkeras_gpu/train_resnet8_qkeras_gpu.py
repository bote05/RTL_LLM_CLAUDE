#!/usr/bin/env python
"""
GPU retrain of the QKeras 8-bit QAT ResNet-8 on CIFAR-10 -- RQ2 Leg C (hls4ml).

WHY THIS REWRITE (diagnosing the prior 82.67% gap)
--------------------------------------------------
The prior CPU run (/root/rq2_training/qkeras, best.json epoch 200) plateaued at
val_acc 82.67% with train_acc only ~84% -> the net was UNDER-FIT, not overfit.
Three recipe defects vs the proven Brevitas W4A4 winner (86.68% on the SAME
topology at HALF the bits) and the official MLPerf Tiny image_classification
reference (~87% float):

  1. LR SCHEDULE (the big one): prior used exponential 0.001 * 0.99^epoch, which
     by epoch 200 is still ~1.3e-4 and NEVER anneals near zero -> the net keeps
     wobbling and cannot settle into a sharp minimum. FIX: cosine annealing to 0
     over the full run (exactly what drove the Brevitas net's train_acc to 93.3%).
  2. AUGMENTATION: prior used rotation_range=15 + shift 0.1 on 32x32 tiles, which
     warps these tiny images and adds noise. FIX: the canonical CIFAR-10 / bnn_pynq
     augmentation = RandomCrop(32, padding=4 reflect) + RandomHorizontalFlip ONLY
     (the Brevitas winner used exactly this).
  3. BATCH SIZE: prior used 32 (noisy, slow). FIX: 128 (Brevitas winner + MLPerf).

Also fixed: prior evaluated only every 10 epochs (validation_freq=10) so the true
best could be missed; here we eval EVERY epoch on GPU (cheap) and save best.

KEPT IDENTICAL (fairness invariant + reproducibility):
  * topology: build_resnet8_qkeras from the original resnet8_qkeras8.py (78,666
    params, contract asserted) -- byte-identical model graph.
  * quantization: quantized_bits(8,0,alpha=1) weights, quantized_relu(8,2) acts.
  * input scaling: X/256.0 (clean 8-bit fixed-point for hls4ml; same as prior).
  * softmax stripped for the hls4ml export copy.
  * Adam optimizer base lr 1e-3, categorical cross-entropy.

GPU: tensorflow[and-cuda]==2.15.1 in /root/rq2_gpu_venv (RTX 4060 bound, verified).
Threads capped (host/WSL neighbors) but on GPU the CPU only feeds data.
"""

import argparse
import json
import math
import os
import random
import sys
import time

import numpy as np

DEFAULT_DATA_DIR = "/mnt/d/RTL_LLM_CLAUDE/rq2_resnet8/data/cifar-10-batches-py"
DEFAULT_OUT_DIR = "/root/rq2_training/qkeras_gpu"
# original model definition (topology + quant contract) lives here:
ORIG_MODEL_DIR = "/root/rq2_training/qkeras"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    p.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--warmup-epochs", type=int, default=5,
                   help="linear LR warmup from lr/10 to lr before cosine")
    p.add_argument("--min-lr", type=float, default=0.0,
                   help="cosine floor (0 = anneal fully to zero)")
    p.add_argument("--ckpt-every", type=int, default=50)
    p.add_argument("--threads", type=int, default=6,
                   help="CPU op threads (GPU does the math; CPU only feeds data)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch-log-every", type=int, default=0,
                   help="print running loss every N batches (0 = off; verbose=2 prints per-epoch)")
    return p.parse_args()


def load_cifar10(data_dir):
    """Load CIFAR-10 python batches exactly like the official train.py."""
    import pickle

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

    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    os.environ.setdefault("OMP_NUM_THREADS", str(args.threads))
    os.environ.setdefault("TF_NUM_INTRAOP_THREADS", str(args.threads))
    os.environ.setdefault("TF_NUM_INTEROP_THREADS", str(args.threads))
    import tensorflow as tf

    gpus = tf.config.list_physical_devices("GPU")
    print("[gpu] visible devices: %s" % str(gpus), flush=True)
    for g in gpus:
        try:
            tf.config.experimental.set_memory_growth(g, True)
        except Exception as e:
            print("[gpu] memory_growth note: %s" % e, flush=True)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)

    from tensorflow.keras.callbacks import Callback, CSVLogger, LearningRateScheduler
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from tensorflow.keras.utils import to_categorical

    # import the ORIGINAL model definition (topology + quant contract preserved)
    sys.path.insert(0, ORIG_MODEL_DIR)
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
    x_train = (x_train / 256.0).astype("float32")
    x_test = (x_test / 256.0).astype("float32")
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)
    print("[data] train %s test %s" % (str(x_train.shape), str(x_test.shape)), flush=True)

    # ---- model -------------------------------------------------------------
    builder_kwargs = dict(
        num_filters=16, total_bits=8, weight_int_bits=0, act_int_bits=2, alpha=1
    )
    model = build_resnet8_qkeras(final_activation=True, **builder_kwargs)
    n_params = assert_param_contract(model)
    print("[model] resnet8_qkeras8 params=%d (contract OK)" % n_params, flush=True)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    # ---- canonical CIFAR-10 augmentation (RandomCrop pad-4 reflect + h-flip) -
    # ImageDataGenerator with width/height_shift_range as INTEGER pixels =
    # random translation, fill_mode="reflect" ~= RandomCrop(32, padding=4).
    datagen = ImageDataGenerator(
        width_shift_range=4,        # pixels (== padding=4 crop jitter)
        height_shift_range=4,       # pixels
        fill_mode="reflect",
        horizontal_flip=True,
    )
    datagen.fit(x_train)

    # ---- warmup + cosine-to-zero LR schedule --------------------------------
    base_lr = args.lr
    warm = args.warmup_epochs
    total = args.epochs
    min_lr = args.min_lr

    def lr_schedule(epoch):
        if warm > 0 and epoch < warm:
            # linear warmup from base_lr/10 to base_lr
            lr = base_lr * (0.1 + 0.9 * (epoch + 1) / float(warm))
        else:
            t = (epoch - warm) / float(max(1, total - warm))
            t = min(max(t, 0.0), 1.0)
            lr = min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * t))
        return float(lr)

    class EpochTimer(Callback):
        def on_epoch_begin(self, epoch, logs=None):
            self.t0 = time.time()

        def on_epoch_end(self, epoch, logs=None):
            print("[time] epoch %d wall %.1fs lr %.6g"
                  % (epoch + 1, time.time() - self.t0, lr_schedule(epoch)), flush=True)

    class PeriodicCheckpoint(Callback):
        def on_epoch_end(self, epoch, logs=None):
            if (epoch + 1) % args.ckpt_every == 0:
                path = os.path.join(ckpt_dir, "epoch_%03d.h5" % (epoch + 1))
                self.model.save(path)
                print("[ckpt] saved %s" % path, flush=True)

    class BestSaver(Callback):
        def __init__(self):
            super().__init__()
            self.best = -1.0
            self.best_epoch = -1

        def on_epoch_end(self, epoch, logs=None):
            va = (logs or {}).get("val_accuracy")
            if va is None:
                return
            if va > self.best:
                self.best = float(va)
                self.best_epoch = int(epoch + 1)
                best_path = os.path.join(out_dir, "resnet8_qkeras8_best.h5")
                self.model.save(best_path)
                # strip-softmax copy for hls4ml ingest (exact weight transfer)
                nosm = build_resnet8_qkeras(final_activation=False, **builder_kwargs)
                nosm.set_weights(self.model.get_weights())
                nosm_path = os.path.join(out_dir, "resnet8_qkeras8_best_nosoftmax.h5")
                nosm.save(nosm_path)
                with open(os.path.join(out_dir, "best.json"), "w") as f:
                    json.dump({
                        "epoch": self.best_epoch,
                        "val_accuracy": self.best,
                        "best_h5": best_path,
                        "best_nosoftmax_h5": nosm_path,
                    }, f, indent=2)
                print("[best] epoch %d val_acc %.4f -> saved (+nosoftmax)"
                      % (self.best_epoch, self.best), flush=True)

    best_saver = BestSaver()
    callbacks = [
        LearningRateScheduler(lr_schedule),
        EpochTimer(),
        PeriodicCheckpoint(),
        best_saver,
        CSVLogger(os.path.join(out_dir, "train_history.csv"), append=True),
    ]

    steps = int(np.ceil(len(x_train) / float(args.batch_size)))
    t_start = time.time()
    model.fit(
        datagen.flow(x_train, y_train, batch_size=args.batch_size),
        steps_per_epoch=steps,
        epochs=args.epochs,
        validation_data=(x_test, y_test),
        validation_freq=1,            # eval EVERY epoch (cheap on GPU)
        callbacks=callbacks,
        verbose=2,
    )
    t_total = time.time() - t_start

    last_path = os.path.join(out_dir, "resnet8_qkeras8_last.h5")
    model.save(last_path)
    loss, acc = model.evaluate(x_test, y_test, verbose=0)
    print("[final] test loss %.4f acc %.4f" % (loss, acc), flush=True)
    print("[done] %d epochs in %.1fs (%.1fs/epoch) best_val_acc %.4f @epoch %d"
          % (args.epochs, t_total, t_total / max(args.epochs, 1),
             best_saver.best, best_saver.best_epoch), flush=True)


if __name__ == "__main__":
    main()
