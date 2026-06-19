#!/usr/bin/env python
"""GPU train of the FIXED 8-bit QKeras ResNet-8 (CIFAR-10) -- RQ2 Leg C.

Identical recipe to train_resnet8_qkeras_gpu.py (cosine-to-0 LR, pad-4 crop +
h-flip aug, batch 128, X/256 input) EXCEPT the model is imported from
resnet8_qkeras8_fixed.py (widened residual-add quantizers -- the proven fix).

--mode fixed  : the fixed QKeras model (default)
--mode orig   : original quantizers (control, for the smoke comparison)
--mode float  : non-quantized twin (sanity: should hit >90% train fast)
"""
import argparse, json, math, os, random, sys, time
import numpy as np

DEFAULT_DATA_DIR = "/mnt/d/RTL_LLM_CLAUDE/rq2_resnet8/data/cifar-10-batches-py"
FIXED_MODEL_DIR = "/mnt/d/RTL_LLM_CLAUDE/rq2_resnet8"
ORIG_MODEL_DIR = "/root/rq2_training/qkeras"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    p.add_argument("--out-dir", default="/root/rq2_training/qkeras_gpu_fixed")
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--warmup-epochs", type=int, default=5)
    p.add_argument("--min-lr", type=float, default=0.0)
    p.add_argument("--ckpt-every", type=int, default=50)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--mode", choices=["fixed", "orig", "float"], default="fixed")
    return p.parse_args()


def load_cifar10(data_dir):
    import pickle
    def unpickle(f):
        with open(f, "rb") as fo:
            return pickle.load(fo, encoding="bytes")
    xs, ys = [], []
    for i in range(1, 6):
        d = unpickle(os.path.join(data_dir, "data_batch_%d" % i))
        xs.append(d[b"data"]); ys += list(d[b"labels"])
    x_train = np.vstack(xs).reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    y_train = np.array(ys)
    d = unpickle(os.path.join(data_dir, "test_batch"))
    x_test = d[b"data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    y_test = np.array(d[b"labels"])
    return x_train, y_train, x_test, y_test


def build_float_resnet8(num_filters=16):
    """Non-quantized twin (same topology) -- sanity that float reaches >90% fast."""
    from tensorflow.keras.layers import (
        Activation, Add, AveragePooling2D, BatchNormalization, Conv2D, Dense,
        Flatten, Input)
    from tensorflow.keras.models import Model
    from tensorflow.keras.regularizers import l2
    def conv(f, k, s, n):
        return Conv2D(f, k, strides=s, padding="same", kernel_initializer="he_normal",
                      kernel_regularizer=l2(1e-4), name=n)
    inp = Input((32, 32, 3), name="in_image")
    nf = num_filters
    x = Activation("relu")(BatchNormalization()(conv(nf, 3, 1, "stem")(inp)))
    y = Activation("relu")(BatchNormalization()(conv(nf, 3, 1, "s1c1")(x)))
    y = BatchNormalization()(conv(nf, 3, 1, "s1c2")(y))
    x = Activation("relu")(Add()([x, y]))
    nf *= 2
    y = Activation("relu")(BatchNormalization()(conv(nf, 3, 2, "s2c1")(x)))
    y = BatchNormalization()(conv(nf, 3, 1, "s2c2")(y))
    x = conv(nf, 1, 2, "s2p")(x)
    x = Activation("relu")(Add()([x, y]))
    nf *= 2
    y = Activation("relu")(BatchNormalization()(conv(nf, 3, 2, "s3c1")(x)))
    y = BatchNormalization()(conv(nf, 3, 1, "s3c2")(y))
    x = conv(nf, 1, 2, "s3p")(x)
    x = Activation("relu")(Add()([x, y]))
    p = int(np.amin([int(d) for d in x.shape[1:3]]))
    x = Flatten()(AveragePooling2D(p)(x))
    x = Dense(10, kernel_initializer="he_normal", name="dense")(x)
    x = Activation("softmax")(x)
    return Model(inp, x, name="resnet8_float")


def main():
    args = parse_args()
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    for v in ("OMP_NUM_THREADS", "TF_NUM_INTRAOP_THREADS", "TF_NUM_INTEROP_THREADS"):
        os.environ.setdefault(v, str(args.threads))
    import tensorflow as tf
    print("[gpu] visible:", tf.config.list_physical_devices("GPU"), flush=True)
    for g in tf.config.list_physical_devices("GPU"):
        try: tf.config.experimental.set_memory_growth(g, True)
        except Exception: pass
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    from tensorflow.keras.callbacks import Callback, CSVLogger, LearningRateScheduler
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from tensorflow.keras.utils import to_categorical

    random.seed(args.seed); np.random.seed(args.seed); tf.random.set_seed(args.seed)
    out_dir = args.out_dir
    ckpt_dir = os.path.join(out_dir, "ckpt"); os.makedirs(ckpt_dir, exist_ok=True)
    print("[cfg] %s" % json.dumps(vars(args)), flush=True)

    x_train, y_train, x_test, y_test = load_cifar10(args.data_dir)
    x_train = (x_train / 256.0).astype("float32")
    x_test = (x_test / 256.0).astype("float32")
    y_train = to_categorical(y_train, 10); y_test = to_categorical(y_test, 10)
    print("[data] train %s test %s" % (x_train.shape, x_test.shape), flush=True)

    builder_kwargs = dict(num_filters=16, total_bits=8, weight_int_bits=0,
                          act_int_bits=2, alpha=1)
    if args.mode == "float":
        model = build_float_resnet8(16)
        nparams = int(model.count_params())
        print("[model] FLOAT twin params=%d" % nparams, flush=True)
        build_nosm = None
    else:
        if args.mode == "fixed":
            sys.path.insert(0, FIXED_MODEL_DIR)
            from resnet8_qkeras8_fixed import assert_param_contract, build_resnet8_qkeras
        else:  # orig
            sys.path.insert(0, ORIG_MODEL_DIR)
            from resnet8_qkeras8 import assert_param_contract, build_resnet8_qkeras
        model = build_resnet8_qkeras(final_activation=True, **builder_kwargs)
        nparams = assert_param_contract(model)
        print("[model] %s params=%d (contract OK)" % (args.mode, nparams), flush=True)
        build_nosm = lambda: build_resnet8_qkeras(final_activation=False, **builder_kwargs)

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr),
                  loss="categorical_crossentropy", metrics=["accuracy"])

    datagen = ImageDataGenerator(width_shift_range=4, height_shift_range=4,
                                 fill_mode="reflect", horizontal_flip=True)
    datagen.fit(x_train)

    base_lr, warm, total, min_lr = args.lr, args.warmup_epochs, args.epochs, args.min_lr
    def lr_schedule(epoch):
        if warm > 0 and epoch < warm:
            return float(base_lr * (0.1 + 0.9 * (epoch + 1) / float(warm)))
        t = min(max((epoch - warm) / float(max(1, total - warm)), 0.0), 1.0)
        return float(min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * t)))

    class EpochTimer(Callback):
        def on_epoch_begin(self, e, logs=None): self.t0 = time.time()
        def on_epoch_end(self, e, logs=None):
            print("[time] epoch %d wall %.1fs lr %.6g" % (e+1, time.time()-self.t0, lr_schedule(e)), flush=True)

    class PeriodicCheckpoint(Callback):
        def on_epoch_end(self, e, logs=None):
            if (e+1) % args.ckpt_every == 0:
                p = os.path.join(ckpt_dir, "epoch_%03d.h5" % (e+1))
                self.model.save(p); print("[ckpt] saved %s" % p, flush=True)

    class BestSaver(Callback):
        def __init__(self): super().__init__(); self.best=-1.0; self.best_epoch=-1
        def on_epoch_end(self, e, logs=None):
            va = (logs or {}).get("val_accuracy")
            if va is None or va <= self.best: return
            self.best=float(va); self.best_epoch=int(e+1)
            bp = os.path.join(out_dir, "resnet8_qkeras8_best.h5"); self.model.save(bp)
            nosm_path = None
            if build_nosm is not None:
                nosm = build_nosm(); nosm.set_weights(self.model.get_weights())
                nosm_path = os.path.join(out_dir, "resnet8_qkeras8_best_nosoftmax.h5")
                nosm.save(nosm_path)
            with open(os.path.join(out_dir, "best.json"), "w") as f:
                json.dump({"epoch": self.best_epoch, "val_accuracy": self.best,
                           "mode": args.mode, "best_h5": bp,
                           "best_nosoftmax_h5": nosm_path}, f, indent=2)
            print("[best] epoch %d val_acc %.4f -> saved%s" %
                  (self.best_epoch, self.best, " (+nosoftmax)" if nosm_path else ""), flush=True)

    best_saver = BestSaver()
    callbacks = [LearningRateScheduler(lr_schedule), EpochTimer(),
                 PeriodicCheckpoint(), best_saver,
                 CSVLogger(os.path.join(out_dir, "train_history.csv"), append=True)]
    steps = int(np.ceil(len(x_train) / float(args.batch_size)))
    t0 = time.time()
    model.fit(datagen.flow(x_train, y_train, batch_size=args.batch_size),
              steps_per_epoch=steps, epochs=args.epochs,
              validation_data=(x_test, y_test), validation_freq=1,
              callbacks=callbacks, verbose=2)
    model.save(os.path.join(out_dir, "resnet8_qkeras8_last.h5"))
    loss, acc = model.evaluate(x_test, y_test, verbose=0)
    print("[final] %.1fs best_val=%.4f@ep%d last_val=%.4f" %
          (time.time()-t0, best_saver.best, best_saver.best_epoch, acc), flush=True)


if __name__ == "__main__":
    main()
