"""
QKeras 8-bit QAT ResNet-8 (MLPerf Tiny image classification) -- RQ2 Leg C (hls4ml).
FIXED quantizer config (see diag_qkeras_ranges.py evidence).

ROOT CAUSE of the 83% plateau (diagnosed 2026-06-13):
  The residual-add ALIGNMENT quantizers were quantized_bits(8, 2, alpha=1),
  i.e. signed fixed-point with range +/-3.97. But the BN-normalized branch
  outputs (s2_bn2, s3_bn2) have a HUGE dynamic range (max 65.7 / 81.4, p99.9
  15.8 / 23.2). With integer=2 the add operands clipped:
      s2_branch_q  5.50% saturated
      s3_branch_q 24.05% saturated   <-- a QUARTER of the deepest residual
                                          features clamped to the ceiling.
  => the network literally cannot represent its deep features -> train acc
  capped at ~83% (UNDERFIT). The activation quantized_relu(8,2) layers were
  fine (<3% sat) -- the throttle was the add-operand integer-bit count.

FIX (8-bit kept; hls4ml-ingestable; topology IDENTICAL; alpha po2-friendly):
  * add-operand quantizers: quantized_bits(8, ADD_INT, alpha=1) with ADD_INT
    large enough that the p99.9 of the branch/skip is NOT clipped. integer=5
    -> +/-31.75 covers s2 (15.8) and s3 (23.2) with margin; resolution 0.25.
    Both add operands share ONE fixed-point format (common grid) -> hls4ml add
    stays exact, alpha=1 stays po2.
  * post-add activations quantized_relu(8, RELU_OUT_INT): the add can now reach
    ~16-24, so the relu that consumes it needs headroom too. integer=4
    -> [0,16) covers it. The intermediate relus (post-BN, magnitudes < 4)
    stay quantized_relu(8,2) -- they were never the problem (<0.1% sat).
  Net: 8 weight bits, 8 act bits everywhere; only the integer-bit SPLIT and the
  add-operand range were wrong.

Defaults below encode the proven fix; the original is recoverable by passing
the original integer-bit args.
"""

import numpy as np
from tensorflow.keras.layers import (
    Activation, Add, AveragePooling2D, BatchNormalization, Flatten, Input,
)
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2

from qkeras import QActivation, QConv2D, QDense
from qkeras.quantizers import quantized_bits, quantized_relu

EXPECTED_PARAMS = 78666


def build_resnet8_qkeras(
    num_filters=16,
    total_bits=8,
    weight_int_bits=0,
    act_int_bits=2,        # intermediate post-BN relus: [0,4) -- fine, unchanged
    alpha=1,
    add_int_bits=5,        # FIX: residual-add operands +/-31.75 (was 2 -> +/-3.97)
    add_relu_int_bits=4,   # FIX: post-add relus [0,16) (was 2 -> [0,4))
    final_activation=True,
):
    def kq():
        return quantized_bits(total_bits, weight_int_bits, alpha=alpha)

    def aq():
        # intermediate (post-BN, pre-residual) relu -- magnitudes < 4, unchanged
        return quantized_relu(total_bits, act_int_bits)

    def aq_out():
        # post-residual-add relu -- needs headroom for the widened add
        return quantized_relu(total_bits, add_relu_int_bits)

    def addq():
        # signed fixed-point alignment quantizer for adder operands (WIDENED)
        return quantized_bits(total_bits, add_int_bits, alpha=alpha)

    def qconv(filters, ksize, strides, name):
        return QConv2D(
            filters, kernel_size=ksize, strides=strides, padding="same",
            kernel_initializer="he_normal", kernel_regularizer=l2(1e-4),
            kernel_quantizer=kq(), bias_quantizer=kq(), name=name,
        )

    inputs = Input(shape=(32, 32, 3), name="in_image")

    nf = num_filters
    x = qconv(nf, 3, 1, "stem_conv")(inputs)
    x = BatchNormalization(name="stem_bn")(x)
    x = QActivation(aq(), name="stem_relu")(x)

    # ---- stack 1: identity skip -------------------------------------------
    y = qconv(nf, 3, 1, "s1_conv1")(x)
    y = BatchNormalization(name="s1_bn1")(y)
    y = QActivation(aq(), name="s1_relu1")(y)
    y = qconv(nf, 3, 1, "s1_conv2")(y)
    y = BatchNormalization(name="s1_bn2")(y)
    y = QActivation(addq(), name="s1_branch_q")(y)
    x = Add(name="s1_add")([x, y])
    x = QActivation(aq_out(), name="s1_relu_out")(x)

    # ---- stack 2: projection skip (1x1 s2, NO BN) -------------------------
    nf = num_filters * 2
    y = qconv(nf, 3, 2, "s2_conv1")(x)
    y = BatchNormalization(name="s2_bn1")(y)
    y = QActivation(aq(), name="s2_relu1")(y)
    y = qconv(nf, 3, 1, "s2_conv2")(y)
    y = BatchNormalization(name="s2_bn2")(y)
    y = QActivation(addq(), name="s2_branch_q")(y)
    x = qconv(nf, 1, 2, "s2_proj")(x)
    x = QActivation(addq(), name="s2_proj_q")(x)
    x = Add(name="s2_add")([x, y])
    x = QActivation(aq_out(), name="s2_relu_out")(x)

    # ---- stack 3: projection skip (1x1 s2, NO BN) -------------------------
    nf = num_filters * 4
    y = qconv(nf, 3, 2, "s3_conv1")(x)
    y = BatchNormalization(name="s3_bn1")(y)
    y = QActivation(aq(), name="s3_relu1")(y)
    y = qconv(nf, 3, 1, "s3_conv2")(y)
    y = BatchNormalization(name="s3_bn2")(y)
    y = QActivation(addq(), name="s3_branch_q")(y)
    x = qconv(nf, 1, 2, "s3_proj")(x)
    x = QActivation(addq(), name="s3_proj_q")(x)
    x = Add(name="s3_add")([x, y])
    x = QActivation(aq_out(), name="s3_relu_out")(x)

    # ---- tail: GAP 8x8 -> Dense 10 ----------------------------------------
    pool_size = int(np.amin([int(d) for d in x.shape[1:3]]))
    x = AveragePooling2D(pool_size=pool_size, name="avg_pool")(x)
    x = Flatten(name="flatten")(x)
    x = QDense(
        10, kernel_initializer="he_normal",
        kernel_quantizer=kq(), bias_quantizer=kq(), name="dense",
    )(x)
    if final_activation:
        x = Activation("softmax", name="softmax")(x)

    return Model(inputs=inputs, outputs=x, name="resnet8_qkeras8")


def assert_param_contract(model):
    n = int(model.count_params())
    if n != EXPECTED_PARAMS:
        raise AssertionError(
            "topology contract violated: %d params, expected %d" % (n, EXPECTED_PARAMS))
    return n


def load_qkeras_h5(path):
    try:
        from qkeras.utils import load_qmodel
        return load_qmodel(path)
    except Exception:
        from qkeras.utils import _add_supported_quantized_objects
        from tensorflow.keras.models import load_model
        co = {}
        _add_supported_quantized_objects(co)
        return load_model(path, custom_objects=co)


if __name__ == "__main__":
    m = build_resnet8_qkeras()
    m.summary()
    print("param contract OK:", assert_param_contract(m))
