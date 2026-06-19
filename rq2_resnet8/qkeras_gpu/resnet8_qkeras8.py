"""
QKeras 8-bit QAT ResNet-8 (MLPerf Tiny image classification) -- RQ2 Leg C (hls4ml).

TOPOLOGY CONTRACT (must match the verified reference EXACTLY -- thesis fairness invariant):
  stem   : Conv3x3-16 -> BN -> ReLU
  stack1 : {Conv3x3-16 BN ReLU, Conv3x3-16 BN} + identity skip -> add -> ReLU
  stack2 : {Conv3x3-32 s2 BN ReLU, Conv3x3-32 BN} + Conv1x1-32 s2 projection skip -> add -> ReLU
  stack3 : {Conv3x3-64 s2 BN ReLU, Conv3x3-64 BN} + Conv1x1-64 s2 projection skip -> add -> ReLU
  tail   : AveragePooling2D(8x8) -> Flatten -> Dense 10 (+ softmax in TRAINING only)

Matches mlcommons/tiny benchmark/training/image_classification/keras_model.py
resnet_v1_eembc with filters 16/32/64: conv biases kept, l2(1e-4) on conv kernels,
he_normal init, NO BatchNorm on the two 1x1 projection convs (that is what pins the
reference parameter count). Total params = 78,666 (asserted below).

QUANTIZATION (mirrors the official MLPerf Tiny v1.0 open/hls4ml RN08 submission,
mlcommons/tiny_results_v1.0 open/hls4ml/code/ic/RN08/training/resnet_v1_eembc.py,
widened from 7 to 8 total bits per the leg spec):
  kernels/biases : quantized_bits(8, 0, alpha=1)   # po2-friendly alpha=1, hls4ml-safest
  activations    : quantized_relu(8, 2)            # official RN08 used (7,2); (8,2) keeps
                                                   # the same dynamic range [0,4) +1 frac bit
  adder inputs   : QActivation(quantized_bits(8, 2, alpha=1)) on the conv branch (post-BN)
                   and on the projection-skip output, so both add operands have a declared
                   fixed-point format (same trick as the official submission)
  BatchNorm      : KEPT as float layers (hls4ml folds/implements them at conversion)

Working env combo (verified 2026-06-12, /root/rq2_venv in WSL Ubuntu):
  tensorflow-cpu 2.15.1 + keras 2.15.0 + qkeras 0.9.0 + pyparsing 3.1.4
  + tf-keras 2.15.1 (--no-deps, needed by tensorflow-model-optimization)
  + hls4ml 1.3.0 + protobuf 3.20.3
  NOTE: a bare `pip install qkeras hls4ml` PULLS IN tensorflow 2.21 + keras 3 and
  BREAKS the venv -- see training/qkeras/requirements-pin.txt for the repair pin set.
"""

import numpy as np
from tensorflow.keras.layers import (
    Activation,
    Add,
    AveragePooling2D,
    BatchNormalization,
    Flatten,
    Input,
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
    act_int_bits=2,
    alpha=1,
    final_activation=True,
):
    """Build the QKeras QAT ResNet-8.

    final_activation=True  -> training model (softmax head).
    final_activation=False -> export model for hls4ml (softmax stripped); same
                              weighted layers in the same order, so
                              nosm.set_weights(trained.get_weights()) is exact.
    """

    def kq():
        return quantized_bits(total_bits, weight_int_bits, alpha=alpha)

    def aq():
        return quantized_relu(total_bits, act_int_bits)

    def addq():
        # signed fixed-point alignment quantizer for adder operands
        return quantized_bits(total_bits, act_int_bits, alpha=alpha)

    def qconv(filters, ksize, strides, name):
        return QConv2D(
            filters,
            kernel_size=ksize,
            strides=strides,
            padding="same",
            kernel_initializer="he_normal",
            kernel_regularizer=l2(1e-4),
            kernel_quantizer=kq(),
            bias_quantizer=kq(),
            name=name,
        )

    # NB: hls4ml ModelGraph reserves the layer name "input" -- do not use it.
    inputs = Input(shape=(32, 32, 3), name="in_image")

    # ---- stem -------------------------------------------------------------
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
    # identity path is already on the quantized_relu grid -> add directly
    x = Add(name="s1_add")([x, y])
    x = QActivation(aq(), name="s1_relu_out")(x)

    # ---- stack 2: projection skip (1x1 s2, NO BN -- reference-exact) ------
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
    x = QActivation(aq(), name="s2_relu_out")(x)

    # ---- stack 3: projection skip (1x1 s2, NO BN -- reference-exact) ------
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
    x = QActivation(aq(), name="s3_relu_out")(x)

    # ---- tail: GAP 8x8 -> Dense 10 ----------------------------------------
    pool_size = int(np.amin([int(d) for d in x.shape[1:3]]))  # = 8
    x = AveragePooling2D(pool_size=pool_size, name="avg_pool")(x)
    x = Flatten(name="flatten")(x)
    x = QDense(
        10,
        kernel_initializer="he_normal",
        kernel_quantizer=kq(),
        bias_quantizer=kq(),
        name="dense",
    )(x)
    if final_activation:
        x = Activation("softmax", name="softmax")(x)

    model = Model(inputs=inputs, outputs=x, name="resnet8_qkeras8")
    return model


def assert_param_contract(model):
    n = int(model.count_params())
    if n != EXPECTED_PARAMS:
        raise AssertionError(
            "topology contract violated: %d params, expected %d" % (n, EXPECTED_PARAMS)
        )
    return n


def load_qkeras_h5(path):
    """Load a saved QKeras .h5 with the proper custom objects."""
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
