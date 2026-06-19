# FINN ZCU104 build: ResNet-8 (MLPerf Tiny), Brevitas W4A4, FINN v0.10.1 bare-metal.
# Env (FINN_ROOT, FINN_BUILD_DIR, Xilinx settings, PYTHONPATH) is set by launch_resnet8.sh.
#
# Residual handling: mainline FINN's DEFAULT build steps OMIT the transforms needed
# to streamline + map skip connections. We therefore replace step_streamline and
# step_convert_to_hw with CUSTOM steps cloned from the finn-examples resnet50
# custom_steps.py recipe:
#   - streamline_linear  : ~21 reorder/absorb/collapse transforms
#   - streamline_nonlinear: MoveLinearPastEltwiseAdd + MoveLinearPastFork (the
#                           residual-specific movers), iterated to a fixed point
#   - convert_to_hw       : default infers + InferAddStreamsLayer (the elementwise
#                           residual add) + InferDuplicateStreamsLayer (the fork that
#                           feeds both the main branch and the skip).
# The Brevitas model already carries SHARED residual-add quantizers + same-sign ReLUs,
# so the #1090 "Scaling factors are different" join killer is pre-handled in the model.
# We complete ALL streamlining BEFORE CreateDataflowPartition to avoid the
# cycle-free-graph / disjoint-partition errors. Auto FIFO sizing guards branch deadlock.

import os

from onnx import helper

import finn.builder.build_dataflow as build
import finn.builder.build_dataflow_config as build_cfg
from finn.builder.build_dataflow_config import DataflowBuildConfig, VerificationStepType
from finn.builder.build_dataflow_steps import verify_step

# --- transforms ---
import finn.transformation.streamline.absorb as absorb
import finn.transformation.streamline.reorder as reorder
import finn.transformation.fpgadataflow.convert_to_hw_layers as to_hw
from finn.transformation.streamline import Streamline
from finn.transformation.streamline.round_thresholds import RoundAndClipThresholds
from finn.transformation.streamline.collapse_repeated import (
    CollapseRepeatedAdd,
    CollapseRepeatedMul,
)
from finn.transformation.move_reshape import RemoveCNVtoFCFlatten
from finn.transformation.streamline.reorder import (
    MoveAddPastMul,
    MoveScalarMulPastMatMul,
    MoveMulPastMaxPool,
    MoveScalarAddPastMatMul,
    MoveAddPastConv,
    MoveScalarMulPastConv,
    MoveScalarLinearPastInvariants,
    MoveLinearPastEltwiseAdd,
    MoveLinearPastFork,
    MoveMulPastFork,
    MoveAddPastFork,
    MakeMaxPoolNHWC,
)
from qonnx.transformation.general import SortGraph
from finn.transformation.streamline.absorb import (
    AbsorbAddIntoMultiThreshold,
    AbsorbMulIntoMultiThreshold,
    AbsorbSignBiasIntoMultiThreshold,
    AbsorbTransposeIntoMultiThreshold,
    AbsorbConsecutiveTransposes,
    Absorb1BitMulIntoMatMul,
    Absorb1BitMulIntoConv,
    FactorOutMulSignMagnitude,
    AbsorbScalarMulAddIntoTopK,
)
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.batchnorm_to_affine import BatchNormToAffine
from qonnx.transformation.general import (
    ConvertDivToMul,
    ConvertSubToAdd,
    GiveReadableTensorNames,
    GiveUniqueNodeNames,
    RemoveUnusedTensors,
    RemoveStaticGraphInputs,
)
from qonnx.transformation.fold_constants import FoldConstants
from qonnx.transformation.infer_data_layouts import InferDataLayouts
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.transformation.lower_convs_to_matmul import LowerConvsToMatMul
from qonnx.transformation.remove import RemoveIdentityOps
from qonnx.util.cleanup import cleanup_model

MODEL_FILE = "/root/rq2_training/brevitas/resnet8_w4a4.qonnx"
OUTPUT_DIR = "/root/rq2_training/finn_resnet8/out_resnet8_zcu104"


# ---------------------------------------------------------------------------
# Custom residual-aware streamlining (finn-examples resnet50 recipe, adapted)
# ---------------------------------------------------------------------------
def step_resnet8_streamline_linear(model: ModelWrapper, cfg: DataflowBuildConfig):
    streamline_transformations = [
        ConvertSubToAdd(),
        ConvertDivToMul(),
        BatchNormToAffine(),
        AbsorbSignBiasIntoMultiThreshold(),
        MoveScalarLinearPastInvariants(),
        MoveAddPastMul(),
        MoveScalarAddPastMatMul(),
        MoveAddPastConv(),
        MoveScalarMulPastMatMul(),
        MoveScalarMulPastConv(),
        MoveAddPastMul(),
        CollapseRepeatedAdd(),
        CollapseRepeatedMul(),
        AbsorbAddIntoMultiThreshold(),
        FactorOutMulSignMagnitude(),
        AbsorbMulIntoMultiThreshold(),
        Absorb1BitMulIntoMatMul(),
        Absorb1BitMulIntoConv(),
        RoundAndClipThresholds(),
    ]
    for trn in streamline_transformations:
        model = model.transform(trn)
        model = model.transform(GiveUniqueNodeNames())
        model = model.transform(GiveReadableTensorNames())
        model = model.transform(InferDataTypes())
    return model


def step_resnet8_streamline_nonlinear(model: ModelWrapper, cfg: DataflowBuildConfig):
    # the residual-specific movers. CRITICAL ORDERING (verified interactively):
    # the fork-resolving movers (MoveMulPastFork / MoveAddPastFork) MUST run
    # BEFORE MoveLinearPastEltwiseAdd. The skip connection makes the per-tensor
    # scale a FORKED tensor (it feeds both the main branch and the identity skip);
    # MoveLinearPastEltwiseAdd's move_node rewires/removes the matched Mul in place
    # and ORPHANS the stem if the Mul still sits on a fork (observed: stem input
    # became a dangling tensor, 3 nodes silently dropped). Duplicating the linear
    # op across the fork first gives each consumer its own copy, after which
    # MoveLinearPastEltwiseAdd fires cleanly (0 dangling tensors). Iterate to a
    # fixed point.
    for _ in range(4):
        model = model.transform(MoveMulPastFork())
        model = model.transform(MoveAddPastFork())
        model = model.transform(MoveLinearPastFork())
        model = model.transform(SortGraph())
        model = model.transform(InferShapes())
        model = model.transform(InferDataTypes())
        model = model.transform(GiveUniqueNodeNames())
        model = model.transform(GiveReadableTensorNames())
        model = model.transform(MoveLinearPastEltwiseAdd())
        model = model.transform(MoveScalarLinearPastInvariants())
        model = model.transform(SortGraph())
        model = model.transform(InferShapes())
        model = model.transform(InferDataTypes())
        model = model.transform(GiveUniqueNodeNames())
        model = model.transform(GiveReadableTensorNames())
    return model


def _quantavgpool_to_globalavgpool(model):
    """The trained model's GAP is a QONNX QuantAvgPool2d (kernel==stride==spatial=8,
    i.e. a true global average pool). FINN's InferGlobalAccPoolLayer only matches the
    plain GlobalAveragePool op (the QuantAvgPool2d->InferPool path instead emits an
    Im2Col+Pool with extra transposes that wedge the partition). Rewrite it to a plain
    GlobalAveragePool so it maps to a clean GlobalAccPool HW node + scalar Mul."""
    g = model.graph
    for node in list(g.node):
        if node.op_type != "QuantAvgPool2d":
            continue
        attrs = {a.name: a for a in node.attribute}
        ishape = model.get_tensor_shape(node.input[0])  # NCHW
        k = attrs["kernel"].i
        h, w = ishape[2], ishape[3]
        assert k == h == w, "GAP is not global: kernel=%d HxW=%dx%d" % (k, h, w)
        new = helper.make_node(
            "GlobalAveragePool", [node.input[0]], [node.output[0]],
            name="GlobalAveragePool_" + node.name)
        idx = list(g.node).index(node)
        g.node.insert(idx, new)
        g.node.remove(node)
    model = model.transform(InferShapes())
    model = model.transform(InferDataTypes())
    return model


def step_resnet8_streamline(model: ModelWrapper, cfg: DataflowBuildConfig):
    model = step_resnet8_streamline_linear(model, cfg)
    model = step_resnet8_streamline_nonlinear(model, cfg)
    # collapse any newly-adjacent scalar linear ops produced by the movers,
    # then run the standard Streamline pass to mop up.
    model = model.transform(Streamline())
    model = step_resnet8_streamline_linear(model, cfg)
    model = step_resnet8_streamline_nonlinear(model, cfg)
    model = model.transform(Streamline())
    # ---- GAP + dense-tail prep (done HERE, while the dense is still a plain MatMul,
    # so the scalar-mul movers can commute the per-tensor scale past it; MVAU has no
    # such mover). Result tail: ...MultiThreshold -> GlobalAveragePool -> Reshape ->
    # MatMul -> Mul -> Add. The Reshape now feeds MatMul directly, so
    # RemoveCNVtoFCFlatten (in convert_to_hw) absorbs the flatten cleanly. The trailing
    # Mul+Add output dequant stays in the parent graph (pure tail; argmax-invariant). ----
    model = _quantavgpool_to_globalavgpool(model)
    for _ in range(6):
        model = model.transform(MoveScalarLinearPastInvariants())
        model = model.transform(MoveScalarMulPastMatMul())
        model = model.transform(CollapseRepeatedMul())
        model = model.transform(SortGraph())
        model = model.transform(InferShapes())
        model = model.transform(InferDataTypes())
        model = model.transform(GiveUniqueNodeNames())
        model = model.transform(GiveReadableTensorNames())
    model = model.transform(RemoveUnusedTensors())
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(GiveReadableTensorNames())
    if VerificationStepType.STREAMLINED_PYTHON in cfg._resolve_verification_steps():
        verify_step(model, cfg, "streamlined_python", need_parent=False)
    return model


def _hw_cleanup(model):
    model = model.transform(SortGraph())
    model = model.transform(InferShapes())
    model = model.transform(InferDataTypes())
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(GiveReadableTensorNames())
    return model


def _relocate_gap_mul_past_dense(model):
    """InferGlobalAccPoolLayer emits  GlobalAccPool -> Mul(1/(H*W)) -> Transpose ->
    Reshape -> MVAU(dense).  That scalar Mul sits between the flatten and the dense
    MVAU and blocks RemoveCNVtoFCFlatten (which needs Transpose->Reshape->MVAU
    adjacent). The dense is linear, so a scalar applied to its input equals the same
    scalar applied to its output: (s*x).W == s*(x.W). Move the Mul to AFTER the MVAU.
    Then Transpose->Reshape->MVAU is contiguous (flatten absorbs into weights) and the
    relocated Mul collapses with the trailing dequant Mul. Argmax-invariant (s>0)."""
    g = model.graph
    for mul in list(g.node):
        if mul.op_type != "Mul":
            continue
        prod = model.find_producer(mul.input[0])
        if prod is None or prod.op_type != "GlobalAccPool":
            continue
        scale_t = mul.input[1]
        if model.get_initializer(scale_t) is None:
            continue
        gap_out = mul.input[0]
        mul_out = mul.output[0]
        # find the dense MVAU downstream of this Mul (skip Transpose/Reshape)
        cur = mul_out
        chain = []
        while True:
            cons = model.find_consumer(cur)
            if cons is None:
                break
            chain.append(cons)
            if cons.op_type.startswith("MVAU"):
                break
            cur = cons.output[0]
        if not chain or not chain[-1].op_type.startswith("MVAU"):
            continue
        mvau = chain[-1]
        mvau_out = mvau.output[0]
        # bypass the Mul: GlobalAccPool feeds the (Transpose/Reshape ->) MVAU directly
        chain[0].input[0] = gap_out
        # re-point MVAU output through the Mul to whatever consumed mvau_out
        downstream = model.find_consumer(mvau_out)
        new_mvau_t = model.make_new_valueinfo_name()
        mvau.output[0] = new_mvau_t
        mul.input[0] = new_mvau_t
        mul.output[0] = mvau_out
        # leave 'downstream' reading mvau_out (now produced by the relocated Mul)
        _ = downstream
    model = model.transform(InferShapes())
    model = model.transform(InferDataTypes())
    return model


def step_resnet8_convert_to_hw(model: ModelWrapper, cfg: DataflowBuildConfig):
    # Lower convs to MatMul + Im2Col now (so InferConvInpGen sees the
    # Transpose->Im2Col->Transpose sandwich it expects).
    model = model.transform(LowerConvsToMatMul())
    model = model.transform(MakeMaxPoolNHWC())
    model = _hw_cleanup(model)
    # CRITICAL: cancel the NCHW<->NHWC transposes that lowering inserts on EVERY
    # conv BEFORE inferring HW layers. The residual skips fork the activation
    # tensor (it feeds the main conv AND the skip), so the per-branch transposes
    # must be pushed past the (plain-ONNX) fork and the join-add first, then the
    # now-consecutive inverse transposes cancel. Doing this AFTER HW inference is
    # too late: the transposes get wedged between HW nodes (DuplicateStreams /
    # AddStreams), MoveTransposePastFork no longer matches them, and the leftover
    # interleaved Transposes make CreateDataflowPartition raise
    # "cycle-free graph violated: partition depends on itself".
    # Verified: 18 transposes -> 3 structural (stem-input + GAP boundary, which
    # InferConvInpGen / GAP conversion then consume).
    for _ in range(8):
        model = model.transform(reorder.MoveTransposePastFork())
        model = _hw_cleanup(model)
        model = model.transform(reorder.MoveTransposePastJoinAdd())
        model = _hw_cleanup(model)
        model = model.transform(AbsorbConsecutiveTransposes())
        model = _hw_cleanup(model)
        model = model.transform(absorb.AbsorbTransposeIntoMultiThreshold())
        model = _hw_cleanup(model)
    model = model.transform(InferDataLayouts())
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(GiveReadableTensorNames())

    # standard inferences
    if cfg.standalone_thresholds:
        model = model.transform(to_hw.InferThresholdingLayer())
    model = model.transform(to_hw.InferBinaryMatrixVectorActivation())
    model = model.transform(to_hw.InferQuantizedMatrixVectorActivation())
    model = model.transform(to_hw.InferLabelSelectLayer())
    model = model.transform(to_hw.InferThresholdingLayer())
    # convolutions
    model = model.transform(to_hw.InferConvInpGen())
    model = model.transform(to_hw.InferStreamingMaxPool())
    # ----- RESIDUAL HANDLING (the bit the default flow omits) -----
    # eltwise residual add -> AddStreams layer
    model = model.transform(to_hw.InferAddStreamsLayer())
    # the fork that feeds main branch + skip -> DuplicateStreams layer
    model = model.transform(to_hw.InferDuplicateStreamsLayer())
    # GAP -> GlobalAccPool layer (+ a scalar Mul = 1/(H*W) and NCHW<->NHWC transposes)
    model = model.transform(to_hw.InferGlobalAccPoolLayer())
    # channelwise affine (leftover scale/bias) -> ChannelwiseOp
    model = model.transform(to_hw.InferChannelwiseLinearLayer())
    model = _hw_cleanup(model)
    # Relocate the GAP scalar Mul to AFTER the dense MVAU so GlobalAccPool ->
    # Transpose -> Reshape -> MVAU is contiguous, then absorb the conv->FC flatten
    # into the dense weights. Without this the GAP/flatten/Mul nodes split the HW
    # graph into two partitions and CreateDataflowPartition raises
    # "cycle-free graph violated: partition depends on itself".
    model = _relocate_gap_mul_past_dense(model)
    model = _hw_cleanup(model)
    for _ in range(6):
        model = model.transform(AbsorbConsecutiveTransposes())
        model = model.transform(RemoveCNVtoFCFlatten())
        model = model.transform(CollapseRepeatedMul())
        model = _hw_cleanup(model)
    # clean up
    model = model.transform(AbsorbConsecutiveTransposes())
    model = model.transform(RemoveIdentityOps())
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(InferDataLayouts())
    return model


def main():
    # custom step list: tidy/qonnx default, our residual streamline + convert_to_hw,
    # then the standard downstream steps by name.
    custom_steps = [
        "step_qonnx_to_finn",
        "step_tidy_up",
        step_resnet8_streamline,
        step_resnet8_convert_to_hw,
        "step_create_dataflow_partition",
        "step_specialize_layers",
        "step_target_fps_parallelization",
        "step_apply_folding_config",
        "step_minimize_bit_width",
        "step_generate_estimate_reports",
        "step_hw_codegen",
        "step_hw_ipgen",
        "step_set_fifo_depths",
        "step_create_stitched_ip",
        "step_measure_rtlsim_performance",
        "step_out_of_context_synthesis",
        "step_synthesize_bitfile",
        "step_make_pynq_driver",
        "step_deployment_package",
    ]

    stop_step = os.environ.get("RESNET8_STOP_STEP", None)
    start_step = os.environ.get("RESNET8_START_STEP", None)

    cfg = build_cfg.DataflowBuildConfig(
        output_dir=OUTPUT_DIR,
        synth_clk_period_ns=10.0,
        target_fps=1000,  # modest; ResNet-8 on CIFAR-10 32x32 is tiny
        board="ZCU104",
        shell_flow_type=build_cfg.ShellFlowType.VIVADO_ZYNQ,
        steps=custom_steps,
        start_step=start_step,
        stop_step=stop_step,
        auto_fifo_depths=True,  # auto FIFO sizing guards branch deadlock
        split_large_fifos=True,
        standalone_thresholds=True,
        enable_build_pdb_debug=False,
        verify_steps=[VerificationStepType.STREAMLINED_PYTHON]
        if os.environ.get("RESNET8_VERIFY", "0") == "1"
        else [],
        generate_outputs=[
            build_cfg.DataflowOutputType.ESTIMATE_REPORTS,
            build_cfg.DataflowOutputType.STITCHED_IP,
            build_cfg.DataflowOutputType.BITFILE,
            build_cfg.DataflowOutputType.PYNQ_DRIVER,
            build_cfg.DataflowOutputType.DEPLOYMENT_PACKAGE,
        ],
    )
    build.build_dataflow_cfg(MODEL_FILE, cfg)


if __name__ == "__main__":
    main()
