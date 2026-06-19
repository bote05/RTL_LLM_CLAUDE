import collections
import numpy as np
from onnx import helper
from qonnx.core.modelwrapper import ModelWrapper
import build_resnet8_zcu104 as b
import finn.builder.build_dataflow_config as build_cfg
import finn.transformation.fpgadataflow.convert_to_hw_layers as to_hw
from finn.builder.build_dataflow_steps import (
    step_qonnx_to_finn, step_tidy_up, step_create_dataflow_partition,
)
from qonnx.transformation.general import (
    GiveUniqueNodeNames, GiveReadableTensorNames, SortGraph, RemoveUnusedTensors,
)
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.transformation.infer_data_layouts import InferDataLayouts
from finn.transformation.streamline.absorb import AbsorbConsecutiveTransposes
from finn.transformation.move_reshape import RemoveCNVtoFCFlatten
from finn.transformation.streamline import Streamline
from finn.transformation.streamline.reorder import (
    MoveScalarLinearPastInvariants, MoveScalarMulPastMatMul,
)
from finn.transformation.streamline.collapse_repeated import CollapseRepeatedMul

cfg = build_cfg.DataflowBuildConfig(
    output_dir="/root/rq2_training/finn_resnet8/_interactive",
    synth_clk_period_ns=10.0, target_fps=1000, board="ZCU104",
    shell_flow_type=build_cfg.ShellFlowType.VIVADO_ZYNQ,
    standalone_thresholds=True, auto_fifo_depths=True,
    generate_outputs=[build_cfg.DataflowOutputType.ESTIMATE_REPORTS])


def hist(m, tag):
    h = collections.Counter(n.op_type for n in m.graph.node)
    print("[%s] nodes=%d hist=%s" % (tag, len(m.graph.node), dict(h)))


def quantavgpool_to_globalavgpool(model):
    g = model.graph
    for node in list(g.node):
        if node.op_type != "QuantAvgPool2d":
            continue
        attrs = {a.name: a for a in node.attribute}
        ishape = model.get_tensor_shape(node.input[0])
        k = attrs["kernel"].i
        H, W = ishape[2], ishape[3]
        assert k == H == W
        new = helper.make_node("GlobalAveragePool", [node.input[0]], [node.output[0]],
                               name="GlobalAveragePool_" + node.name)
        idx = list(g.node).index(node)
        g.node.insert(idx, new)
        g.node.remove(node)
    return model.transform(InferShapes()).transform(InferDataTypes())


def cleanup(m):
    m = m.transform(SortGraph())
    m = m.transform(InferShapes())
    m = m.transform(InferDataTypes())
    m = m.transform(GiveUniqueNodeNames())
    m = m.transform(GiveReadableTensorNames())
    return m


m = ModelWrapper("/root/rq2_training/brevitas/resnet8_w4a4.qonnx")
m = step_qonnx_to_finn(m, cfg)
m = step_tidy_up(m, cfg)
m = b.step_resnet8_streamline(m, cfg)
hist(m, "streamlined")

# --- GAP + dense-tail handling while MatMul is STILL plain MatMul ---
m = quantavgpool_to_globalavgpool(m)
m = cleanup(m)
# commute the (per-tensor scale) Muls around the dense MatMul + flatten so the
# GAP -> Transpose -> Reshape -> MatMul chain is clean (RemoveCNVtoFCFlatten then
# absorbs the flatten into the MatMul weights). Iterate to a fixed point.
for _ in range(6):
    m = m.transform(MoveScalarLinearPastInvariants())
    m = m.transform(MoveScalarMulPastMatMul())
    m = m.transform(CollapseRepeatedMul())
    m = cleanup(m)
hist(m, "after dense-tail commute")
print("=== tail ===")
for i, n in enumerate(m.graph.node):
    if i < len(m.graph.node) - 8:
        continue
    print("%2d %-22s in:%s -> out:%s" % (i, n.op_type, list(n.input)[:1], list(n.output)[:1]))
m.save("/root/rq2_training/finn_resnet8/_dbg_gap3.onnx")
