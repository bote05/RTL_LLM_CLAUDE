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
from finn.transformation.streamline.absorb import (
    AbsorbConsecutiveTransposes, AbsorbScalarMulAddIntoTopK,
)
import finn.transformation.streamline.reorder as reorder

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
    """QuantAvgPool2d (kernel==stride==spatial => global) -> GlobalAveragePool.
    Then InferGlobalAccPoolLayer maps it cleanly (GlobalAccPool + scalar Mul)."""
    g = model.graph
    for node in list(g.node):
        if node.op_type != "QuantAvgPool2d":
            continue
        attrs = {a.name: a for a in node.attribute}
        ishape = model.get_tensor_shape(node.input[0])  # NCHW [N,C,H,W]
        k = attrs["kernel"].i
        H, W = ishape[2], ishape[3]
        assert k == H == W, "not a global pool: kernel=%d HxW=%dx%d" % (k, H, W)
        new = helper.make_node(
            "GlobalAveragePool", [node.input[0]], [node.output[0]],
            name="GlobalAveragePool_" + node.name,
        )
        idx = list(g.node).index(node)
        g.node.insert(idx, new)
        g.node.remove(node)
    model = model.transform(InferShapes())
    model = model.transform(InferDataTypes())
    return model


m = ModelWrapper("/root/rq2_training/brevitas/resnet8_w4a4.qonnx")
m = step_qonnx_to_finn(m, cfg)
m = step_tidy_up(m, cfg)
m = b.step_resnet8_streamline(m, cfg)
m = b.step_resnet8_convert_to_hw(m, cfg)
hist(m, "convert_to_hw (pre-GAP)")

# GAP handling
m = quantavgpool_to_globalavgpool(m)
m = m.transform(InferDataLayouts())
m = m.transform(to_hw.InferGlobalAccPoolLayer())
m = m.transform(GiveUniqueNodeNames())
m = m.transform(GiveReadableTensorNames())
hist(m, "after GAP->GlobalAccPool")

nonhw = [(n.op_type, n.name) for n in m.graph.node
         if not n.domain.startswith("finn.custom_op.fpgadataflow")]
print("NON-HW:", nonhw)
m.save("/root/rq2_training/finn_resnet8/_dbg_hw_gap.onnx")

# Try the partition
m = step_create_dataflow_partition(m, cfg)
hist(m, "PARENT after partition")
print("PARTITION OK")
