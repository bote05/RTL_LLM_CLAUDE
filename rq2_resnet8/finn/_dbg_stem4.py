import collections
from qonnx.core.modelwrapper import ModelWrapper
import build_resnet8_zcu104 as b
import finn.builder.build_dataflow_config as build_cfg
from finn.builder.build_dataflow_steps import step_qonnx_to_finn, step_tidy_up
from qonnx.transformation.general import (
    GiveUniqueNodeNames, GiveReadableTensorNames, SortGraph, RemoveUnusedTensors,
)
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.transformation.infer_shapes import InferShapes
from finn.transformation.streamline.reorder import (
    MoveLinearPastEltwiseAdd, MoveLinearPastFork, MoveMulPastFork, MoveAddPastFork,
    MoveScalarLinearPastInvariants,
)

cfg = build_cfg.DataflowBuildConfig(
    output_dir="/tmp/x", synth_clk_period_ns=10.0, target_fps=1000, board="ZCU104",
    shell_flow_type=build_cfg.ShellFlowType.VIVADO_ZYNQ,
    generate_outputs=[build_cfg.DataflowOutputType.ESTIMATE_REPORTS])
m = ModelWrapper("/root/rq2_training/brevitas/resnet8_w4a4.qonnx")
m = step_qonnx_to_finn(m, cfg)
m = step_tidy_up(m, cfg)
m = b.step_resnet8_streamline_linear(m, cfg)


def report(m, tag):
    convs = [n for n in m.graph.node if n.op_type == "Conv"]
    stem = convs[0]
    prod = m.find_producer(stem.input[0])
    pt = None if prod is None else prod.op_type
    dangling = []
    for n in m.graph.node:
        for inp in n.input:
            if inp == "":
                continue
            if (m.find_producer(inp) is None and m.get_initializer(inp) is None
                    and inp not in [i.name for i in m.graph.input]):
                dangling.append(inp)
    h = collections.Counter(n.op_type for n in m.graph.node)
    print("[%s] stem_prod=%s nodes=%d Mul=%d Add=%d dangling=%s" % (
        tag, pt, len(m.graph.node), h.get("Mul", 0), h.get("Add", 0), sorted(set(dangling))))


def cleanup(m):
    m = m.transform(InferShapes())
    m = m.transform(InferDataTypes())
    m = m.transform(GiveUniqueNodeNames())
    m = m.transform(GiveReadableTensorNames())
    return m


report(m, "post linear")
# Fork-resolving movers FIRST (duplicate linears across forks)
for it in range(3):
    m = m.transform(MoveMulPastFork())
    m = m.transform(MoveAddPastFork())
    m = cleanup(m)
report(m, "after Mul/AddPastFork x3")
m = m.transform(MoveLinearPastEltwiseAdd())
m = cleanup(m)
report(m, "after MoveLinearPastEltwiseAdd")
m = m.transform(MoveScalarLinearPastInvariants())
m = cleanup(m)
report(m, "after MoveScalarLinearPastInvariants")
