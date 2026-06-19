import collections
from qonnx.core.modelwrapper import ModelWrapper
import build_resnet8_zcu104 as b
import finn.builder.build_dataflow_config as build_cfg
from finn.builder.build_dataflow_steps import step_qonnx_to_finn, step_tidy_up
from qonnx.transformation.general import GiveUniqueNodeNames, GiveReadableTensorNames
from qonnx.transformation.infer_datatypes import InferDataTypes
from finn.transformation.streamline.reorder import (
    MoveLinearPastEltwiseAdd, MoveLinearPastFork, MoveScalarLinearPastInvariants,
)

cfg = build_cfg.DataflowBuildConfig(
    output_dir="/tmp/x", synth_clk_period_ns=10.0, target_fps=1000, board="ZCU104",
    shell_flow_type=build_cfg.ShellFlowType.VIVADO_ZYNQ,
    generate_outputs=[build_cfg.DataflowOutputType.ESTIMATE_REPORTS])
m = ModelWrapper("/root/rq2_training/brevitas/resnet8_w4a4.qonnx")
m = step_qonnx_to_finn(m, cfg)
m = step_tidy_up(m, cfg)
m = b.step_resnet8_streamline_linear(m, cfg)


def stem_ok(m, tag):
    convs = [n for n in m.graph.node if n.op_type == "Conv"]
    stem = convs[0]
    prod = m.find_producer(stem.input[0])
    pt = None if prod is None else prod.op_type
    print("[%s] stem_in=%s producer=%s nodes=%d" % (tag, stem.input[0], pt, len(m.graph.node)))


stem_ok(m, "post linear")
for trn, nm in [
    (MoveLinearPastEltwiseAdd(), "MoveLinearPastEltwiseAdd"),
    (MoveLinearPastFork(), "MoveLinearPastFork"),
    (MoveScalarLinearPastInvariants(), "MoveScalarLinearPastInvariants"),
]:
    m = m.transform(trn)
    m = m.transform(GiveUniqueNodeNames())
    m = m.transform(GiveReadableTensorNames())
    m = m.transform(InferDataTypes())
    stem_ok(m, "after " + nm)
