import collections
from qonnx.core.modelwrapper import ModelWrapper
import build_resnet8_zcu104 as b
import finn.builder.build_dataflow_config as build_cfg
import finn.transformation.fpgadataflow.convert_to_hw_layers as to_hw
from finn.builder.build_dataflow_steps import (
    step_qonnx_to_finn, step_tidy_up, step_create_dataflow_partition,
)
from qonnx.transformation.general import (
    GiveUniqueNodeNames, GiveReadableTensorNames, SortGraph,
)
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.transformation.infer_data_layouts import InferDataLayouts
from finn.transformation.streamline.absorb import AbsorbConsecutiveTransposes
from finn.transformation.move_reshape import RemoveCNVtoFCFlatten
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


m = ModelWrapper("/root/rq2_training/brevitas/resnet8_w4a4.qonnx")
m = step_qonnx_to_finn(m, cfg)
m = step_tidy_up(m, cfg)
m = b.step_resnet8_streamline(m, cfg)
m = b.step_resnet8_convert_to_hw(m, cfg)
hist(m, "convert_to_hw")
print("PARTITION via step:")
m = step_create_dataflow_partition(m, cfg)
hist(m, "PARENT")
print("PARTITION OK")
