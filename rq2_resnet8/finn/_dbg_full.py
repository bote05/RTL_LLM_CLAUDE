import collections
from qonnx.core.modelwrapper import ModelWrapper
import build_resnet8_zcu104 as b
import finn.builder.build_dataflow_config as build_cfg
from finn.builder.build_dataflow_steps import (
    step_qonnx_to_finn, step_tidy_up, step_create_dataflow_partition,
)

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
hist(m, "streamlined")

# stem check
convs = [n for n in m.graph.node if n.op_type == "Conv"]
prod = m.find_producer(convs[0].input[0]) if convs else None
print("STEM producer:", None if prod is None else prod.op_type)

m = b.step_resnet8_convert_to_hw(m, cfg)
hist(m, "convert_to_hw")
nonhw = [(n.op_type, n.name) for n in m.graph.node
         if not n.domain.startswith("finn.custom_op.fpgadataflow")]
print("NON-HW remaining:", nonhw if nonhw else "NONE")
m.save("/root/rq2_training/finn_resnet8/_dbg_hw2.onnx")

# dataflow partition
m = step_create_dataflow_partition(m, cfg)
hist(m, "dataflow partition (child)")
print("PARTITION OK")
