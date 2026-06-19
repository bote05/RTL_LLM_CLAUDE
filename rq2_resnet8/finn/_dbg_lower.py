import collections
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.general import (
    GiveUniqueNodeNames, GiveReadableTensorNames, SortGraph,
)
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.transformation.infer_data_layouts import InferDataLayouts
from qonnx.transformation.lower_convs_to_matmul import LowerConvsToMatMul
from finn.transformation.streamline.absorb import (
    AbsorbConsecutiveTransposes, AbsorbTransposeIntoMultiThreshold,
)
from finn.transformation.streamline.reorder import (
    MoveTransposePastFork, MoveTransposePastJoinAdd, MakeMaxPoolNHWC,
)

m = ModelWrapper("/root/rq2_training/finn_resnet8/_dbg_streamlined.onnx")


def report(m, tag):
    h = collections.Counter(n.op_type for n in m.graph.node)
    print("[%s] nodes=%d Transpose=%d Im2Col=%d" % (
        tag, len(m.graph.node), h.get("Transpose", 0), h.get("Im2Col", 0)))


def cleanup(m):
    m = m.transform(SortGraph())
    m = m.transform(InferShapes())
    m = m.transform(InferDataTypes())
    m = m.transform(GiveUniqueNodeNames())
    m = m.transform(GiveReadableTensorNames())
    return m


m = m.transform(LowerConvsToMatMul())
m = m.transform(MakeMaxPoolNHWC())
report(m, "after lower")
for it in range(8):
    m = m.transform(MoveTransposePastFork())
    m = cleanup(m)
    m = m.transform(MoveTransposePastJoinAdd())
    m = cleanup(m)
    m = m.transform(AbsorbConsecutiveTransposes())
    m = cleanup(m)
    m = m.transform(AbsorbTransposeIntoMultiThreshold())
    m = cleanup(m)
    report(m, "iter %d" % it)
m = m.transform(InferDataLayouts())
report(m, "FINAL")
print("=== remaining transposes ===")
for n in m.graph.node:
    if n.op_type == "Transpose":
        perm = [list(a.ints) for a in n.attribute if a.name == "perm"]
        prod = m.find_producer(n.input[0])
        cons = m.find_consumers(n.output[0])
        print(n.name, perm, "<-", None if prod is None else prod.op_type,
              "->", [c.op_type for c in cons] if cons else "OUTPUT")
m.save("/root/rq2_training/finn_resnet8/_dbg_lowered.onnx")
