import collections
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.general import (
    GiveUniqueNodeNames, GiveReadableTensorNames, SortGraph,
)
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.transformation.infer_data_layouts import InferDataLayouts
from finn.transformation.streamline.absorb import AbsorbConsecutiveTransposes
from finn.transformation.streamline.reorder import (
    MoveTransposePastFork, MoveTransposePastJoinAdd,
)

m = ModelWrapper("/root/rq2_training/finn_resnet8/_dbg_hw2.onnx")


def report(m, tag):
    h = collections.Counter(n.op_type for n in m.graph.node)
    print("[%s] nodes=%d Transpose=%d" % (tag, len(m.graph.node), h.get("Transpose", 0)))


def cleanup(m):
    m = m.transform(SortGraph())
    m = m.transform(InferShapes())
    m = m.transform(InferDataTypes())
    m = m.transform(GiveUniqueNodeNames())
    return m


report(m, "start")
for it in range(5):
    m = m.transform(MoveTransposePastFork())
    m = cleanup(m)
    m = m.transform(MoveTransposePastJoinAdd())
    m = cleanup(m)
    m = m.transform(AbsorbConsecutiveTransposes())
    m = cleanup(m)
    report(m, "iter %d" % it)
m = m.transform(InferDataLayouts())
report(m, "final")
m.save("/root/rq2_training/finn_resnet8/_dbg_hw3.onnx")
print("=== remaining transposes ===")
for n in m.graph.node:
    if n.op_type == "Transpose":
        perm = [list(a.ints) for a in n.attribute if a.name == "perm"]
        prod = m.find_producer(n.input[0])
        cons = m.find_consumers(n.output[0])
        print(n.name, perm, "<-", None if prod is None else prod.op_type,
              "->", [c.op_type for c in cons])
print("=== non-HW remaining ===")
nonhw = [(n.op_type, n.name) for n in m.graph.node
         if not n.domain.startswith("finn.custom_op.fpgadataflow")]
print(nonhw)
