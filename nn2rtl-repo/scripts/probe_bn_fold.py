import json
from pathlib import Path
import numpy as np, torch
from torchvision.models import resnet50, ResNet50_Weights
ROOT = Path(__file__).resolve().parent.parent
IR = json.loads((ROOT/"output/layer_ir.json").read_text())
layers = IR["layers"] if isinstance(IR,dict) and "layers" in IR else IR
by_id = {(L.get("module_id") or ""):L for L in layers}
m = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2).eval()

# stem: conv1 -> bn1 ; reconstruct BN fold factor and compare to weight_scale_per_oc/s_gen ratio
def bn_scale(bn): return (bn.weight.detach().numpy()/np.sqrt(bn.running_var.detach().numpy()+bn.eps))

W = m.conv1.weight.detach().numpy().astype(np.float64); oc=W.shape[0]
qmax=7
s_gen = np.abs(W.reshape(oc,-1)).max(1)/qmax
wspo = np.asarray(by_id["node_conv_196"]["weight_scale_per_oc"],dtype=np.float64)[:oc]
ratio = wspo/s_gen
bn = np.abs(bn_scale(m.bn1))[:oc]
print("=== conv_196 (conv1->bn1), first 8 OC ===")
print("ratio  wspo/s_gen :", np.round(ratio[:8],4).tolist())
print("|BN fold gamma/sqrt(var+eps)|:", np.round(bn[:8],4).tolist())
print(f"corr(ratio, BN) = {np.corrcoef(ratio,bn)[0,1]:.4f}   median|ratio-BN|/BN = {np.median(np.abs(ratio-bn)/bn):.4f}")

# Also: does the ONNX model (what deployment quantizes) have separate BN nodes or folded?
try:
    import onnx
    g = onnx.load(str(ROOT/"checkpoints/resnet50_full.onnx")).graph
    from collections import Counter
    c = Counter(n.op_type for n in g.node)
    print("\n=== ONNX node op_type counts ===")
    for k in ("Conv","BatchNormalization","Relu","Add","Gemm","GlobalAveragePool","MaxPool"):
        print(f"   {k}: {c.get(k,0)}")
    print("   (Conv>>BatchNormalization => BN is FOLDED into Conv weights)")
except Exception as e:
    print("onnx load failed:", e)
