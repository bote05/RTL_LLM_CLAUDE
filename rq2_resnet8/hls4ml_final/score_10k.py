"""Score the retrained QKeras ResNet-8 .h5 on the full CIFAR-10 10k test set.
This is the accuracy number for the RQ2 comparison row (val==test on CIFAR-10 10k).
Uses the SAME /256.0 preprocessing the QAT training used.
"""
import os, pickle, sys
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("OMP_NUM_THREADS", "8")
import numpy as np
sys.path.insert(0, "/root/rq2_training/qkeras")
from resnet8_qkeras8 import load_qkeras_h5

DATA = "/mnt/d/RTL_LLM_CLAUDE/rq2_resnet8/data/cifar-10-batches-py"
# Score BOTH the no-softmax export (used by hls4ml) and the with-softmax best.h5;
# argmax is identical (softmax is monotonic) -- report the export model.
M_NOSM = "/root/rq2_training/qkeras_gpu_fixed_full/resnet8_qkeras8_best_nosoftmax.h5"
M_SM = "/root/rq2_training/qkeras_gpu_fixed_full/resnet8_qkeras8_best.h5"

def load_test():
    with open(os.path.join(DATA, "test_batch"), "rb") as fo:
        d = pickle.load(fo, encoding="bytes")
    x = d[b"data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    y = np.array(d[b"labels"])
    x = (x / 256.0).astype("float32")
    return x, y

x, y = load_test()
print("[score] test set %s labels %s" % (x.shape, y.shape), flush=True)

m = load_qkeras_h5(M_NOSM)
logits = m.predict(x, batch_size=256, verbose=0)
pred = logits.argmax(axis=1)
acc = float((pred == y).mean())
print("NOSOFTMAX_TOP1_10k=%.4f (%d/%d)" % (acc, int((pred == y).sum()), len(y)), flush=True)

# sanity: with-softmax model must give identical argmax
ms = load_qkeras_h5(M_SM)
probs = ms.predict(x, batch_size=256, verbose=0)
preds = probs.argmax(axis=1)
accs = float((preds == y).mean())
print("SOFTMAX_TOP1_10k=%.4f (%d/%d)  argmax_match_nosm=%d/%d"
      % (accs, int((preds == y).sum()), len(y), int((preds == pred).sum()), len(y)),
      flush=True)
print("SCORE_DONE", flush=True)
