#!/usr/bin/env python
# Quick recipe probe for the W4A4 ResNet-8: does <optimizer, lr> learn in N iters?
# Usage: probe_lr.py {adam|sgd} LR ITERS
import os
import sys
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from resnet8_w4a4 import quant_resnet8_w4a4  # noqa: E402

torch.set_num_threads(4)
torch.manual_seed(2026)

opt_name, lr, iters = sys.argv[1], float(sys.argv[2]), int(sys.argv[3])
tf = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()])
ds = CIFAR10(root="/mnt/d/RTL_LLM_CLAUDE/rq2_resnet8/data", train=True,
             download=False, transform=tf)
dl = DataLoader(ds, batch_size=128, shuffle=True)

m = quant_resnet8_w4a4()
m.train()
if opt_name == "adam":
    opt = torch.optim.Adam(m.parameters(), lr=lr)
else:
    opt = torch.optim.SGD(m.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
crit = nn.CrossEntropyLoss()

t0, win, correct, total = time.time(), [], 0, 0
it = 0
while it < iters:
    for x, y in dl:
        if it >= iters:
            break
        opt.zero_grad()
        out = m(x)
        loss = crit(out, y)
        loss.backward()
        opt.step()
        win.append(loss.item())
        correct += (out.argmax(1) == y).sum().item()
        total += y.numel()
        it += 1
        if it % 25 == 0:
            print(f"it {it:4d}: loss(w25) {sum(win) / len(win):.4f} "
                  f"acc_sofar {100 * correct / total:.1f}%", flush=True)
            win = []
print(f"PROBE {opt_name} lr={lr}: acc {100 * correct / total:.1f}% "
      f"({iters} iters, {time.time() - t0:.0f}s)")
