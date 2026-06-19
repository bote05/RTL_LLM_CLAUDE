#!/usr/bin/env python
# RQ2 Leg B (FINN): standalone CIFAR-10 QAT trainer for the Brevitas W4A4 ResNet-8.
"""
RUN (WSL Ubuntu, hls4ml venv, FINN-pinned brevitas):
  PYTHONPATH=/root/tools/finn/deps/brevitas/src:/root/rq2_training/brevitas/pydeps_slim \
  OMP_NUM_THREADS=4 /root/.venv-hls4ml/bin/python \
  /root/rq2_training/brevitas/train_resnet8_w4a4.py --epochs 300

=========================== PREPROCESSING CONTRACT =============================
What FINN's deploy driver MUST replicate (bnn_pynq CNV convention; NO mean/std
normalization anywhere):
  1. x01 = uint8 RGB pixel / 255.0            (torchvision ToTensor; NCHW; RGB order)
  2. INSIDE the exported QONNX graph: x = 2*x01 - 1, then the 8-bit input Quant node
     (scale = 2^-7 = 0.0078125, zero_point = 0, signed, narrow_range = False)
     -> integer input level = clamp(round((2*p/255 - 1) * 128), -128, 127)
  Because the 2x-1 affine and the input Quant node are part of the exported graph,
  the FINN driver simply feeds float x01 in [0,1] NCHW (or lets streamlining absorb
  the affine into a uint8 input + MultiThreshold, the standard bnn_pynq-CNV flow).
Augmentation (train only, NEVER at deploy):
  RandomCrop(32, padding=4) + RandomHorizontalFlip  (bnn_pynq trainer verbatim)
================================================================================

Recipe (VALIDATED by smoke probes 2026-06-12): Adam lr 1e-3, weight decay 0, cosine
annealing to 0 over --epochs, batch 128, cross-entropy. ADAM is the bnn_pynq trainer
default; the MLPerf Tiny ResNet-8 reference also trains with Adam 1e-3.
WARNING: SGD lr 0.1 momentum 0.9 (the classic float-CIFAR recipe) PERMANENTLY KILLS
this W4A4 net — loss spikes to ~7 in the first 100 iters, then collapses to the
degenerate ln(10)=2.3026 plateau (10% acc) and never recovers (3-epoch probe).
Adam 1e-3 learns immediately (loss 3.84 -> 1.78 in 200 iters, ~29% acc).
SGD remains available via --optimizer sgd (use --warmup-epochs and lr <= 0.02).
Checkpoints: ckpt/epoch_NNNN.pt every --ckpt-every (25) epochs + ckpt/best.pt at every
test-set improvement (eval every --eval-every (10) epochs and at the final epoch).
After training, the BEST checkpoint is exported with brevitas.export.export_qonnx to
--qonnx-out (resnet8_w4a4.qonnx) and structurally verified with qonnx (9 Conv + 1 Gemm
+ 3 residual Add + 9 BatchNormalization + Quant/Trunc nodes).
"""

import argparse
import math
import os
import random
import sys
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from resnet8_w4a4 import quant_resnet8_w4a4, verify_topology  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="W4A4 ResNet-8 QAT (FINN Leg B)")
    p.add_argument("--data-root", default="/mnt/d/RTL_LLM_CLAUDE/rq2_resnet8/data",
                   help="dir that CONTAINS cifar-10-batches-py/")
    p.add_argument("--out-dir", default="/root/rq2_training/brevitas")
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--eval-batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--momentum", type=float, default=0.9, help="SGD only")
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--optimizer", choices=["sgd", "adam"], default="adam")
    p.add_argument("--scheduler", choices=["cosine", "step"], default="cosine")
    p.add_argument("--milestones", default="150,225",
                   help="for --scheduler step (gamma 0.1, bnn_pynq MultiStepLR style)")
    p.add_argument("--warmup-epochs", type=int, default=0,
                   help="linear LR warmup (start at lr/10) before the main schedule; "
                        "recommended 5 for the full 300-epoch W4A4 run")
    p.add_argument("--log-every", type=int, default=100,
                   help="print a windowed train-loss line every N iterations")
    p.add_argument("--eval-every", type=int, default=10)
    p.add_argument("--ckpt-every", type=int, default=25)
    p.add_argument("--threads", type=int, default=4,
                   help="HARD CAP: Vivado route on the host — never raise above 4")
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--resume", default="", help="checkpoint to resume from")
    p.add_argument("--export-only", default="",
                   help="skip training; export this checkpoint to QONNX and verify")
    p.add_argument("--qonnx-out", default="",
                   help="default <out-dir>/resnet8_w4a4.qonnx")
    return p.parse_args()


def evaluate(model, loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            logits = model(x)
            correct += (logits.argmax(dim=1) == y).sum().item()
            total += y.numel()
    model.train()
    return 100.0 * correct / total


def save_ckpt(path, model, optimizer, scheduler, epoch, best_acc, test_acc):
    torch.save({
        "epoch": epoch,
        "model": model.state_dict(),
        "optim": optimizer.state_dict() if optimizer is not None else None,
        "sched": scheduler.state_dict() if scheduler is not None else None,
        "best_acc": best_acc,
        "test_acc": test_acc,
    }, path)


def export_and_verify_qonnx(model, qonnx_path):
    """Export with brevitas export_qonnx, then structurally verify with qonnx."""
    from brevitas.export import export_qonnx
    model.eval()
    model.cpu()
    # dynamo=False: brevitas 0.10 QONNX symbolic handlers require the legacy
    # TorchScript ONNX exporter; torch>=2.9 defaults to the dynamo exporter,
    # which chokes on data-dependent control flow in brevitas' handlers.
    export_qonnx(model, args=torch.randn(1, 3, 32, 32), export_path=qonnx_path,
                 dynamo=False)
    print(f"[export] wrote {qonnx_path} ({os.path.getsize(qonnx_path)} bytes)", flush=True)

    from qonnx.core.modelwrapper import ModelWrapper
    m = ModelWrapper(qonnx_path)
    ops = {}
    for n in m.graph.node:
        ops[n.op_type] = ops.get(n.op_type, 0) + 1
    print(f"[export] QONNX op histogram: {dict(sorted(ops.items()))}", flush=True)
    n_conv = ops.get("Conv", 0)
    n_fc = ops.get("Gemm", 0) + ops.get("MatMul", 0)
    n_add = ops.get("Add", 0)
    n_quant = ops.get("Quant", 0)
    n_trunc = ops.get("Trunc", 0)
    n_bn = ops.get("BatchNormalization", 0)
    assert n_conv == 9, f"expected 9 Conv nodes, got {n_conv}"
    assert n_fc == 1, f"expected 1 Gemm/MatMul node, got {n_fc}"
    assert n_add >= 3, f"expected >=3 Add nodes (residual skips), got {n_add}"
    assert n_quant >= 15, f"expected >=15 Quant nodes (weights+acts), got {n_quant}"
    assert n_trunc >= 1, f"expected Trunc node for the GAP TruncAvgPool, got {n_trunc}"
    assert n_bn == 9, f"expected 9 BatchNormalization nodes (BN after every conv), got {n_bn}"
    print(f"[export] VERIFIED: Conv={n_conv} Gemm/MatMul={n_fc} Add={n_add} "
          f"Quant={n_quant} Trunc={n_trunc} BN={n_bn}", flush=True)
    return ops


def main():
    args = parse_args()
    torch.set_num_threads(args.threads)  # HARD CAP — Vivado route shares the host
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass  # already started parallel work (e.g. on resume probing)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    ckpt_dir = os.path.join(args.out_dir, "ckpt")
    os.makedirs(ckpt_dir, exist_ok=True)
    qonnx_out = args.qonnx_out or os.path.join(args.out_dir, "resnet8_w4a4.qonnx")

    model = quant_resnet8_w4a4()
    n_params = verify_topology(model)
    print(f"[model] topology contract OK; trainable params = {n_params}", flush=True)

    if args.export_only:
        pkg = torch.load(args.export_only, map_location="cpu", weights_only=False)
        model.load_state_dict(pkg["model"])
        print(f"[export-only] loaded {args.export_only} "
              f"(epoch {pkg.get('epoch')}, test_acc {pkg.get('test_acc')})", flush=True)
        export_and_verify_qonnx(model, qonnx_out)
        return

    # Data — bnn_pynq trainer transforms VERBATIM (ToTensor only; NO normalization).
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()])
    transform_test = transforms.Compose([transforms.ToTensor()])
    train_set = CIFAR10(root=args.data_root, train=True, download=False,
                        transform=transform_train)
    test_set = CIFAR10(root=args.data_root, train=False, download=False,
                       transform=transform_test)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, drop_last=False)
    test_loader = DataLoader(test_set, batch_size=args.eval_batch_size, shuffle=False,
                             num_workers=args.workers)

    if args.optimizer == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)
    else:  # bnn_pynq ADAM convention
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                     weight_decay=args.weight_decay)
    main_epochs = max(1, args.epochs - args.warmup_epochs)
    if args.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                               T_max=main_epochs)
    else:
        milestones = [int(i) for i in args.milestones.split(",")]
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                         milestones=milestones,
                                                         gamma=0.1)
    if args.warmup_epochs > 0:
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.1, total_iters=args.warmup_epochs)
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, [warmup, scheduler], milestones=[args.warmup_epochs])
    criterion = nn.CrossEntropyLoss()

    start_epoch, best_acc = 1, -1.0
    if args.resume:
        pkg = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(pkg["model"])
        if pkg.get("optim"):
            optimizer.load_state_dict(pkg["optim"])
        if pkg.get("sched"):
            scheduler.load_state_dict(pkg["sched"])
        start_epoch = pkg["epoch"] + 1
        best_acc = pkg.get("best_acc", -1.0)
        print(f"[resume] {args.resume} -> starting at epoch {start_epoch}, "
              f"best_acc {best_acc:.2f}", flush=True)

    print(f"[train] epochs={args.epochs} batch={args.batch_size} opt={args.optimizer} "
          f"lr={args.lr} wd={args.weight_decay} sched={args.scheduler} "
          f"threads={args.threads} train_iters/epoch={len(train_loader)}", flush=True)

    model.train()
    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        run_loss, run_correct, run_total = 0.0, 0, 0
        win_loss, win_n = 0.0, 0
        for it, (x, y) in enumerate(train_loader):
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            lval = loss.item()
            if not math.isfinite(lval):
                raise RuntimeError(
                    f"NaN/Inf loss at epoch {epoch} iter {it}: {lval} — aborting")
            loss.backward()
            optimizer.step()
            run_loss += lval * y.numel()
            run_correct += (logits.argmax(dim=1) == y).sum().item()
            run_total += y.numel()
            win_loss += lval
            win_n += 1
            if (it + 1) % args.log_every == 0:
                print(f"[epoch {epoch:4d} it {it + 1:4d}/{len(train_loader)}] "
                      f"loss(win{args.log_every}) {win_loss / win_n:.4f}", flush=True)
                win_loss, win_n = 0.0, 0
        scheduler.step()
        dt = time.time() - t0
        print(f"[epoch {epoch:4d}/{args.epochs}] loss {run_loss / run_total:.4f} "
              f"train_acc {100.0 * run_correct / run_total:6.2f}% "
              f"lr {optimizer.param_groups[0]['lr']:.5f} time {dt:7.1f}s", flush=True)

        test_acc = None
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            test_acc = evaluate(model, test_loader)
            star = ""
            if test_acc > best_acc:
                best_acc = test_acc
                save_ckpt(os.path.join(ckpt_dir, "best.pt"), model, optimizer,
                          scheduler, epoch, best_acc, test_acc)
                star = "  <- new best (ckpt/best.pt)"
            print(f"[eval  {epoch:4d}] test_acc {test_acc:6.2f}% "
                  f"(best {best_acc:6.2f}%){star}", flush=True)
        if epoch % args.ckpt_every == 0 or epoch == args.epochs:
            path = os.path.join(ckpt_dir, f"epoch_{epoch:04d}.pt")
            save_ckpt(path, model, optimizer, scheduler, epoch, best_acc, test_acc)
            print(f"[ckpt  {epoch:4d}] wrote {path}", flush=True)

    # Export the BEST checkpoint to QONNX (epoch-1 smoke exercises the same path).
    best_path = os.path.join(ckpt_dir, "best.pt")
    if os.path.exists(best_path):
        pkg = torch.load(best_path, map_location="cpu", weights_only=False)
        export_model = quant_resnet8_w4a4()
        export_model.load_state_dict(pkg["model"])
        print(f"[export] exporting BEST checkpoint (epoch {pkg['epoch']}, "
              f"test_acc {pkg['test_acc']:.2f}%)", flush=True)
        export_and_verify_qonnx(export_model, qonnx_out)
    else:
        print("[export] WARNING: no best.pt found, exporting the live model", flush=True)
        export_and_verify_qonnx(model, qonnx_out)
    print(f"[done] best test_acc {best_acc:.2f}%  qonnx: {qonnx_out}", flush=True)


if __name__ == "__main__":
    main()
