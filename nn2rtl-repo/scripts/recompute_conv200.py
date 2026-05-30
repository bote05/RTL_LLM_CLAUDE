#!/usr/bin/env python3
"""DECISIVE test: recompute conv_200 from the BYTE-EXACT conv_198 probe capture
(= conv_200 input after relu_1) + on-disk flat weights + scale.mem + bias, then
compare to (a) the golden and (b) the RTL capture.

conv_198 is byte-exact => the input activation LAYOUT is known-good (no guessing).
This separates "RTL datapath bug" from "stale/different golden":
  recompute==golden, recompute!=RTL  => RTL datapath is BUGGY (have a reference now)
  recompute==RTL,    recompute!=gold => golden is stale/different (phantom)
"""
import struct, glob
from pathlib import Path
import numpy as np

ROOT = Path("c:/Users/User/Desktop/RTL_LLM_CLAUDE/nn2rtl-repo")
IC=OC=64; IH=IW=OH=OW=56; KH=KW=3; PH=PW=1; SH=SW=1

def goldbytes(mod):
    g=Path(glob.glob(str(ROOT/f"output/goldens/contracts/node_{mod}_*/node_{mod}.goldout"))[0]).read_bytes()
    _,nv,_,spv,bps=struct.unpack("<4sIIII",g[:20]); return np.frombuffer(g[20:20+spv*bps],dtype=np.int8).astype(np.int64)

pdir=ROOT/"output/reports_integrated/verilator_nn2rtl_top_probe"
# --- conv_198 capture (byte-exact) -> conv_200 input (relu) ---
c198=np.frombuffer((pdir/"probe_node_conv_198.bin").read_bytes(),dtype=np.int8).astype(np.int64)  # 6272*32
inp=c198.reshape(IH*IW, IC)        # 3136 pixels x 64 ch (beat0=ch0-31, beat1=ch32-63 -> contiguous 64)
inp=np.maximum(inp,0)              # relu_1
inp=inp.reshape(IH,IW,IC)

# --- weights flat [OC,IC,KH,KW] C-order: flat[oc*576 + ic*9 + kh*3 + kw] ---
wtxt=(ROOT/"output/weights/node_conv_200_weights.hex").read_text().split()
W=np.array([int(x,16) for x in wtxt],dtype=np.int64); W=np.where(W>127,W-256,W)
print(f"weights: {W.size} (expect {OC*IC*KH*KW})")
W=W.reshape(OC,IC,KH,KW)

# --- bias ---
btxt=(ROOT/"output/weights/node_conv_200_bias.hex").read_text().split()
bias=np.array([int(x,16) for x in btxt],dtype=np.int64)
# sign-extend by width: bias hex may be wide (32-bit). detect nibble len.
bw=len(btxt[0])*4
bias=np.where(bias>=(1<<(bw-1)), bias-(1<<bw), bias)
print(f"bias: {bias.size} entries, {bw}-bit, range[{bias.min()},{bias.max()}]")

# --- scale.mem: {shift[21:16], mult[15:0]} ---
stxt=(ROOT/"output/weights/node_conv_200_scale.mem").read_text().split()
sv=np.array([int(x,16) for x in stxt],dtype=np.int64)
mult=sv & 0xFFFF; shift=(sv>>16)&0x3F
print(f"scale: {sv.size} entries, mult range[{mult.min()},{mult.max()}], shift range[{shift.min()},{shift.max()}]")

# --- conv2d pad1 stride1, then per-OC requant ---
pad=np.zeros((IH+2*PH, IW+2*PW, IC),dtype=np.int64); pad[PH:PH+IH, PW:PW+IW]=inp
acc=np.zeros((OH,OW,OC),dtype=np.int64)
for kh in range(KH):
    for kw in range(KW):
        patch=pad[kh:kh+OH, kw:kw+OW, :]            # OH,OW,IC
        # W[:, :, kh, kw] is OC x IC ; contract over IC
        acc += np.tensordot(patch, W[:,:,kh,kw], axes=([2],[1]))  # OH,OW,OC
biased=acc + bias[None,None,:]
scaled=biased * mult[None,None,:]
rnd=np.where(shift>0, (1<<np.maximum(shift-1,0)), 0)[None,None,:]
v=(scaled + rnd) >> shift[None,None,:]              # arithmetic (numpy >> on int64 is arithmetic)
out=np.clip(v,-128,127).astype(np.int64)
recompute=out.reshape(OH*OW, OC).reshape(-1)        # 3136*64 = 200704 bytes order ch0..63 per pixel

gold=goldbytes("conv_200")
rtl=np.frombuffer((pdir/"probe_node_conv_200.bin").read_bytes(),dtype=np.int8).astype(np.int64)
n=min(recompute.size,gold.size,rtl.size)
rc,gd,rt=recompute[:n],gold[:n],rtl[:n]
def cmp(a,b,label):
    pos=100*(a!=b).mean(); ms=100*(np.sort(a)!=np.sort(b)).mean()
    print(f"  {label:28s}: POSITION {pos:5.1f}%  MULTISET {ms:5.1f}%")
print(f"\nrecompute range[{rc.min()},{rc.max()}]  gold[{gd.min()},{gd.max()}]  rtl[{rt.min()},{rt.max()}]")
cmp(rc,gd,"recompute vs GOLDEN")
cmp(rc,rt,"recompute vs RTL capture")
cmp(rt,gd,"RTL capture vs GOLDEN (ref)")
