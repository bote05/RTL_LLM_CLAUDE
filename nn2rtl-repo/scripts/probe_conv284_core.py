import struct
from pathlib import Path
import numpy as np

ROOT = Path("c:/Users/User/Desktop/RTL_LLM_CLAUDE/nn2rtl-repo")
IC=OC=512; IH=IW=14; OH=OW=7; KH=KW=3; PH=PW=1; SH=SW=2

def loadgold(path):
    g=Path(path).read_bytes()
    magic,nv,_,spv,bps=struct.unpack("<4sIIII",g[:20])
    data=np.frombuffer(g[20:20+spv*bps],dtype=np.int8).astype(np.int64)
    return data,spv,bps

ts=ROOT/"output/goldens/contracts/node_conv_284_tiled-streaming_conv2d_512x512x3x3_s14x14_st2x2_p1x1_d1x1_g1_mp4_i"
dr=ROOT/"output/goldens/contracts/node_conv_284_dram-backed-weights_conv2d_512x512x3x3_s14x14_st2x2_p1x1_d1x1_g1_m"
gin_ts,_,_=loadgold(ts/"node_conv_284.goldin")
gout_ts,_,_=loadgold(ts/"node_conv_284.goldout")
gin_dr,_,_=loadgold(dr/"node_conv_284.goldin")
gout_dr,_,_=loadgold(dr/"node_conv_284.goldout")

# weights flat [OC,IC,KH,KW]
wtxt=(ROOT/"output/weights/node_conv_284_weights.hex").read_text().split()
W=np.array([int(x,16) for x in wtxt],dtype=np.int64); W=np.where(W>127,W-256,W)  # INT4 stored as 8-bit 2c
print("weights:",W.size,"expect",OC*IC*KH*KW,"range",W.min(),W.max())
W=W.reshape(OC,IC,KH,KW)

btxt=(ROOT/"output/weights/node_conv_284_bias.hex").read_text().split()
bias=np.array([int(x,16) for x in btxt],dtype=np.int64)
bw=len(btxt[0])*4
bias=np.where(bias>=(1<<(bw-1)), bias-(1<<bw), bias)
print("bias:",bias.size,bw,"bit range",bias.min(),bias.max())

stxt=(ROOT/"output/weights/node_conv_284_scale.mem").read_text().split()
sv=np.array([int(x,16) for x in stxt],dtype=np.int64)
mult=sv & 0xFFFF; shift=(sv>>16)&0x3F
print("scale:",sv.size,"mult",mult.min(),mult.max(),"shift",shift.min(),shift.max())

def requant(acc):
    # acc: OH,OW,OC
    biased=acc + bias[None,None,:]
    scaled=biased*mult[None,None,:]
    rnd=np.where(shift>0,(1<<np.maximum(shift-1,0)),0)[None,None,:]
    v=(scaled+rnd)>>shift[None,None,:]
    return np.clip(v,-128,127).astype(np.int64)

def conv(inp_hwc):
    inp=inp_hwc  # IH,IW,IC
    pad=np.zeros((IH+2*PH,IW+2*PW,IC),dtype=np.int64); pad[PH:PH+IH,PW:PW+IW]=inp
    acc=np.zeros((OH,OW,OC),dtype=np.int64)
    for kh in range(KH):
        for kw in range(KW):
            patch=pad[kh:kh+OH*SH:SH, kw:kw+OW*SW:SW, :]   # OH,OW,IC
            acc += np.tensordot(patch, W[:,:,kh,kw], axes=([2],[1]))
    return requant(acc)

def cmp(a,b):
    n=min(a.size,b.size)
    pos=100*(a[:n]!=b[:n]).mean()
    ms=100*(np.sort(a)!=np.sort(b))[:].mean() if a.size==b.size else float('nan')
    return pos,ms

# standard layout: goldin raster pixel-major beat k -> ch[32k:32k+32]
inp_std_ts=gin_ts.reshape(196,16,32).reshape(196,512).reshape(14,14,512)
inp_std_dr=gin_dr.reshape(196,16,32).reshape(196,512).reshape(14,14,512)
out_ts=conv(inp_std_ts).reshape(OH*OW,OC).reshape(-1)
out_dr=conv(inp_std_dr).reshape(OH*OW,OC).reshape(-1)

print("\n=== STANDARD layout recompute ===")
for inlabel,outv in [("from goldin_ts",out_ts),("from goldin_dr",out_dr)]:
    for goldlabel,gold in [("goldout_ts",gout_ts),("goldout_dr",gout_dr)]:
        pos,ms=cmp(outv,gold)
        print(f"  {inlabel} -> {goldlabel}: pos {pos:.2f}%  multiset {ms:.2f}%")

# per-OC correlation diag for the best pairing (ts->ts)
def corrdiag(out,gold):
    o=out.reshape(OH*OW,OC); g=gold.reshape(OH*OW,OC)
    cs=[]
    for c in range(OC):
        a=o[:,c]; b=g[:,c]
        if a.std()>0 and b.std()>0:
            cs.append(np.corrcoef(a,b)[0,1])
    cs=np.array(cs)
    return cs.mean(), (cs>0.99).mean()
mc,frac=corrdiag(out_ts,gout_ts)
print(f"\nts->ts per-OC mean corr {mc:.4f}, frac>0.99 {frac:.3f}")
