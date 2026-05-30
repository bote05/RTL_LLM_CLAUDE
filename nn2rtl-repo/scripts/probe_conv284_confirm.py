import struct
from pathlib import Path
import numpy as np

ROOT = Path("c:/Users/User/Desktop/RTL_LLM_CLAUDE/nn2rtl-repo")
IC=OC=512; IH=IW=14; OH=OW=7; KH=KW=3; PH=PW=1; SH=SW=2

def loadgold(path):
    g=Path(path).read_bytes()
    magic,nv,_,spv,bps=struct.unpack("<4sIIII",g[:20])
    return np.frombuffer(g[20:20+spv*bps],dtype=np.int8).astype(np.int64)

ts=ROOT/"output/goldens/contracts/node_conv_284_tiled-streaming_conv2d_512x512x3x3_s14x14_st2x2_p1x1_d1x1_g1_mp4_i"
gin=loadgold(ts/"node_conv_284.goldin")
gout=loadgold(ts/"node_conv_284.goldout")

wtxt=(ROOT/"output/weights/node_conv_284_weights.hex").read_text().split()
W=np.array([int(x,16) for x in wtxt],dtype=np.int64); W=np.where(W>127,W-256,W); W=W.reshape(OC,IC,KH,KW)
btxt=(ROOT/"output/weights/node_conv_284_bias.hex").read_text().split()
bias=np.array([int(x,16) for x in btxt],dtype=np.int64); bw=len(btxt[0])*4
bias=np.where(bias>=(1<<(bw-1)),bias-(1<<bw),bias)
sv=np.array([int(x,16) for x in (ROOT/"output/weights/node_conv_284_scale.mem").read_text().split()],dtype=np.int64)
mult=sv&0xFFFF; shift=(sv>>16)&0x3F

inp=gin.reshape(196,16,32).reshape(196,512).reshape(14,14,512)
pad=np.zeros((IH+2,IW+2,IC),dtype=np.int64); pad[1:1+IH,1:1+IW]=inp
acc=np.zeros((OH,OW,OC),dtype=np.int64)
for kh in range(3):
    for kw in range(3):
        patch=pad[kh:kh+OH*SH:SH, kw:kw+OW*SW:SW, :]
        acc += np.tensordot(patch,W[:,:,kh,kw],axes=([2],[1]))
biased=acc+bias[None,None,:]; scaled=biased*mult[None,None,:]
rnd=np.where(shift>0,(1<<np.maximum(shift-1,0)),0)[None,None,:]
v=(scaled+rnd)>>shift[None,None,:]
out=np.clip(v,-128,127).astype(np.int64).reshape(OH*OW,OC).reshape(-1)

print("recompute(goldin_ts std-layout) vs goldout_ts: pos%",
      round(100*(out!=gout).mean(),4),"multiset%",round(100*(np.sort(out)!=np.sort(gout)).mean(),4))
print("exact equal arrays?", np.array_equal(out,gout))

# RTL relationship
pdir=ROOT/"output/reports_integrated/verilator_nn2rtl_top_probe"
rtl=np.frombuffer((pdir/"probe_node_conv_284.bin").read_bytes(),dtype=np.int8).astype(np.int64)
print("RTL vs goldout_ts pos%", round(100*(rtl!=gout).mean(),3),"multiset%",round(100*(np.sort(rtl)!=np.sort(gout)).mean(),3))
print("RTL vs recompute pos%", round(100*(rtl!=out).mean(),3))
diff=rtl-gout
print("RTL-goldout diff: nonzero%",round(100*(diff!=0).mean(),3),"max abs", int(np.abs(diff).max()),"mean abs", round(np.abs(diff).mean(),3))
