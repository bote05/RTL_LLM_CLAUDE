#!/usr/bin/env python3
"""STEP 1 of the structural-localization plan: characterize the relu_48 per-channel
error STRUCTURE (from the trustworthy m_axis output, not confounded probes).
Is the errored-channel set contiguous / periodic / boundary-aligned (sharp fingerprint)
or broad? Checks alignment to tile(32), MAC/oc_pass(256), half(512), bank boundaries."""
import struct, glob
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parent.parent
OUT_SCALE = 0.719477255513349
def gapvec(arr):
    a=arr[:100352].astype(np.float32).reshape(49,64,32).reshape(49,2048)
    return a.mean(0)*OUT_SCALE
rtl = gapvec(np.frombuffer(Path(ROOT/"output/reports_integrated/relu48_native_dump.bin").read_bytes(),dtype=np.int8))
r=Path(glob.glob(str(ROOT/"output/goldens/contracts/node_relu_48_*/node_relu_48.goldout"))[0]).read_bytes()
_,nv,_,spv,bps=struct.unpack("<4sIIII",r[:20])
gold=gapvec(np.frombuffer(r[20:20+spv*bps],dtype=np.int8))
err=np.abs(rtl-gold)
THR=0.1
bad=np.where(err>THR)[0]
print(f"channels with |GAP err|>{THR}: {len(bad)}/2048 ({100*len(bad)/2048:.1f}%)  max={err.max():.2f}@ch{int(err.argmax())}")
print(f"bad-channel range: [{bad.min()},{bad.max()}]  span={bad.max()-bad.min()}  (contiguous if span~=count)")
# contiguity: largest gap between consecutive bad channels
if len(bad)>1:
    gaps=np.diff(bad); print(f"  consecutive-gap: median={int(np.median(gaps))} max={int(gaps.max())}  (1=contiguous, big=clustered)")
# boundary alignment: are bad channels concentrated within specific blocks of size B?
for B in [32,64,128,256,512,1024]:
    blocks=np.zeros(2048//B)
    for c in bad: blocks[c//B]+=1
    occ=int((blocks>0).sum()); tot=2048//B
    # is the error CONCENTRATED in few blocks of this size? (sharp if few blocks hold most)
    frac_top=blocks.max()/max(len(bad),1)
    print(f"  block={B:4d}: bad spread over {occ}/{tot} blocks; busiest block holds {int(blocks.max())} ({100*frac_top:.0f}% of bad)")
# periodicity within tile (is error at a specific byte-lane within the 32-ch tile?)
lane=np.zeros(32)
for c in bad: lane[c%32]+=1
print(f"  per-tile-lane (c%32) histogram of bad: max-lane={int(lane.argmax())} holds {int(lane.max())} (uniform~{len(bad)/32:.0f}/lane => not lane-specific)")
# top errored channels
ti=np.argsort(err)[::-1][:16]
print(f"  top-16 errored channels: {ti.tolist()}")
print(f"  their oc_pass(/256): {sorted(set((ti//256).tolist()))}  tile(/32): {sorted(set((ti//32).tolist()))[:10]}")
