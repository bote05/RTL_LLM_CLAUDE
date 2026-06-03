#!/usr/bin/env python3
"""Set OC/OUT_KIND/POSITIONS on the HIGH-OC engine_output_bridge instances of the
engine top, so the rewritten 3-mode bridge delivers the engine's ceil(OC/256)
beats/position correctly:
  OUT_KIND=2 (flat-gather): OC=384 blocks -> 1 contiguous 3072b beat/pos (DATA_W=3072).
  OUT_KIND=1 (tiled-256):   OC=576/960/1280 -> ceil(OC/32) real 256b tiles/pos (DATA_W=256).
Low-OC slots keep the default OUT_KIND=0 (byte-identical legacy). SLOT32 (320, ldr33
redundant) left legacy for now. Idempotent + asserts each insert applies once.
"""
import re, pathlib
TOP = pathlib.Path("output/mobilenet-v2/rtl/nn2rtl_top_engine.v")
# nodeid -> (OC, OUT_KIND, POSITIONS)
P = {
    852:(384,2,196), 858:(384,2,196), 864:(384,2,196), 870:(384,2,196),   # SLOT13/15/17/19
    876:(576,1,196), 882:(576,1,196), 888:(576,1,196),                     # SLOT21/23/25
    894:(960,1,49),  900:(960,1,49),  906:(960,1,49),                      # SLOT27/29/31
    912:(1280,1,49),                                                       # SLOT33
}
def main():
    src = TOP.read_text(); n=0
    for node,(oc,kind,pos) in P.items():
        # match the instance's NUM_DISPATCHES line immediately before ") u_engine_out_node_conv_<node> ("
        pat = re.compile(r"(\.NUM_DISPATCHES\(\d+\))(\s*\)\s*u_engine_out_node_conv_%d\s*\()" % node)
        if (".OC(%d)" % oc) in src and ("u_engine_out_node_conv_%d" % node) in src and \
           re.search(r"\.OUT_KIND\(\d+\)\s*\)\s*u_engine_out_node_conv_%d" % node, src):
            print(f"  {node}: already has params (skip)"); continue
        m = pat.search(src)
        assert m, f"NUM_DISPATCHES anchor before u_engine_out_node_conv_{node} not found"
        ins = f"{m.group(1)},\n        .OC({oc}), .OUT_KIND({kind}), .POSITIONS({pos}){m.group(2)}"
        src = src[:m.start()] + ins + src[m.end():]
        n += 1
        print(f"  u_engine_out_node_conv_{node}: OC={oc} OUT_KIND={kind} POSITIONS={pos}")
    TOP.write_text(src)
    print(f"\nApplied {n} high-OC bridge param blocks.")
if __name__ == "__main__":
    main()
