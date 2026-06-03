#!/usr/bin/env python3
"""SOUND FIX for the retile-bridge producer/consumer duplication deadlock.

ROOT CAUSE (workflow wf_9ee1675f, high-confidence, adversarially verified):
Every ENABLE_BACKPRESSURE elastic producer (relu n4_*, depthwise node_conv_*) that
feeds a retile_gather/retile_scatter HOLDS its valid_out with stale data while its
out_ready_in is low. But its out_ready_in was wired to a resource TWO hops downstream
(the loader's in_ready / the next conv's ready_in), NOT to the bridge's own per-beat
intake accept. The bridge latches on its OWN condition do_write = valid_in & wsel_empty
(retile_bridge.v:184/320), decoupled from out_ready_in. So whenever the far resource is
momentarily not-ready but the bridge has a free write-buffer slot, the bridge greedily
re-writes the SAME held beat into successive tile slots g_idx,g_idx+1,... -> PHANTOM
tiles -> the loader fills to its TOTAL_BRAM_WORDS cap from a handful of REAL positions
-> loaded=1 latches in_ready=0 -> whole chain back-pressures -> engine_output_fifo can't
drain -> scheduler wedged in S_WAIT_DRAIN forever (the observed dispatch-21 deadlock).

THE FIX (provably 1:1, no dup / no lost beat): set each such producer's out_ready_in to
the bridge's per-beat accept = wsel_empty (NOT ready_out=write_free, which is unsound:
reachable state full0=0,full1=1,wsel=1 has write_free=1 but wsel_empty=0 -> would DROP a
beat). And DROP the '& spatial_run' term (the bridge intake do_write does not use
spatial_run, so the producer must not either, else they re-desync when spatial_run drops).
Then: producer advances (valid_out & out_ready_in) == bridge writes (valid_in & wsel_empty)
on exactly the same cycles -> 18 distinct tiles/pixel, 1 OUT beat/pixel, loader fills only
after the REAL number of positions.

Implementation:
  1) retile_bridge.v: expose `output wire wr_accept = wsel_empty;` on BOTH modules.
  2) nn2rtl_top_engine.v: for every retile_gather/retile_scatter instance u_br_*,
     - add .wr_accept(<wire>_wr_accept), declare that wire,
     - re-point its PRODUCER (auto-derived from the bridge's .valid_in(<prod>_valid_out))
       out_ready_in to <wire>_wr_accept (no spatial_run).
On-disk surgical patch ONLY; never regenerate the top. Idempotent (asserts if already done).
"""
import re, pathlib, sys

TOP = pathlib.Path("output/mobilenet-v2/rtl/nn2rtl_top_engine.v")
BR  = pathlib.Path("rtl_library/retile_bridge.v")

def patch_bridge():
    b = BR.read_text()
    if "assign wr_accept = wsel_empty;" in b:
        print("  retile_bridge.v already has wr_accept -- skipping module patch")
        return
    n1 = b.count("output wire                 stall_out")
    assert n1 == 2, "expected 2 'output wire stall_out' (gather+scatter), got %d" % n1
    b = b.replace(
        "output wire                 stall_out",
        "output wire                 wr_accept,   // per-beat intake accept (= wsel_empty)\n    output wire                 stall_out")
    n2 = b.count("wire wsel_empty = wsel ? ~full1 : ~full0;")
    assert n2 == 2, "expected 2 wsel_empty decls, got %d" % n2
    b = b.replace(
        "wire wsel_empty = wsel ? ~full1 : ~full0;",
        "wire wsel_empty = wsel ? ~full1 : ~full0;\n    assign wr_accept = wsel_empty;")
    BR.write_text(b)
    print("  retile_bridge.v: added wr_accept(=wsel_empty) to retile_gather + retile_scatter")

def patch_top():
    s = TOP.read_text()
    # find every retile bridge instance + its producer (from .valid_in(<prod>_valid_out))
    pairs = []
    # All u_br_* instances ARE retile_gather/scatter. Match by instance name + body.
    # Use DOTALL .*? (NOT [^;]): port-comment text contains ';' e.g. "always-accept;".
    for m in re.finditer(r"(u_br_\w+)\s*\((.*?)\);", s, re.DOTALL):
        inst, body = m.group(1), m.group(2)
        if "valid_in" not in body:
            continue
        vm = re.search(r"\.valid_in\(\s*(\w+?)_valid_out\b", body)
        assert vm, "no '<prod>_valid_out' .valid_in in %s" % inst
        prod = vm.group(1)        # n4_23 / node_conv_878 / ...
        wire = inst[2:]           # u_br_878 -> br_878
        pairs.append((inst, wire, prod))
    assert pairs, "no retile bridge instances found!"
    print("  found %d retile bridge instances" % len(pairs))

    changed_inst = changed_wire = changed_prod = 0
    for inst, wire, prod in pairs:
        wa = "%s_wr_accept" % wire
        # (a) add .wr_accept(...) to the bridge instance (after its .stall_out(...))
        so = ".stall_out(%s_stall_out)" % wire
        if wa not in s:   # only if not already wired
            assert so in s, "stall_out conn for %s not found" % wire
            s = s.replace(so, so + ",\n        .wr_accept(%s)" % wa, 1)
            changed_inst += 1
        # (b) declare the wire (after the matching 'wire <wire>_stall_out;' decl)
        decl_re = re.compile(r"(wire\s+%s_stall_out\s*;)" % re.escape(wire))
        if ("wire %s;" % wa) not in s:
            dm = decl_re.search(s)
            assert dm, "stall_out wire decl for %s not found" % wire
            s = s[:dm.end()] + ("\n    wire %s;" % wa) + s[dm.end():]
            changed_wire += 1
        # (c) re-point the producer's out_ready_in -> <wire>_wr_accept (drop spatial_run)
        pm = re.search(r"(\bu_%s\s*\()(.*?)(\);)" % re.escape(prod), s, re.DOTALL)
        assert pm, "producer instance u_%s not found" % prod
        block = pm.group(2)
        newblock, k = re.subn(r"\.out_ready_in\([^)]*\)", ".out_ready_in(%s)" % wa, block, count=1)
        assert k == 1, "no .out_ready_in in producer u_%s" % prod
        if newblock != block:
            s = s[:pm.start(2)] + newblock + s[pm.end(2):]
            changed_prod += 1
        print("    %-14s <- producer u_%-16s out_ready_in -> %s" % (inst, prod, wa))

    TOP.write_text(s)
    print("  top: +%d .wr_accept conns, +%d wire decls, %d producers re-gated"
          % (changed_inst, changed_wire, changed_prod))

def main():
    patch_bridge()
    patch_top()
    print("DONE.")

if __name__ == "__main__":
    main()
