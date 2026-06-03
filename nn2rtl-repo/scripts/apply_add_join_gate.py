#!/usr/bin/env python3
"""
apply_add_join_gate.py  (2026-06-02)

Fix the residual-add two-input-JOIN DESYNC in the MobileNetV2 engine top.

Each of the 10 residual adds (node_add_N) consumes a paired beat:
    valid_in = <lhs>_valid_out & node_add_N_skip_valid & spatial_run
where <lhs> is the engine-output bridge for node_conv_LHS and the skip (rhs) is
buffered in u_skip_node_add_N (a FIFO holding the EARLIER residual block's output).

BUG: the two sources popped INDEPENDENTLY whenever the add was merely "ready":
    skip_fifo .out_ready(node_add_N_ready_in)                 # not gated on lhs valid
    bridge    .ready_out((node_add_N_ready_in & spatial_run)) # not gated on skip valid
Because the skip is produced (early dispatch) long BEFORE the lhs (later dispatch),
the skip FIFO drained into the void while the add had no lhs -> by the time the lhs
streamed, the skip was empty -> the add never produced -> the next dispatch's input
loader starved -> S_WAIT_LOAD deadlock (observed at dispatch 5 / node_add_198).
PROVEN via probe: u_skip_node_add_198 wr=3136 rd=3136 empty=1 while ldr5 word_count=0.

FIX (standard elastic two-input join): pop each source ONLY when BOTH are present and
the add is ready, so the streams stay in lock-step (and the early skip stays buffered
until its lhs arrives):
    skip_fifo .out_ready(node_add_N_ready_in & <lhs>_valid_out & spatial_run)
    bridge    .ready_out((node_add_N_ready_in & node_add_N_skip_valid & spatial_run))

Byte-exact: pairing order is preserved (both streams are position-ordered 0..K-1 and
pop together), so the add sees (lhs[i], skip[i]) exactly as before -- only the *timing*
of the pop is corrected. Idempotent + asserts each edit applies exactly once.
"""
import sys, pathlib

TOP = pathlib.Path("output/mobilenet-v2/rtl/nn2rtl_top_engine.v")

# node_add_N -> node_conv_LHS (the lhs / main-path producer feeding that add)
ADD_LHS = {
    198: 826, 336: 838, 408: 844, 546: 856, 618: 862,
    690: 868, 828: 880, 900: 886, 1038: 898, 1110: 904,
}

def main():
    src = TOP.read_text()
    orig = src
    n_edits = 0
    for add, lhs in ADD_LHS.items():
        # 1) skip_fifo out_ready: gate on lhs valid
        old_skip = f".out_ready(node_add_{add}_ready_in)"
        new_skip = f".out_ready(node_add_{add}_ready_in & node_conv_{lhs}_valid_out & spatial_run)"
        # 2) bridge ready_out: gate on skip valid
        old_brdg = f".ready_out((node_add_{add}_ready_in & spatial_run)),"
        new_brdg = f".ready_out((node_add_{add}_ready_in & node_add_{add}_skip_valid & spatial_run)),"

        if new_skip in src and new_brdg in src:
            print(f"  node_add_{add}: already gated (idempotent skip)")
            continue

        c_skip = src.count(old_skip)
        c_brdg = src.count(old_brdg)
        assert c_skip == 1, f"node_add_{add}: skip out_ready pattern count={c_skip} (expected 1): {old_skip}"
        assert c_brdg == 1, f"node_add_{add}: bridge ready_out pattern count={c_brdg} (expected 1): {old_brdg}"
        src = src.replace(old_skip, new_skip)
        src = src.replace(old_brdg, new_brdg)
        n_edits += 2
        print(f"  node_add_{add}: gated skip on node_conv_{lhs}_valid_out + bridge on skip_valid")

    if src == orig:
        print("No changes (already applied).")
        return
    TOP.write_text(src)
    print(f"\nApplied {n_edits} edits to {TOP}")

if __name__ == "__main__":
    main()
