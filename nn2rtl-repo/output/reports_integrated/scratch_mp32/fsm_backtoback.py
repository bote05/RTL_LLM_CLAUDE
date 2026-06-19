#!/usr/bin/env python3
"""Back-to-back / re-arm test: does conv_datapath_mp_k cleanly accept a NEW
start_mac after a frame, at MP=32? And confirm the per-frame cycle count formula
(OUTPUT timing) vs MP=16. This probes whether the datapath itself can stall or
double-fire across consecutive frames (the thing the chain hammers it with)."""
from __future__ import annotations
import random
import importlib.util, pathlib

spec = importlib.util.spec_from_file_location(
    "fsm_model", str(pathlib.Path(__file__).with_name("fsm_model.py")))
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)

def run_multi_frame(IC, OC, KH, KW, MP, MP_K, n_frames=4, gap=0, seed=7):
    """Drive start_mac for frame 0 at cyc=0, and re-pulse start_mac `gap` cycles
    after each return to IDLE. Count valid_out pulses; must equal n_frames."""
    random.seed(seed)
    K_TOTAL = IC*KH*KW
    w_oc_k = [[random.randint(-8,7) for _ in range(K_TOTAL)] for _ in range(OC)]
    biases = [random.randint(-(1<<20),(1<<20)-1) for _ in range(OC)]
    scale_rom = [(((random.randint(0,23)&0x3F)<<16)|(random.randint(1,32767)&0xFFFF)) for _ in range(OC)]
    rom = m.pack_weights_wide(w_oc_k, OC, K_TOTAL, MP, MP_K)
    windows = [[random.randint(-128,127) for _ in range(K_TOTAL)] for _ in range(n_frames)]

    dp = m.ConvDatapath(IC, OC, KH, KW, MP, MP_K, rom, biases, scale_rom)
    OC_PASSES = dp.OC_PASSES; K_GROUPS = dp.K_GROUPS

    # inline a multi-frame driver mirroring ConvDatapath.run but with re-arming
    ST_IDLE,ST_MAC,ST_BIAS,ST_SCALE,ST_OUTPUT = range(5)
    state=ST_IDLE; valid_out=0; data_out=[0]*OC; k_group=0; oc_group=0
    mvq1=mvq2=0; mog1=mog2=0; mdi=0
    acc=[0]*MP; biased=[0]*MP; scaled=[0]*MP; partial=[0]*MP
    wwq=0; tapq=[0]*MP_K
    frames_done=0; cur_window=windows[0]
    pending_start_at=0  # cycle to pulse start
    valid_cycles=[]; outs=[]; idle_seen_for_next=False
    cyc=0; MAXC=500000
    next_window_idx=1
    while cyc<MAXC and frames_done<n_frames:
        start_mac = (cyc==pending_start_at and pending_start_at is not None)
        wf = m.build_window_flat(cur_window, IC, KH, KW)
        # comb sum
        sum_lane=[0]*MP
        for lane in range(MP):
            s=0
            for kp in range(MP_K):
                wn=m.sext((wwq>>((lane*MP_K+kp)*4))&0xF,4)
                s+=m.sext((wn*tapq[kp])&0xFFFF,16)
            sum_lane[lane]=m.sext(s&((1<<dp.TREE_W)-1),dp.TREE_W)
        addr=oc_group*K_GROUPS+k_group
        n_wwq=rom[addr]; n_tapq=[dp.tap_at(k_group*MP_K+i,wf) for i in range(MP_K)]
        # defaults
        ns=state; nvo=0; ndo=list(data_out); nkg=k_group; nog=oc_group
        nmvq1=mvq1; nmvq2=mvq1; nmog1=mog1; nmog2=mog1; nmdi=mdi
        nacc=list(acc); nbiased=list(biased); nscaled=list(scaled); npartial=list(sum_lane)
        if mvq2:
            for lane in range(MP):
                if mog2*MP+lane<OC:
                    nacc[lane]=m.sext((acc[lane]+partial[lane])&((1<<dp.ACC_W)-1),dp.ACC_W)
        if state==ST_IDLE:
            if start_mac:
                ns=ST_MAC; nkg=0; nog=0; nmvq1=0; nmvq2=0; nmdi=0
                for lane in range(MP): nacc[lane]=0
        elif state==ST_MAC:
            if mdi:
                nmvq1=0
                if (not mvq1) and (not mvq2): nmdi=0; ns=ST_BIAS
            else:
                nmog1=oc_group; nmvq1=1
                if k_group==K_GROUPS-1: nmdi=1
                else: nkg=k_group+1
        elif state==ST_BIAS:
            for lane in range(MP):
                boc=oc_group*MP+lane
                if boc<OC:
                    v=m.sext(acc[lane],dp.ACC_W)+m.sext(biases[boc]&0xFFFFFFFF,32)
                    nbiased[lane]=m.sext(v&((1<<dp.BIASED_W)-1),dp.BIASED_W)
                else: nbiased[lane]=0
            ns=ST_SCALE
        elif state==ST_SCALE:
            for lane in range(MP):
                soc=oc_group*MP+lane
                if soc<OC:
                    mult=scale_rom[soc]&0xFFFF
                    v=m.sext(biased[lane],dp.BIASED_W)*mult
                    nscaled[lane]=m.sext(v&((1<<dp.SCALED_W)-1),dp.SCALED_W)
                else: nscaled[lane]=0
            ns=ST_OUTPUT
        elif state==ST_OUTPUT:
            for lane in range(MP):
                ooc=oc_group*MP+lane
                if ooc<OC:
                    sh=(scale_rom[ooc]>>16)&0x3F
                    orr=0 if sh==0 else (1<<(sh-1))
                    vt=(m.sext(scaled[lane],dp.SCALED_W)+orr)>>sh
                    ndo[ooc]=127 if vt>127 else (-128 if vt<-128 else m.sext(vt&0xFF,8))
            if oc_group==OC_PASSES-1:
                nvo=1; ns=ST_IDLE
            else:
                nog=oc_group+1; nkg=0
                for lane in range(MP): nacc[lane]=0
                ns=ST_MAC
        else:
            ns=ST_IDLE
        # commit
        state=ns; valid_out=nvo; data_out=ndo; k_group=nkg; oc_group=nog
        mvq1=nmvq1; mvq2=nmvq2; mog1=nmog1; mog2=nmog2; mdi=nmdi
        acc=nacc; biased=nbiased; scaled=nscaled; partial=npartial; wwq=n_wwq; tapq=n_tapq
        if valid_out:
            valid_cycles.append(cyc); outs.append(list(data_out)); frames_done+=1
            # schedule next frame's start `gap` cycles after we return to IDLE.
            # We will detect IDLE next and re-pulse.
            pending_start_at=None
        if pending_start_at is None and state==ST_IDLE and frames_done<n_frames and frames_done>0:
            pending_start_at=cyc+1+gap
            cur_window=windows[next_window_idx]; next_window_idx+=1
        cyc+=1
    # references
    refs=[m.golden_ref(windows[i],w_oc_k,biases,scale_rom,OC,K_TOTAL) for i in range(frames_done)]
    return {"frames_done":frames_done,"valid_cycles":valid_cycles,
            "outs_match_ref":[outs[i]==refs[i] for i in range(frames_done)],
            "intervals":[valid_cycles[i+1]-valid_cycles[i] for i in range(len(valid_cycles)-1)]}

def main():
    print("=== Multi-frame re-arm test (datapath accepts back-to-back start_mac) ===\n")
    cases=[(64,64,3,3,9,"conv_200 3x3 OC=64"),
           (64,256,1,1,8,"conv_202 1x1 OC=256")]
    for gap in (0,1,3):
        print(f"--- start re-pulsed gap={gap} cyc after IDLE ---")
        for IC,OC,KH,KW,MP_K,label in cases:
            for MP in (16,32):
                r=run_multi_frame(IC,OC,KH,KW,MP,MP_K,n_frames=4,gap=gap)
                ok = r["frames_done"]==4 and all(r["outs_match_ref"])
                print(f"  {label} MP={MP}: frames={r['frames_done']}/4 "
                      f"valid@={r['valid_cycles']} interval={r['intervals']} "
                      f"all_ref_match={all(r['outs_match_ref'])} -> {'OK' if ok else 'FAIL'}")
        print()

if __name__=="__main__":
    main()
