#!/usr/bin/env python3
"""Generate the chain-probe artifacts for localizing the e2e all-zero bug.

Emits, from the residual-add accept conditions parsed out of nn2rtl_top.v:
  - tb/nn2rtl_top_probe.vlt        Verilator config making each checkpoint's
                                   data_out / valid_out / consumer-skid in_ready
                                   PUBLIC (read-only) so the C++ harness can
                                   sample them via dut->rootp.
  - tb/probe_capture.inc           C++ snippet: per-checkpoint accept test +
                                   capture; plus the vector decls and the dump
                                   loop. #included by nn2rtl_top_probe_tb.cpp.
  - probe_manifest.json            {name: goldout_module_id} for Python compare.

Checkpoints = stem (conv_196, max_pool2d) + all 16 residual adds. Each add_N's
output stream == the next module's input, and has a per-module contract goldout
(node_add_N), so the first checkpoint whose captured stream != its goldout (or
goes all-zero) localizes the integration bug to within one residual block.
"""
from __future__ import annotations
import json, re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TOP = ROOT / "output/rtl/nn2rtl_top.v"
VLT = ROOT / "tb/nn2rtl_top_probe.vlt"
INC = ROOT / "tb/probe_capture.inc"
MAN = ROOT / "output/reports_integrated/verilator_nn2rtl_top_probe/probe_manifest.json"

ACC = "dut->rootp->nn2rtl_top__DOT__"  # Verilator top-scope public var accessor prefix


def parse_accepts(src_lines: list[str]) -> dict[str, str]:
    """checkpoint module-id -> consuming-skid in_ready wire."""
    out: dict[str, str] = {}
    targets = ["node_conv_196", "node_max_pool2d"] + ["node_add"] + [f"node_add_{k}" for k in range(1, 16)]
    for tgt in targets:
        for i, l in enumerate(src_lines):
            if re.search(rf"\.in_data\(({re.escape(tgt)})_data_out\)", l):
                for j in range(max(0, i - 3), min(len(src_lines), i + 4)):
                    mr = re.search(r"\.in_ready\((\w+)\)", src_lines[j])
                    if mr:
                        out[tgt] = mr.group(1)
                        break
                break
    return out


def main() -> None:
    src = TOP.read_text().splitlines()
    accepts = parse_accepts(src)
    order = ["node_conv_196", "node_max_pool2d", "node_add"] + [f"node_add_{k}" for k in range(1, 16)]
    # unified form: (name, [signals to AND for accept])
    probes = [(p, [f"{p}_valid_out", "spatial_run", accepts[p]]) for p in order if p in accepts]
    # ENGINE-REGION extra probes (stage3-block0): engine input (conv_244, spatial),
    # first engine output (conv_246), spatial expand (conv_248), engine downsample
    # (conv_250). Accept = AND of these explicit signals (parsed from nn2rtl_top.v).
    EXTRA = {
        "node_conv_244": ["node_conv_244_valid_out", "spatial_run", "skid_node_relu_22_ready"],
        "node_conv_246": ["node_conv_246_valid_out", "skid_node_relu_23_ready", "spatial_run"],
        "node_conv_248": ["node_conv_248_valid_out", "node_add_7_skip_in_ready", "spatial_run"],
        "node_conv_250": ["node_conv_250_valid_out", "node_add_7_ready_in", "node_add_7_skip_valid", "spatial_run"],
        # BISECTION of the never-probed late-engine span conv_252..conv_282 (workflow wm9trddo2):
        # conv_252 = first spatial node past the byte-exact frontier; conv_266 = engine d5 midpoint;
        # conv_282 = engine d8 = conv_284's DIRECT producer. Locates onset of the in-chain input corruption.
        "node_conv_252": ["node_conv_252_valid_out", "spatial_run", "skid_node_relu_25_ready"],
        "node_conv_266": ["node_conv_266_valid_out", "spatial_run", "skid_node_relu_32_ready"],
        "node_conv_282": ["node_conv_282_valid_out", "spatial_run", "skid_node_relu_40_ready"],
    }
    for name, sigs in EXTRA.items():
        probes.append((name, sigs))
    missing = [p for p in order if p not in accepts]
    if missing:
        print(f"[warn] no accept parsed for: {missing}")

    # ---- .vlt ----
    # base + DYNAMIC engine-read instrumentation vars (captured custom in the tb,
    # NOT via PROBE_CAPTURE macros): the engine's actual in-chain act_in reads.
    pub_vars = {"spatial_run", "engine_act_in_rd_addr", "engine_act_in_rd_en",
                "engine_act_in_rd_data", "sched_dispatch_idx", "engine_busy",
                # config-write sequence (scheduler -> engine AXI-Lite) + weight reads
                "sched_axil_awaddr", "sched_axil_awvalid", "sched_axil_awready",
                "sched_axil_wdata", "sched_axil_wvalid", "sched_axil_wready",
                "engine_weight_rd_addr", "engine_weight_rd_en", "engine_weight_rd_data",
                # engine RAW output (pre-bridge) to isolate compute vs output-routing
                "engine_act_out_wr_addr", "engine_act_out_wr_en", "engine_act_out_wr_data"}
    for name, sigs in probes:
        pub_vars |= {f"{name}_data_out"} | set(sigs)
    # TB-HARDCODED taps (tb/nn2rtl_top_probe_tb.cpp CAPCONV + conv_262/add9): these
    # vars are read directly by the handwritten TB and MUST stay public even though
    # they're not in the auto probe set. (Was previously a manual .vlt addition; folding
    # it in here so re-running this generator can't drop them.)
    pub_vars |= {
        "node_conv_198_valid_out", "skid_node_relu_1_ready", "node_conv_198_data_out",
        "node_conv_200_valid_out", "skid_node_relu_2_ready", "node_conv_200_data_out",
        "node_conv_206_valid_out", "skid_node_relu_4_ready", "node_conv_206_data_out",
        "node_conv_212_valid_out", "skid_node_relu_7_ready", "node_conv_212_data_out",
        "node_conv_284_valid_out", "skid_node_relu_41_ready", "node_conv_284_data_out",
        "node_relu_3_valid_out", "node_relu_3_ready_out_combined", "node_relu_3_data_out",
        "node_conv_262_valid_out", "node_conv_262_data_out",
        "node_add_9_skip_valid", "node_add_9_ready_in", "node_add_9_skip_data",
    }
    with VLT.open("w") as f:
        f.write("`verilator_config\n")
        for v in sorted(pub_vars):
            f.write(f'public_flat_rd -module "nn2rtl_top" -var "{v}"\n')
    print(f"[written] {VLT}  ({len(pub_vars)} public vars)")

    # ---- C++ capture include ----
    with INC.open("w") as f:
        f.write("// AUTO-GENERATED by scripts/gen_chain_probe.py — do not edit.\n")
        f.write("// Provides: PROBE_DECLS, PROBE_CAPTURE(dut), PROBE_DUMP(dir)\n\n")
        # vector decls
        f.write("#define PROBE_DECLS \\\n")
        for name, _ in probes:
            f.write(f"  std::vector<std::array<uint32_t,8>> cap_{name}; \\\n")
        f.write("  /* end */\n\n")
        # capture (called every cycle, after eval)
        f.write("#define PROBE_CAPTURE(dut) do { \\\n")
        for name, sigs in probes:
            cond = " && ".join(f"({ACC}{s})" for s in sigs)
            d = f"{ACC}{name}_data_out"
            f.write(f"  if ({cond}) {{ std::array<uint32_t,8> w; "
                    f"for(int _i=0;_i<8;_i++) w[_i]=({d})[_i]; cap_{name}.push_back(w); }} \\\n")
        f.write("  } while(0)\n\n")
        # dump
        f.write("#define PROBE_DUMP(dir) do { \\\n")
        for name, _ in probes:
            f.write(f"  {{ std::string p=std::string(dir)+\"/probe_{name}.bin\"; std::ofstream o(p,std::ios::binary); "
                    f"for(auto&w:cap_{name}) o.write((const char*)w.data(),32); "
                    f"std::printf(\"[probe] {name}: %zu beats -> %s\\n\", cap_{name}.size(), p.c_str()); }} \\\n")
        f.write("  } while(0)\n")
    print(f"[written] {INC}  ({len(probes)} probes)")

    # ---- manifest ----
    MAN.parent.mkdir(parents=True, exist_ok=True)
    MAN.write_text(json.dumps({name: name for name, _ in probes}, indent=2))
    print(f"[written] {MAN}")
    for name, sigs in probes:
        print(f"   {name:16s} accept= {' & '.join(sigs)}")


if __name__ == "__main__":
    main()
