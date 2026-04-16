# Sky130 standard cell library

The Yosys synth flow in [mcp/tools.ts](../../mcp/tools.ts) maps to Sky130
(`sky130_fd_sc_hd__tt_025C_1v80.lib`) — a free, Apache 2.0 standard cell
library from Google / SkyWater. The file is ~13 MB of cell timing/power
data, so we do not commit it; run the download script once after cloning.

```bash
# from repo root
bash vendor/sky130/download.sh
```

## Source

- OpenROAD-flow-scripts mirror: <https://github.com/The-OpenROAD-Project/OpenROAD-flow-scripts/raw/master/flow/platforms/sky130hd/lib/sky130_fd_sc_hd__tt_025C_1v80.lib>
- SHA-256: `ec0e1067a35c8bf20b11e58d1e8ac53326067e4dac84a125cc1b917a3518d0d9`
- License: Apache 2.0
- Process: 130 nm, typical-typical corner, 25°C, 1.80 V

## Why not commit it?

PDK blobs are third-party data, not source. Vendoring them would bloat every
clone and every branch switch; pinning keeps `run_yosys` reproducible without
the checkout cost. If the OpenROAD mirror ever disappears, the canonical
source is <https://github.com/google/skywater-pdk>.
