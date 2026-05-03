# 09 - DRAM-backed weights contract

This is contract guidance, not a passing RTL reference. Use it when
`contract_id == "dram-backed-weights"`.

## Interface

The top level includes the seven base activation-stream ports from
`01_context.md` plus the AXI read-channel ports declared in
`contracts/dram-backed-weights/metadata.json`:

- `weights_arvalid`, `weights_arready`, `weights_araddr`, `weights_arlen`
- `weights_rvalid`, `weights_rready`, `weights_rdata`, `weights_rlast`

Do not collapse this contract back to the seven-port flat-bus interface.
Do not store the full `OC*K_TOTAL` weight tensor on chip.

## Exact-latency rule

The verifier measures `pipeline_latency_cycles` from the first accepted
`valid_in` beat. Therefore any memory warm-up needed for output-channel pass 0
must complete before the module raises `ready_in` for the first input beat.

Required sequence:

1. After reset, keep `ready_in = 0` while prefetching pass-0 weights.
2. Issue AXI reads for the pass-0 weight window.
3. Cache only the active weight window, not the full layer tensor.
4. Raise `ready_in = 1` only after pass-0 weights are available.
5. Once the first input beat is accepted, the first `valid_out` must occur at
   exactly `pipeline_latency_cycles`.

Do not add first-pass DRAM latency after `first_valid_in`; that violates the
LayerIR timing contract.

## Weight window formulas

Use formulas, not module-specific constants:

```
K_TOTAL        = (IC / groups) * KH * KW
OC_PASSES      = ceil(OC / MP)
PASS_WEIGHTS   = MP * K_TOTAL
PASS_BYTES     = PASS_WEIGHTS
AXI_BYTES      = 8
BEATS_PER_PASS = ceil(PASS_BYTES / AXI_BYTES)
pass_base_addr = oc_pass * PASS_BYTES
```

Each 64-bit AXI data beat carries eight INT8 weights in little-endian byte
order. Address units are bytes into `weights_path`.

## Progress rule

For an exact-latency design, overlap next-pass weight fetch with current-pass
compute whenever possible:

- compute pass `N` from cache A
- prefetch pass `N+1` into cache B
- swap caches at the pass boundary

If a simpler first version fetches pass `N+1` only after pass `N` finishes, it
must either still meet the declared latency or fail deterministically. Do not
hide the added cycles by changing `pipeline_latency_cycles`.
