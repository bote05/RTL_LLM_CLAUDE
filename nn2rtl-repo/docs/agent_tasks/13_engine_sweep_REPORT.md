# Engine Sweep Report

- Generator: `scripts/run_engine_sweep_all.sh` -> `scripts/engine_sweep_driver.py`
- Total dispatches run: 2
- PASS: 2
- FAIL: 0
- Wall clock: 1572.8s (26.21 min)
- Total engine cycles across all dispatches: 658078

## Per-dispatch results

| idx | module_id | IC | OC | KxK | S | PxP | IHxIW | OHxOW | cycles | status | mismatches | max_err |
|----:|-----------|---:|---:|:---:|:-:|:---:|:-----:|:-----:|------:|:------:|----------:|--------:|
| 0 | node_conv_246 | 256 | 256 | 3x3 | 2 | 1x1 | 28x28 | 14x14 | 453744 | PASS | 0 | 0 |
| 13 | node_conv_300 | 512 | 2048 | 1x1 | 1 | 0x0 | 7x7 | 7x7 | 204334 | PASS | 0 | 0 |
