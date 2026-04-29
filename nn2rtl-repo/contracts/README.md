# Contract Infrastructure

Each subdirectory is a self-contained contract family used by both normal and
self-improving pipeline runs.

Required files per contract:

- `metadata.json` declares the contract name, complexity rank, interface
  signals, fit constraints, supported ops, dependencies, and docs.
- `testbench.cpp` is the Verilator template selected by the orchestrator.
- `golden.py` adapts logical golden vectors into the contract's stream format.
- `latency.ts` documents and checks the contract latency model.

The SDK registry in `sdk/contracts.ts` loads `metadata.json` and selects the
testbench template from the layer's `contract_id` tag. Missing `contract_id`
means `flat-bus`.
