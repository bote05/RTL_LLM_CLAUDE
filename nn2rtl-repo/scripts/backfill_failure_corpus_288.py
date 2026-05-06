"""
One-off back-fill of output/failure_corpus/visible/node_conv_288/ from prior
run_log archives. Mirrors persistFailureCorpusAttempt() in sdk/orchestrate.ts:
per-failure dir + node_conv_288.v + failure.json, plus appends to
visible/index.jsonl.

Reads:
  output/reports/run_log.jsonl*
  output/layer_ir.json (for the base LayerIR; we apply the dram-backed contract
    in-place since every prior attempt walked to that contract)
Writes:
  output/failure_corpus/visible/node_conv_288/<id>/{node_conv_288.v, failure.json}
  output/failure_corpus/visible/index.jsonl (append)
"""
import json
import glob
import os
import re
from pathlib import Path

CORPUS_VIS = Path("output/failure_corpus/visible/node_conv_288")
INDEX = Path("output/failure_corpus/visible/index.jsonl")
APPLIED_LAYER_PATH = Path("scripts/_applied_layer_288.json")

# Cached applied LayerIR (captured from the most recent
# contract_selected_after_skipping_flagged event). The orchestrator persists
# the contract-applied layer (contract_id, io_mode, channel_tile, beat-width
# input/output_width_bits, contract_params); we want corpus entries keyed off
# the same applied layer so spec_hash etc. match a future
# moduleContractKey(layer) lookup byte-for-byte.
APPLIED_LAYER = json.loads(APPLIED_LAYER_PATH.read_text(encoding="utf-8"))


def path_part(s: str) -> str:
    s = re.sub(r"[^a-z0-9._-]+", "_", (s or "").lower()).strip("_")
    return s[:96] or "entry"


def ts_compact(iso: str) -> str:
    return re.sub(r"\.\d{3}Z$", "Z", iso.replace("-", "").replace(":", ""))


def extract_records():
    records = []
    seen = set()
    for p in sorted(glob.glob("output/reports/run_log.jsonl*")):
        with open(p, encoding="utf-8") as f:
            last_payload = None
            last_session = None
            last_agent = None
            for L in f:
                if not L.strip():
                    continue
                try:
                    e = json.loads(L)
                except Exception:
                    continue
                if e.get("module_id") != "node_conv_288":
                    continue
                ev = e.get("event", "")
                if ev == "agent_result" and e.get("payload", {}).get("module_id") == "node_conv_288":
                    last_payload = e.get("payload")
                    last_session = e.get("session_id")
                    last_agent = e.get("agent")
                elif ev == "state_transition" and e.get("from") == "verifying":
                    r = e.get("pipeline_state", {}).get("results", {}).get("node_conv_288")
                    if r and r.get("status") in ("fail", "syntax_error"):
                        ts = e.get("timestamp", "")
                        key = (
                            ts,
                            r.get("timing_actual_cycles"),
                            r.get("first_mismatch_index"),
                            r.get("failure_class"),
                            r.get("status_class"),
                        )
                        if key in seen:
                            continue
                        seen.add(key)
                        records.append(
                            {
                                "ts": ts,
                                "reason": e.get("reason"),
                                "attempts": e.get("pipeline_state", {})
                                .get("attempts", {})
                                .get("node_conv_288"),
                                "agent": last_agent if last_payload else None,
                                "spec_hash": (last_payload or {}).get("spec_hash"),
                                "verilog_source": (last_payload or {}).get("verilog_source", ""),
                                "session_id": last_session,
                                "verif": r,
                            }
                        )
    records.sort(key=lambda r: r["ts"])
    return records


def verif_score(r):
    actual = r.get("timing_actual_cycles")
    expected = r.get("timing_expected_cycles")
    timing_delta = (
        actual - expected
        if isinstance(actual, (int, float))
        and isinstance(expected, (int, float))
        and actual >= 0
        and expected >= 0
        else None
    )
    outputs_received = r.get("outputs_received")
    outputs_expected = r.get("outputs_expected")
    completion = (
        outputs_received / outputs_expected
        if isinstance(outputs_received, (int, float))
        and isinstance(outputs_expected, (int, float))
        and outputs_expected > 0
        else None
    )
    return {
        "syntax_ok": r.get("status") != "syntax_error",
        "sim_completed": r.get("status_class") in ("sim_passed", "sim_completed_mismatch")
        or r.get("status") == "pass",
        "timing_delta_cycles": timing_delta,
        "timing_abs_delta_cycles": abs(timing_delta) if timing_delta is not None else None,
        "outputs_received": outputs_received,
        "outputs_expected": outputs_expected,
        "output_completion_ratio": completion,
        "first_mismatch_index": r.get("first_mismatch_index"),
        "max_error": r.get("max_error"),
        "mean_error": r.get("mean_error"),
        "exact_match_count": r.get("exact_match_count"),
        "mismatch_count": r.get("mismatch_count"),
        "signed_error_sum": r.get("signed_error_sum"),
        "positive_error_count": r.get("positive_error_count"),
        "negative_error_count": r.get("negative_error_count"),
        "axi_out_of_range_reads": r.get("axi_weight_out_of_range_reads"),
    }


def diag_summary(r):
    score = verif_score(r)
    return {
        "status": r.get("status"),
        "status_class": r.get("status_class"),
        "failure_class": r.get("failure_class"),
        "failure_category": r.get("failure_category"),
        "timing_pass": r.get("timing_pass"),
        "timing_actual_cycles": r.get("timing_actual_cycles"),
        "timing_expected_cycles": r.get("timing_expected_cycles"),
        "timing_delta_cycles": score["timing_delta_cycles"],
        "outputs_received": r.get("outputs_received"),
        "outputs_expected": r.get("outputs_expected"),
        "output_completion_ratio": score["output_completion_ratio"],
        "first_mismatch": {
            "flat_index": r.get("first_mismatch_index"),
            "vector_index": r.get("first_mismatch_vector_index"),
            "output_index": r.get("first_mismatch_output_index"),
            "channel_index": r.get("first_mismatch_channel_index"),
            "expected": r.get("first_mismatch_expected"),
            "got": r.get("first_mismatch_got"),
        },
        "error_stats": {
            "max_error": r.get("max_error"),
            "mean_error": r.get("mean_error"),
            "exact_match_count": r.get("exact_match_count"),
            "mismatch_count": r.get("mismatch_count"),
            "signed_error_sum": r.get("signed_error_sum"),
            "positive_error_count": r.get("positive_error_count"),
            "negative_error_count": r.get("negative_error_count"),
        },
        "gap": {
            "missing_index_start": r.get("missing_index_start"),
            "missing_index_end": r.get("missing_index_end"),
            "output_gap_histogram": r.get("output_gap_histogram"),
            "last_valid_out_cycle": r.get("last_valid_out_cycle"),
            "simulation_end_cycle": r.get("simulation_end_cycle"),
        },
        "axi_weight_trace": {
            "model_enabled": r.get("axi_weight_memory_model_enabled"),
            "model_status": r.get("axi_weight_memory_model_status"),
            "ar_handshakes": r.get("axi_weight_ar_handshakes"),
            "r_beats": r.get("axi_weight_r_beats"),
            "completed_bursts": r.get("axi_weight_completed_bursts"),
            "out_of_range_reads": r.get("axi_weight_out_of_range_reads"),
        },
    }


def shape_summary(layer):
    return {
        "input_shape": layer["input_shape"],
        "output_shape": layer["output_shape"],
        "weight_shape": layer["weight_shape"],
        "input_width_bits": layer["input_width_bits"],
        "output_width_bits": layer["output_width_bits"],
        "stride": layer.get("stride"),
        "padding": layer.get("padding"),
        "dilation": layer.get("dilation"),
        "groups": layer.get("groups"),
        "mac_parallelism": layer.get("mac_parallelism"),
        "io_mode": layer.get("io_mode"),
        "channel_tile": layer.get("channel_tile"),
    }


def stage_for(rec):
    return "surgeon_assayer" if rec["agent"] == "Surgeon" else "foundry_assayer"


def main():
    records = extract_records()
    print(f"records to write: {len(records)}")
    attempt_counter = 0
    new_index_lines = []
    for rec in records:
        if not rec.get("verilog_source"):
            print(f"  skip {rec['ts']}: no verilog_source captured")
            continue
        attempt_counter += 1
        stage = stage_for(rec)
        timestamp = ts_compact(rec["ts"])
        id_ = "__".join(
            [path_part("node_conv_288"), f"{attempt_counter:03d}", path_part(stage), timestamp]
        )
        dir_ = CORPUS_VIS / id_
        dir_.mkdir(parents=True, exist_ok=True)
        rtl_abs = dir_ / "node_conv_288.v"
        rtl_abs.write_text(rec["verilog_source"], encoding="utf-8")
        failure_abs = dir_ / "failure.json"
        score = verif_score(rec["verif"])
        summary = diag_summary(rec["verif"])
        shape = shape_summary(APPLIED_LAYER)
        spec_hash = rec.get("spec_hash") or APPLIED_LAYER.get("contract_id", "") + "?"
        rtl_rel = str(rtl_abs).replace("\\", "/")
        failure_rel = str(failure_abs).replace("\\", "/")
        entry = {
            "id": id_,
            "created_at": rec["ts"],
            "module_id": "node_conv_288",
            "stage": stage,
            "attempt_index": attempt_counter,
            "op_type": APPLIED_LAYER["op_type"],
            "contract_id": APPLIED_LAYER["contract_id"],
            "spec_hash": spec_hash,
            "generated_by": rec["agent"],
            "module_attempt": rec.get("attempts"),
            "rtl_path": rtl_rel,
            "failure_path": failure_rel,
            "score": score,
            "summary": summary,
            "shape": shape,
        }
        failure_payload = {
            "entry": entry,
            "layer_ir": APPLIED_LAYER,
            "module": {
                "module_id": "node_conv_288",
                "spec_hash": spec_hash,
                "generated_by": rec["agent"],
                "attempt": rec.get("attempts"),
                "rtl_path": rtl_rel,
            },
            "verif_result": rec["verif"],
            "logs": {
                "session_id": rec.get("session_id"),
                "reason": rec.get("reason"),
                "iverilog_stderr": rec["verif"].get("iverilog_stderr"),
                "verilator_stderr": rec["verif"].get("verilator_stderr"),
                "fix_hint": rec["verif"].get("fix_hint"),
                "classifier_reason": rec["verif"].get("classifier_reason"),
            },
        }
        with open(failure_abs, "w", encoding="utf-8") as f:
            json.dump(failure_payload, f, indent=2)
        new_index_lines.append(json.dumps(entry, ensure_ascii=False))
        print(f"  wrote {id_}")

    INDEX.parent.mkdir(parents=True, exist_ok=True)
    with open(INDEX, "a", encoding="utf-8") as f:
        for line in new_index_lines:
            f.write(line + "\n")
    print(f"\nappended {len(new_index_lines)} entries to {INDEX}")


if __name__ == "__main__":
    main()
