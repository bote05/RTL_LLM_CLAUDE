#!/usr/bin/env python3
"""Compare observed engine output hex against the layer's .goldout.

The observed file is produced by tb/engine_one_layer_tb.v: one 2048-bit
BRAM word per line, big-endian hex (512 hex chars per line). Each word
carries up to 256 output channels of one output pixel (channel 0 = LSByte
= the rightmost two hex chars). Only the first `bytes_per_sample` bytes
of each word are valid (= OC; the upper 256-OC bytes are don't-care
padding the engine emits as zero).

The .goldout file is in the binary NN2V vector format described in
scripts/golden_impl.py. We compare against vector index 0.

Two usage modes:
  1. Direct (legacy): supply --observed, --goldout, --n-out-words,
     --word-bytes (= valid bytes-per-word to compare).
  2. Schedule-driven: supply --dispatch-idx and we look up the
     module_id and per-layer counts from
     output/rtl/nn2rtl_scheduler_schedule.json + output/layer_ir.json.

Outputs (stdout, machine-parseable last line):
  "RESULT_JSON: {json...}"  -- always emitted with pass/fail + stats.
Exit code 0 on PASS, 1 on FAIL.
"""

import argparse
import json
import re
import struct
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent


def read_observed_hex(path: Path, n_pixels: int, oc_passes: int,
                      oc_bytes: int, word_bytes_total: int = 256) -> bytes:
    """Read the TB-dumped hex file and return concatenated valid-byte stream.

    The TB dumps `n_pixels * oc_passes` lines. Each line is a 2048-bit
    BRAM word (512 hex chars, big-endian). Channel 0 is the LSByte.
    For pixel p: bytes [p*oc_passes .. p*oc_passes + oc_passes - 1]
    are concatenated (chunk 0 = channels 0..255, chunk 1 = channels
    256..511, ...) and the first `oc_bytes` are kept.
    """
    n_words = n_pixels * oc_passes
    out = bytearray()
    with path.open("r") as fh:
        lines = [ln.strip() for ln in fh if ln.strip()]
    if len(lines) != n_words:
        raise SystemExit(
            f"observed hex {path} has {len(lines)} lines, expected {n_words}"
        )
    for px in range(n_pixels):
        per_pixel = bytearray()
        for chunk in range(oc_passes):
            ln = lines[px * oc_passes + chunk]
            expected_hex_chars = word_bytes_total * 2
            if len(ln) != expected_hex_chars:
                raise SystemExit(
                    f"observed hex line wrong width: got {len(ln)} chars, "
                    f"expected {expected_hex_chars}"
                )
            word_int = int(ln, 16)
            word_bytes_arr = word_int.to_bytes(word_bytes_total, byteorder="big")
            # Reverse to little-endian byte order (channel 0 first).
            per_pixel.extend(word_bytes_arr[::-1])
        out.extend(per_pixel[:oc_bytes])
    return bytes(out)


def read_goldout_vector(path: Path, vector_index: int) -> tuple[bytes, int, int]:
    with path.open("rb") as fh:
        header = fh.read(20)
        if len(header) != 20:
            raise SystemExit(f"goldout {path} truncated header")
        magic, version, num_vectors, samples_per_vector, bytes_per_sample = (
            struct.unpack("<4sIIII", header)
        )
        if magic != b"NN2V":
            raise SystemExit(f"goldout {path} bad magic: {magic!r}")
        if version != 2:
            raise SystemExit(f"goldout {path} unsupported version {version}")
        if vector_index < 0 or vector_index >= num_vectors:
            raise SystemExit(
                f"goldout {path} vector_index {vector_index} out of range "
                f"[0,{num_vectors})"
            )
        words_per_sample = (bytes_per_sample + 3) // 4
        bytes_per_vector_aligned = samples_per_vector * words_per_sample * 4
        fh.seek(20 + vector_index * bytes_per_vector_aligned)
        raw = fh.read(bytes_per_vector_aligned)
        if len(raw) != bytes_per_vector_aligned:
            raise SystemExit(
                f"goldout {path} could not read vector_index {vector_index}"
            )
    if bytes_per_sample % 4 != 0:
        stripped = bytearray()
        for s in range(samples_per_vector):
            start = s * words_per_sample * 4
            stripped.extend(raw[start : start + bytes_per_sample])
        return bytes(stripped), samples_per_vector, bytes_per_sample
    return raw, samples_per_vector, bytes_per_sample


def _to_repo_path(p: str) -> Path:
    """Translate WSL-style absolute paths in layer_ir.json into
    repo-relative paths under the local Windows checkout."""
    if p.startswith("/mnt/c/Users/User/Desktop/RTL_LLM_CLAUDE/nn2rtl-repo/"):
        rel = p[len("/mnt/c/Users/User/Desktop/RTL_LLM_CLAUDE/nn2rtl-repo/"):]
        return REPO_ROOT / rel
    return Path(p)


def lookup_dispatch(dispatch_idx: int) -> dict:
    schedule_path = REPO_ROOT / "output" / "rtl" / "nn2rtl_scheduler_schedule.json"
    layer_ir_path = REPO_ROOT / "output" / "layer_ir.json"
    schedule = json.loads(schedule_path.read_text())
    layer_ir = json.loads(layer_ir_path.read_text())
    dispatches = schedule["dispatches"]
    if dispatch_idx < 0 or dispatch_idx >= len(dispatches):
        raise SystemExit(
            f"dispatch_idx {dispatch_idx} out of range [0,{len(dispatches)})"
        )
    d = dispatches[dispatch_idx]
    module_id = d["module_id"]
    layer = next((l for l in layer_ir["layers"] if l.get("module_id") == module_id), None)
    if layer is None:
        raise SystemExit(f"layer_ir.json has no entry for {module_id}")
    goldin_path = _to_repo_path(layer["golden_inputs_path"])
    goldout_path = _to_repo_path(layer["golden_outputs_path"])
    return {
        "dispatch": d,
        "layer": layer,
        "module_id": module_id,
        "goldin_path": goldin_path,
        "goldout_path": goldout_path,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dispatch-idx", type=int, default=None,
                        help="If set, look up paths/sizes from schedule + "
                             "layer_ir for this dispatch index.")
    parser.add_argument("--observed", default=None,
                        help="Observed wide-BRAM hex dump (default: "
                             "output/engine_tb_observed.hex)")
    parser.add_argument("--goldout", default=None,
                        help="Override .goldout path")
    parser.add_argument("--vector-index", type=int, default=0,
                        help="Vector index inside .goldout")
    parser.add_argument("--n-out-words", type=int, default=None,
                        help="Number of 2048-bit output words "
                             "(default from layer OH*OW)")
    parser.add_argument("--word-bytes", type=int, default=None,
                        help="Bytes per BRAM word to compare (= OC channels). "
                             "Defaults to layer channel_out, else 256.")
    parser.add_argument("--word-bytes-total", type=int, default=256,
                        help="Total width of each BRAM word in bytes (256).")
    parser.add_argument("--json-out", default=None,
                        help="Optional path to write a single-record JSON "
                             "result.")
    args = parser.parse_args()

    module_id = None
    ow_dim = None
    if args.dispatch_idx is not None:
        info = lookup_dispatch(args.dispatch_idx)
        d = info["dispatch"]
        module_id = info["module_id"]
        oh, ow = d["output_hw"]
        ow_dim = ow
        oc = d["channel_out"]
        n_out_pixels = oh * ow
        oc_passes = (oc + 255) // 256
        oc_bytes = oc
        goldout_path = Path(args.goldout) if args.goldout else info["goldout_path"]
    else:
        # Legacy positional mode.
        if args.n_out_words is None:
            args.n_out_words = 196
        if args.word_bytes is None:
            args.word_bytes = 256
        n_out_pixels = args.n_out_words
        oc_bytes = args.word_bytes
        oc_passes = max(1, (oc_bytes + 255) // 256)
        if args.goldout is None:
            raise SystemExit("--goldout required when --dispatch-idx not given")
        goldout_path = Path(args.goldout)

    observed_path = Path(args.observed) if args.observed else (
        REPO_ROOT / "output" / "engine_tb_observed.hex"
    )

    result = {
        "module_id": module_id,
        "dispatch_idx": args.dispatch_idx,
        "observed": str(observed_path),
        "goldout": str(goldout_path),
        "n_out_pixels": n_out_pixels,
        "oc_passes": oc_passes,
        "oc_bytes": oc_bytes,
        "word_bytes_total": args.word_bytes_total,
        "status": "FAIL",
        "n_total_bytes": 0,
        "n_mismatches": 0,
        "max_error": 0,
        "first_mismatches": [],
        "error": None,
    }

    try:
        observed = read_observed_hex(observed_path, n_out_pixels,
                                     oc_passes, oc_bytes,
                                     args.word_bytes_total)
        expected, n_samples, bps = read_goldout_vector(goldout_path,
                                                       args.vector_index)
    except SystemExit as e:
        result["error"] = str(e)
        _emit(result, args.json_out)
        return 1

    if n_samples != n_out_pixels:
        result["error"] = (f".goldout has {n_samples} samples; "
                           f"expected {n_out_pixels}")
        _emit(result, args.json_out)
        return 1
    if bps != oc_bytes:
        result["error"] = (f".goldout bytes_per_sample={bps}; "
                           f"expected (= OC bytes) {oc_bytes}")
        _emit(result, args.json_out)
        return 1
    if len(observed) != len(expected):
        result["error"] = (f"observed length {len(observed)} != "
                           f"expected length {len(expected)}")
        _emit(result, args.json_out)
        return 1

    mismatches = []
    max_error = 0
    for i, (o, e) in enumerate(zip(observed, expected)):
        o_s = o - 256 if o >= 128 else o
        e_s = e - 256 if e >= 128 else e
        err = abs(o_s - e_s)
        if err > max_error:
            max_error = err
        if o != e:
            mismatches.append((i, e, o, e_s, o_s))

    result["n_total_bytes"] = len(observed)
    result["n_mismatches"] = len(mismatches)
    result["max_error"] = max_error
    # Capture up to 10 first mismatches with pixel/channel coordinates.
    if mismatches:
        for idx, e, o, e_s, o_s in mismatches[:10]:
            sample = idx // oc_bytes
            channel = idx % oc_bytes
            entry = {
                "byte": idx,
                "sample": sample,
                "channel": channel,
                "expected": int(e),
                "got": int(o),
                "expected_s": int(e_s),
                "got_s": int(o_s),
            }
            if ow_dim:
                entry["pixel_row"] = sample // ow_dim
                entry["pixel_col"] = sample % ow_dim
            result["first_mismatches"].append(entry)

    if not mismatches:
        result["status"] = "PASS"
        print(
            f"PASS: {len(observed)} bytes match "
            f"(n_samples={n_samples}, bytes_per_sample={bps}, "
            f"max_error=0, mismatch_count=0)"
        )
        _emit(result, args.json_out)
        return 0

    result["status"] = "FAIL"
    print(f"FAIL: {len(mismatches)} mismatches, max_error={max_error}")
    for m in result["first_mismatches"]:
        loc = f"pixel[{m.get('pixel_row','?')},{m.get('pixel_col','?')}]"
        print(
            f"  byte[{m['byte']}] ({loc} ch{m['channel']}): "
            f"expected 0x{m['expected']:02x} ({m['expected_s']}), "
            f"got 0x{m['got']:02x} ({m['got_s']})"
        )
    if len(mismatches) > 10:
        print(f"  ... ({len(mismatches) - 10} more)")
    _emit(result, args.json_out)
    return 1


def _emit(result: dict, json_out_path: str | None) -> None:
    print("RESULT_JSON: " + json.dumps(result))
    if json_out_path:
        Path(json_out_path).write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    sys.exit(main())
