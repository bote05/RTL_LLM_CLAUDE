#!/usr/bin/env python3
"""Apply the adversarially-verified MP=16 byte-exact edits (workflow wp8eb4eew) to the
17 MobileNetV2 depthwise modules. ATOMIC per module: a module file is written ONLY if
every one of its edits matches EXACTLY ONCE (applied in listed order, since some edits
chain off an earlier one). Preserves the file's original newline style.

Usage: python scripts/apply_mp16_batch.py <path-to-workflow-output-json>
"""
import json
import os
import sys

RTL = r"C:\Users\User\Desktop\RTL_LLM_CLAUDE\nn2rtl-repo\output\mobilenet-v2\rtl"


def main():
    if len(sys.argv) < 2:
        print("usage: apply_mp16_batch.py <workflow-output.json> [--skip mod1,mod2,...]")
        sys.exit(2)
    skip = set()
    if "--skip" in sys.argv:
        skip = set(sys.argv[sys.argv.index("--skip") + 1].split(","))
    with open(sys.argv[1], "r", encoding="utf-8") as f:
        data = json.load(f)
    changes = [c for c in data["result"]["safe_changes"] if c["module"] not in skip]
    if skip:
        print(f"(skipping: {sorted(skip)})")

    n_ok = 0
    n_fail = 0
    total_edits = 0
    for ch in changes:
        mod = ch["module"]
        path = os.path.join(RTL, mod + ".v")
        with open(path, "rb") as f:
            raw = f.read()
        crlf = b"\r\n" in raw
        text = raw.decode("utf-8")
        if crlf:
            text = text.replace("\r\n", "\n")

        work = text
        ok = True
        applied = 0
        fail_msgs = []
        for i, e in enumerate(ch["edits"]):
            old = e["old"]
            new = e["new"]
            cnt = work.count(old)
            if cnt != 1:
                ok = False
                snippet = old.replace("\n", "\\n")[:70]
                fail_msgs.append(f"    edit#{i} count={cnt}: {snippet!r}")
                break
            work = work.replace(old, new, 1)
            applied += 1

        if ok:
            out = work.replace("\n", "\r\n") if crlf else work
            with open(path, "wb") as f:
                f.write(out.encode("utf-8"))
            print(f"OK   {mod}: {applied}/{len(ch['edits'])} edits  MP {ch['MP_old']}->{ch['MP_new']}  OC_PASSES_new={ch['OC_PASSES_new']}")
            n_ok += 1
            total_edits += applied
        else:
            print(f"FAIL {mod}: ATOMIC SKIP (file untouched), {applied} matched before failure:")
            print("\n".join(fail_msgs))
            n_fail += 1

    print("-" * 60)
    print(f"SUMMARY: {n_ok} modules applied ({total_edits} edits), {n_fail} failed")
    if n_fail:
        sys.exit(1)


if __name__ == "__main__":
    main()
