#!/usr/bin/env python3
"""Hoist module-level pure wire/reg declarations in nn2rtl_top.v to the top of the
module body, so iverilog (strict, no use-before-declaration) can elaborate it.
Verilator tolerates use-before-decl; iverilog does not. Operates on a COPY.
Only hoists DEPTH-0 (module-level) single-name pure declarations with NO '=' (assigns
stay in place; their RHS refs are satisfied by the hoisted decls). Output: copy with
all such decls moved to right after the module header."""
import re, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
src = (ROOT/"output/rtl/nn2rtl_top.v").read_text().split("\n")
out_path = ROOT/"output/rtl/nn2rtl_top_iv.v"

# find module header end: the line with ');' closing the port list of `module nn2rtl_top`
mod_start = next(i for i,l in enumerate(src) if re.match(r"\s*module\s+nn2rtl_top\b", l))
hdr_end = next(i for i in range(mod_start, len(src)) if re.match(r"\s*\);", src[i]))

# Module-level declarations are exactly 4-space indented (generate/block-internal are deeper).
# Hoist all single-line 4-space wire/reg declarations (pure AND `= expr` combinational),
# preserving their relative order, so iverilog sees every signal declared before use.
decl_re = re.compile(r"^    (wire|reg|integer)\b.*;\s*$")
pure, withassign, body = [], [], []
endmod = next(i for i in range(hdr_end+1, len(src)) if re.match(r"\s*endmodule\b", src[i]))
for i in range(hdr_end+1, endmod):
    l = src[i]
    if decl_re.match(l) and "assign" not in l:
        (withassign if "=" in l else pure).append(l.strip())  # pure decls vs `= expr` combinational
    else:
        body.append(l)
body.extend(src[endmod:])
# pure declarations FIRST (all signals declared), THEN combinational `= expr` (refs pure wires above)
hoisted = pure + withassign

result = src[:hdr_end+1] + ["    // [IV-HOIST] module-level declarations hoisted for iverilog"] \
       + ["    "+d for d in hoisted] + [""] + body
out_path.write_text("\n".join(result))
print(f"hoisted {len(hoisted)} decls -> {out_path}")
