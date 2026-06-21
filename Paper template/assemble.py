#!/usr/bin/env python3
# Assemble tscit2026_paper.tex from the drafted parts. Run: python3 assemble.py
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

def rd(p):
    with open(p, encoding="utf-8") as f:
        return f.read().strip()

methodology = rd("parts/methodology.tex")
rq1 = rd("parts/rq1.tex"); rq2 = rd("parts/rq2.tex"); rq3 = rd("parts/rq3.tex")
intro = rd("parts/intro.tex"); related = rd("parts/related.tex")
setup = rd("parts/setup.tex"); conclusion = rd("parts/conclusion.tex")
abstract_raw = rd("parts/abstract.tex"); appendix = rd("nn2rtl_appendix_bug_catalogue.tex")

# FIX 1: drop the duplicate fig:arch float in methodology (intro keeps it).
# Now a no-op if methodology already has no figure float (it was removed upstream).
BF = r"\begin{figure}"; EF = r"\end{figure}"
b = methodology.find(BF); e = methodology.find(EF)
if b != -1 and e != -1:
    methodology = (methodology[:b] + methodology[e + len(EF):])
    methodology = methodology.replace("\n\n\n\n", "\n\n").replace("\n\n\n", "\n\n")

# FIX 2: remove the duplicated open-question sentence in RQ1 paragraph 2
dup = " The marginal value of using an LLM for architecture mapping was not isolated and remains an open question."
if dup in rq1:
    rq1 = rq1.replace(dup, "", 1)  # drop the line-5 internal redundancy (the point is made in the same paragraph)

# abstract: split body from the \keywords line
al = abstract_raw.split("\n")
ki = next(i for i, l in enumerate(al) if l.strip().startswith(r"\keywords"))
abstract_body = "\n".join(al[:ki]).strip()
keywords_line = "\n".join(al[ki:]).strip()

pre = open("parts/_preamble.tex", encoding="utf-8").read()
pre = pre.replace("__ABSTRACT__", abstract_body).replace("__KEYWORDS__", keywords_line)

ai_statement = (
    "\\section*{AI Statement}\n"
    "During the preparation of this work the author used Claude (Anthropic) to assist in enhancing the writing of this paper. "
    "After using this tool, the author reviewed and edited the content as needed and takes full responsibility for the content of the work. "
    "This is distinct from the two AI systems studied here, whose roles are described in the body."
)

doc = (
    pre + "\n"
    + intro + "\n\n"
    + related + "\n\n"
    + methodology + "\n\n"
    + setup + "\n\n"
    + "\\section{Results}\n\\label{sec:results}\n\n"
    + rq1 + "\n\n"
    + rq2 + "\n\n"
    + rq3 + "\n\n"
    + conclusion + "\n\n"
    + ai_statement + "\n\n"
    + "\\bibliographystyle{ACM-Reference-Format}\n\\bibliography{my-paper}\n\n"
    + "\\appendix\n\\counterwithin{figure}{section}\n\\counterwithin{table}{section}\n"
    + appendix + "\n\n"
    + "\\end{document}\n"
)

with open("tscit2026_paper.tex", "w", encoding="utf-8") as f:
    f.write(doc)

print("WROTE tscit2026_paper.tex:", len(doc), "chars")
print("em-dash U+2014:", doc.count("—"), "| triple-dash:", doc.count("---"))
print("fig:arch defs:", doc.count(r"\label{fig:arch}"), "(want 1)")
print("top-level sections:", doc.count("\n\\section{"))
print("longtable:", doc.count(r"\begin{longtable}"))
