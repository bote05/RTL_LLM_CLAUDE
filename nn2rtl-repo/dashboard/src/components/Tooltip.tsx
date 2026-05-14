import { HelpCircle } from "lucide-react";
import { useState } from "react";

/** A tiny help-icon tooltip primitive built with no extra dependencies.
 *
 * The dashboard is being designed for users without an electrical-engineering
 * background. Anywhere we display an EE acronym (LUT, FF, DSP, BRAM, Fmax,
 * WNS, MAC, latency_cycles, MP, …) we wrap the label in this `<HelpTooltip>`
 * so a hover or tap reveals a plain-language one-liner.
 *
 * Visible on focus + hover, so keyboard users get it too. */
export function HelpTooltip({ term, hint, size = 13 }: {
  term: string;
  hint: string;
  size?: number;
}): JSX.Element {
  const [open, setOpen] = useState(false);
  return (
    <span
      className="help-anchor"
      onMouseEnter={() => setOpen(true)}
      onMouseLeave={() => setOpen(false)}
      onFocus={() => setOpen(true)}
      onBlur={() => setOpen(false)}
      tabIndex={0}
      aria-describedby={open ? `tt-${term}` : undefined}
    >
      <HelpCircle size={size} className="help-icon" aria-label={`What is ${term}?`} />
      {open && (
        <span className="help-bubble" role="tooltip" id={`tt-${term}`}>
          <strong>{term}</strong>
          <span>{hint}</span>
        </span>
      )}
    </span>
  );
}

/** Curated glossary of EE/FPGA jargon used throughout the dashboard.
 *  These are the one-liners non-EE folks see when they hover the ? icon. */
export const GLOSSARY: Record<string, string> = {
  LUT: "Look-Up Table — the smallest programmable logic element in an FPGA. Fewer LUTs = a smaller, cheaper design.",
  FF: "Flip-Flop — a 1-bit memory cell that holds state between clock cycles. Pipelines and registers consume FFs.",
  DSP: "Digital Signal Processing block — a hard multiplier+adder unit. Using DSPs (instead of LUTs) for multiplies is typically much faster and uses less area.",
  BRAM: "Block RAM — on-chip memory blocks. Weights and intermediate activations are stored here.",
  Fmax: "Maximum clock frequency the design can run at without violating timing. Higher Fmax = faster designs.",
  WNS: "Worst Negative Slack — how many nanoseconds of timing margin you have at the worst path. Positive WNS means timing is met; negative means it failed.",
  MAC: "Multiply-Accumulate — the core conv/dense op: y += w * x. CNNs are dominated by MACs.",
  MP: "MaxPool — downsamples an activation map by taking the maximum over a window.",
  latency_cycles: "How many clock cycles a module needs from first input to last output. Lower = lower end-to-end latency.",
  II: "Initiation Interval — clock cycles between accepting successive inputs in a pipelined design. Lower = higher throughput.",
};

/** A labeled term with an inline help icon — used in tables and metric grids
 *  so EE acronyms always carry their glossary hint with them. */
export function LabeledTerm({ term, glossaryKey }: {
  term: string;
  glossaryKey?: keyof typeof GLOSSARY;
}): JSX.Element {
  const key = (glossaryKey ?? term) as keyof typeof GLOSSARY;
  const hint = GLOSSARY[key];
  if (!hint) return <>{term}</> as JSX.Element;
  return (
    <span className="labeled-term">
      {term}
      <HelpTooltip term={term} hint={hint} />
    </span>
  );
}
