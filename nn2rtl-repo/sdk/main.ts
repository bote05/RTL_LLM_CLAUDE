// Remove the Claude Code per-response output token cap so Surgeon can emit
// full Verilog without being killed mid-generation.
process.env["CLAUDE_CODE_MAX_OUTPUT_TOKENS"] = "200000";

import { runImproveCli } from "./improve.js";
import { handlePipelineError, runCli } from "./orchestrate.js";

const argv = process.argv.slice(2);
const command = argv[0];
const runner = command === "improve" ? runImproveCli(argv.slice(1)) : runCli(argv);

runner.catch(async (error: unknown) => {
  await handlePipelineError(error);
});
