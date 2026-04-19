// Remove the Claude Code per-response output token cap so Surgeon can emit
// full Verilog without being killed mid-generation.
process.env["CLAUDE_CODE_MAX_OUTPUT_TOKENS"] = "200000";

import { handlePipelineError, runCli } from "./orchestrate.js";

runCli().catch(async (error: unknown) => {
  await handlePipelineError(error);
});
