import { handlePipelineError, runCli } from "./orchestrate.js";

runCli().catch(async (error: unknown) => {
  await handlePipelineError(error);
});
