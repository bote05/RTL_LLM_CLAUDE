// Thin compatibility layer over @anthropic-ai/claude-agent-sdk.
// We keep our own narrowed types here (rather than re-exporting the SDK's)
// so we only depend on the fields the orchestrator actually uses. This
// insulates the rest of the codebase from shape changes in the SDK.

export type AgentDefinition = {
  description: string;
  prompt: string;
  tools?: string[];
  disallowedTools?: string[];
  // Accepts tier aliases ("sonnet" / "opus" / "haiku" / "inherit") OR full
  // model IDs ("claude-opus-4-7", "claude-sonnet-4-6"). Prefer full IDs —
  // tier aliases are affected by the user's global ~/.claude/settings default
  // model and produce non-deterministic routing.
  model?: string;
  skills?: string[];
  maxTurns?: number;
};

export type OutputFormat = {
  type: "json_schema";
  schema: Record<string, unknown>;
};

export type SDKResultSuccess = {
  type: "result";
  subtype: "success";
  result: string;
  total_cost_usd: number;
  modelUsage: Record<string, unknown>;
  structured_output?: unknown;
};

export type SDKResultError = {
  type: "result";
  subtype:
    | "error_during_execution"
    | "error_max_turns"
    | "error_max_budget_usd"
    | "error_max_structured_output_retries";
  is_error: boolean;
  total_cost_usd: number;
  modelUsage: Record<string, unknown>;
  structured_output?: unknown;
};

export type SDKResultMessage = SDKResultSuccess | SDKResultError;

export type SDKMessage =
  | SDKResultMessage
  | {
      type: string;
      subtype?: string;
      [key: string]: unknown;
    };

type QueryOptions = {
  cwd?: string;
  tools?: string[];
  allowedTools?: string[];
  plugins?: Array<{ type: "local"; path: string }>;
  agents?: Record<string, AgentDefinition>;
  outputFormat?: OutputFormat;
  maxTurns?: number;
};

type QueryParams = {
  prompt: string;
  options?: QueryOptions;
};

type Query = AsyncGenerator<SDKMessage, void>;

import { query as rawQuery } from "@anthropic-ai/claude-agent-sdk";

export const query = rawQuery as unknown as (_params: QueryParams) => Query;
