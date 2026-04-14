// TODO: Remove this compatibility shim once the published @anthropic-ai/claude-agent-sdk package ships a valid root declaration file for sdk.mjs.

export type AgentDefinition = {
  description: string;
  prompt: string;
  tools?: string[];
  disallowedTools?: string[];
  model?: "sonnet" | "opus" | "haiku" | "inherit";
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
