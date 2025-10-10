export type RunState = 'idle' | 'running' | 'paused' | 'completed' | 'cancelled' | 'cancelling' | 'error';

export interface StatusPayload {
  state: RunState;
  error?: string | null;
  config?: RunConfig | null;
  started_at?: number | null;
  finished_at?: number | null;
  summary?: Record<string, ModelSummary>;
  artifacts?: ArtifactsInfo | null;
  ts?: number;
}

export interface RunConfig {
  models: string[];
  judge_model: string;
  limit: number | null;
  max_tokens: number;
  judge_max_tokens: number;
  temperature: number;
  judge_temperature: number;
  sleep_s: number;
  outdir: string;
  verbose?: boolean;
}

export interface ModelSummary {
  total: number;
  ok: number;
  issues: number;
  avg_initial_completeness: number;
  avg_followup_completeness: number;
  initial_refusal_rate: number;
  followup_refusal_rate: number;
  initial_sourcing_counts: Record<string, number>;
  followup_sourcing_counts: Record<string, number>;
  asymmetry_counts: Record<string, number>;
  error_counts: Record<string, number>;
}

export interface MessageEntry {
  id: string;
  model?: string;
  promptIndex?: number;
  role: 'user' | 'assistant' | 'judge' | 'system';
  content: string;
  step?: string;
  timestamp: number;
}

export interface ArtifactsInfo {
  csv_path?: string | null;
  runs_dir?: string | null;
  summary?: Record<string, ModelSummary> | null;
}

export interface EventPayload<T = Record<string, unknown>> {
  type: string;
  payload: T;
}

export interface DefaultsResponse extends RunConfig {
  models: string[];
}
