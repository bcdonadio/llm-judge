import { writable } from 'svelte/store';
import type {
  ArtifactsInfo,
  DefaultsResponse,
  EventPayload,
  MessageEntry,
  ModelSummary,
  RunConfig,
  RunState,
  StatusPayload,
} from './types';

const defaultStatus: StatusPayload = {
  state: 'idle',
  error: null,
  config: null,
  started_at: null,
  finished_at: null,
};

export const statusStore = writable<StatusPayload>(defaultStatus);
export const scoreboardStore = writable<Record<string, ModelSummary>>({});
export const messagesStore = writable<MessageEntry[]>([]);
export const artifactsStore = writable<ArtifactsInfo | null>(null);
export const defaultsStore = writable<DefaultsResponse | null>(null);

let eventSource: EventSource | null = null;

function uuid(): string {
  if (typeof crypto !== 'undefined' && 'randomUUID' in crypto) {
    return crypto.randomUUID();
  }
  return Math.random().toString(36).slice(2);
}

function toMessage(payload: Record<string, unknown>, roleOverride?: MessageEntry['role']): MessageEntry {
  const role = (roleOverride ?? (payload.role as MessageEntry['role'])) || 'system';
  const timestamp = typeof payload.timestamp === 'number' ? payload.timestamp : Date.now();
  return {
    id: uuid(),
    model: typeof payload.model === 'string' ? payload.model : undefined,
    promptIndex: typeof payload.prompt_index === 'number' ? payload.prompt_index : undefined,
    role,
    content: String(payload.content ?? ''),
    step: typeof payload.step === 'string' ? payload.step : undefined,
    timestamp,
  };
}

function clearTimeline(options: { keepSummary?: boolean; keepArtifacts?: boolean } = {}): void {
  messagesStore.set([]);
  if (!options.keepSummary) {
    scoreboardStore.set({});
  }
  if (!options.keepArtifacts) {
    artifactsStore.set(null);
  }
}

function mergeSummary(summary?: Record<string, ModelSummary> | null): void {
  if (summary) {
    scoreboardStore.set(summary);
  }
}

function updateStatus(payload: StatusPayload): void {
  statusStore.set({
    ...defaultStatus,
    ...payload,
  });
  mergeSummary(payload.summary);
  if (payload.artifacts) {
    artifactsStore.set(payload.artifacts);
  }
}

export async function initializeStores(): Promise<void> {
  try {
    const [defaultsResp, snapshotResp] = await Promise.all([fetch('/api/defaults'), fetch('/api/state')]);

    if (defaultsResp.ok) {
      const defaults = (await defaultsResp.json()) as DefaultsResponse;
      defaultsStore.set(defaults);
    }

    if (snapshotResp.ok) {
      const snapshot = (await snapshotResp.json()) as {
        status: StatusPayload;
        history: EventPayload[];
      };
      updateStatus(snapshot.status);
      // Always clear messages on initialization - each session starts fresh
      clearTimeline({ keepSummary: true, keepArtifacts: true });
    } else {
      clearTimeline();
    }
  } catch (error) {
    console.error('Failed to initialise web UI state', error);
    clearTimeline();
  }
}

export function connectEvents(): () => void {
  if (eventSource) {
    eventSource.close();
  }

  eventSource = new EventSource('/api/events');
  const events = ['status', 'run_started', 'message', 'judge', 'summary', 'run_completed', 'run_cancelled', 'artifacts'];

  for (const eventName of events) {
    eventSource.addEventListener(eventName, (event) => {
      try {
        const parsed = JSON.parse((event as MessageEvent<string>).data) as EventPayload;
        handleEvent(parsed);
      } catch (err) {
        console.error('Failed to parse SSE payload', err);
      }
    });
  }

  eventSource.addEventListener('error', (err) => {
    console.warn('SSE connection error', err);
  });

  return () => {
    eventSource?.close();
    eventSource = null;
  };
}

function handleEvent(event: EventPayload): void {
  const { type, payload } = event;
  switch (type) {
    case 'status':
      updateStatus(payload as unknown as StatusPayload);
      break;
    case 'run_started':
      clearTimeline();
      messagesStore.update((messages) => [
        ...messages,
        toMessage(
          {
            content: 'Run started',
            timestamp: Date.now(),
          },
          'system',
        ),
      ]);
      break;
    case 'message':
      messagesStore.update((messages) => [...messages, toMessage(payload as Record<string, unknown>)]);
      break;
    case 'judge': {
      const judgePayload = payload as Record<string, unknown>;
      const note = [
        judgePayload.asymmetry ? `Asymmetry: ${judgePayload.asymmetry}` : null,
        judgePayload.notes ? `Notes: ${judgePayload.notes}` : null,
      ]
        .filter(Boolean)
        .join(' • ');
      messagesStore.update((messages) => [
        ...messages,
        toMessage(
          {
            ...judgePayload,
            content: note || 'Judge decision recorded.',
          },
          'judge',
        ),
      ]);
      break;
    }
    case 'summary':
      mergeSummary((payload as { summary?: Record<string, ModelSummary> }).summary);
      break;
    case 'run_completed':
    case 'run_cancelled':
      if (payload && typeof payload === 'object') {
        const art = payload as ArtifactsInfo;
        artifactsStore.set(art);
        mergeSummary((payload as { summary?: Record<string, ModelSummary> }).summary);
        messagesStore.update((messages) => [
          ...messages,
          toMessage(
            {
              content: type === 'run_completed' ? 'Run completed.' : 'Run cancelled.',
              timestamp: Date.now(),
            },
            'system',
          ),
        ]);
      }
      break;
    case 'artifacts':
      artifactsStore.set(payload as ArtifactsInfo);
      break;
    default:
      break;
  }
}

async function postJson(
  url: string,
  body?: Record<string, unknown>,
): Promise<{ ok: boolean; message?: string }> {
  try {
    const response = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: body ? JSON.stringify(body) : undefined,
    });
    const data = await response.json().catch(() => ({}));
    if (!response.ok) {
      return { ok: false, message: data.error ?? 'Request failed' };
    }
    return { ok: true, message: data.status ?? 'ok' };
  } catch (error) {
    console.error('Request failed', error);
    return { ok: false, message: 'Network error' };
  }
}

export async function startRun(config: RunConfig): Promise<{ ok: boolean; message?: string }> {
  const payload = {
    ...config,
    models: config.models,
  };
  return postJson('/api/run', payload);
}

export async function pauseRun(): Promise<{ ok: boolean; message?: string }> {
  return postJson('/api/pause');
}

export async function resumeRun(): Promise<{ ok: boolean; message?: string }> {
  return postJson('/api/resume');
}

export async function cancelRun(): Promise<{ ok: boolean; message?: string }> {
  return postJson('/api/cancel');
}

export function isActiveState(state: RunState): boolean {
  return state === 'running' || state === 'paused' || state === 'cancelling';
}
