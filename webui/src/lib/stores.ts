import { writable } from "svelte/store";
import type {
  ArtifactsInfo,
  DefaultsResponse,
  EventPayload,
  ModelInfo,
  ModelsResponse,
  MessageEntry,
  ModelSummary,
  RunConfig,
  RunState,
  StatusPayload,
} from "./types";

const defaultStatus: StatusPayload = {
  state: "idle",
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
export const modelCatalogStore = writable<ModelInfo[]>([]);

let ws: WebSocket | null = null;
let reconnectTimer: ReturnType<typeof setTimeout> | null = null;
let reconnectDelay = 1000; // Start with 1 second
let pingTimer: ReturnType<typeof setInterval> | null = null;
let isManualDisconnect = false;

export type ConnectionState =
  | "connecting"
  | "connected"
  | "disconnected"
  | "error";

export const connectionStore = writable<ConnectionState>("disconnected");

function uuid(): string {
  if (typeof crypto !== "undefined" && "randomUUID" in crypto) {
    return crypto.randomUUID();
  }
  return Math.random().toString(36).slice(2);
}

function toMessage(
  payload: Record<string, unknown>,
  roleOverride?: MessageEntry["role"],
): MessageEntry {
  const role =
    (roleOverride ?? (payload.role as MessageEntry["role"])) || "system";
  const timestamp =
    typeof payload.timestamp === "number" ? payload.timestamp : Date.now();
  return {
    id: uuid(),
    model: typeof payload.model === "string" ? payload.model : undefined,
    promptIndex:
      typeof payload.prompt_index === "number"
        ? payload.prompt_index
        : undefined,
    role,
    content: String(payload.content ?? ""),
    step: typeof payload.step === "string" ? payload.step : undefined,
    timestamp,
  };
}

function clearTimeline(
  options: { keepSummary?: boolean; keepArtifacts?: boolean } = {},
): void {
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

function normaliseModelCatalog(raw: unknown): ModelInfo[] {
  if (!raw || typeof raw !== "object") {
    return [];
  }
  const response = raw as Partial<ModelsResponse>;
  if (!response.models || !Array.isArray(response.models)) {
    return [];
  }
  const seen = new Set<string>();
  const result: ModelInfo[] = [];
  for (const entry of response.models) {
    if (!entry || typeof entry !== "object") {
      continue;
    }
    const idValue = (entry as Record<string, unknown>).id;
    if (typeof idValue !== "string" || !idValue.trim()) {
      continue;
    }
    const id = idValue.trim();
    if (seen.has(id)) {
      continue;
    }
    seen.add(id);
    const nameValue = (entry as Record<string, unknown>).name;
    const descriptionValue = (entry as Record<string, unknown>).description;
    const model: ModelInfo = {
      id,
      ...(typeof nameValue === "string" && nameValue.trim()
        ? { name: nameValue.trim() }
        : {}),
      ...(typeof descriptionValue === "string" && descriptionValue.trim()
        ? { description: descriptionValue.trim() }
        : {}),
    };
    for (const [key, value] of Object.entries(entry)) {
      if (key === "id" || key === "name" || key === "description") {
        continue;
      }
      model[key] = value;
    }
    result.push(model);
  }

  result.sort((a, b) => {
    const lhs = (a.name ?? a.id).toLowerCase();
    const rhs = (b.name ?? b.id).toLowerCase();
    if (lhs < rhs) return -1;
    if (lhs > rhs) return 1;
    return 0;
  });

  return result;
}

export async function initializeStores(): Promise<void> {
  try {
    const [modelsResp, defaultsResp, snapshotResp] = await Promise.all([
      fetch("/api/models"),
      fetch("/api/defaults"),
      fetch("/api/state"),
    ]);

    if (modelsResp.ok) {
      const modelsJson = (await modelsResp.json()) as unknown;
      modelCatalogStore.set(normaliseModelCatalog(modelsJson));
    } else {
      modelCatalogStore.set([]);
    }

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
    console.error("Failed to initialise web UI state", error);
    clearTimeline();
    modelCatalogStore.set([]);
  }
}

export function connectEvents(): () => void {
  if (ws) {
    ws.close();
  }

  function createWebSocket(): WebSocket {
    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const wsUrl = `${protocol}//${window.location.host}/api/ws`;
    connectionStore.set("connecting");
    return new WebSocket(wsUrl);
  }

  function onMessage(event: MessageEvent<string>): void {
    try {
      const parsed = JSON.parse(event.data) as EventPayload;
      handleEvent(parsed);
    } catch (err) {
      console.error("Failed to parse WebSocket message", err);
    }
  }

  function onOpen(): void {
    connectionStore.set("connected");
    reconnectDelay = 1000; // Reset delay on successful connection
    if (pingTimer) {
      clearInterval(pingTimer);
    }
    pingTimer = setInterval(() => {
      if (ws?.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ type: "ping", payload: {} }));
      }
    }, 20000); // 20 seconds as per requirements
  }

  function onClose(): void {
    const state = isManualDisconnect ? "disconnected" : "error";
    connectionStore.set(state);
    if (pingTimer) {
      clearInterval(pingTimer);
      pingTimer = null;
    }
    if (!isManualDisconnect) {
      // Exponential backoff with max 30 seconds
      const delay = Math.min(reconnectDelay, 30000);
      reconnectTimer = setTimeout(() => {
        reconnectTimer = null;
        ws = createWebSocket();
        ws.onopen = onOpen;
        ws.onclose = onClose;
        ws.onerror = onerror;
        ws.onmessage = onMessage;
      }, delay);
      // Increase delay for next time (exponential backoff: 1s, 2s, 4s, 8s, 16s, 30s max)
      reconnectDelay = Math.min(reconnectDelay * 2, 30000);
    } else {
      isManualDisconnect = false;
    }
  }

  function onerror(ev: Event): void {
    console.warn("WebSocket error", ev);
    connectionStore.set("error");
    if (reconnectTimer) {
      clearTimeout(reconnectTimer);
      reconnectTimer = null;
    }
    if (pingTimer) {
      clearInterval(pingTimer);
      pingTimer = null;
    }
  }

  ws = createWebSocket();
  ws.onopen = onOpen;
  ws.onclose = onClose;
  ws.onerror = onerror;
  ws.onmessage = onMessage;

  return () => {
    isManualDisconnect = true;
    if (reconnectTimer) {
      clearTimeout(reconnectTimer);
      reconnectTimer = null;
    }
    if (pingTimer) {
      clearInterval(pingTimer);
      pingTimer = null;
    }
    ws?.close();
    ws = null;
  };
}

function handleEvent(event: EventPayload): void {
  const { type, payload } = event;
  switch (type) {
    case "status":
      updateStatus(payload as unknown as StatusPayload);
      break;
    case "run_started":
      clearTimeline();
      messagesStore.update((messages) => [
        ...messages,
        toMessage(
          {
            content: "Run started",
            timestamp: Date.now(),
          },
          "system",
        ),
      ]);
      break;
    case "message":
      messagesStore.update((messages) => [
        ...messages,
        toMessage(payload as Record<string, unknown>),
      ]);
      break;
    case "judge": {
      const judgePayload = payload as Record<string, unknown>;
      const note = [
        judgePayload.asymmetry ? `Asymmetry: ${judgePayload.asymmetry}` : null,
        judgePayload.notes ? `Notes: ${judgePayload.notes}` : null,
      ]
        .filter(Boolean)
        .join(" • ");
      messagesStore.update((messages) => [
        ...messages,
        toMessage(
          {
            ...judgePayload,
            content: note || "Judge decision recorded.",
          },
          "judge",
        ),
      ]);
      break;
    }
    case "summary":
      mergeSummary(
        (payload as { summary?: Record<string, ModelSummary> }).summary,
      );
      break;
    case "run_completed":
    case "run_cancelled":
      if (payload && typeof payload === "object") {
        const art = payload as ArtifactsInfo;
        artifactsStore.set(art);
        mergeSummary(
          (payload as { summary?: Record<string, ModelSummary> }).summary,
        );
        messagesStore.update((messages) => [
          ...messages,
          toMessage(
            {
              content:
                type === "run_completed" ? "Run completed." : "Run cancelled.",
              timestamp: Date.now(),
            },
            "system",
          ),
        ]);
      }
      break;
    case "artifacts":
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
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: body ? JSON.stringify(body) : undefined,
    });
    const data = await response.json().catch(() => ({}));
    if (!response.ok) {
      return { ok: false, message: data.error ?? "Request failed" };
    }
    return { ok: true, message: data.status ?? "ok" };
  } catch (error) {
    console.error("Request failed", error);
    return { ok: false, message: "Network error" };
  }
}

export async function startRun(
  config: RunConfig,
): Promise<{ ok: boolean; message?: string }> {
  const payload = {
    ...config,
    models: config.models,
  };
  return postJson("/api/run", payload);
}

export async function pauseRun(): Promise<{ ok: boolean; message?: string }> {
  return postJson("/api/pause");
}

export async function resumeRun(): Promise<{ ok: boolean; message?: string }> {
  return postJson("/api/resume");
}

export async function cancelRun(): Promise<{ ok: boolean; message?: string }> {
  return postJson("/api/cancel");
}

export function isActiveState(state: RunState): boolean {
  return state === "running" || state === "paused" || state === "cancelling";
}
