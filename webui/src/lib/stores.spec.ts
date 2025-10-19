import { get } from "svelte/store";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { MockInstance } from "vitest";
import type {
  ArtifactsInfo,
  DefaultsResponse,
  EventPayload,
  ModelSummary,
  StatusPayload,
} from "./types";
import {
  artifactsStore,
  cancelRun,
  connectEvents,
  connectionStore,
  defaultsStore,
  initializeStores,
  isActiveState,
  messagesStore,
  modelCatalogStore,
  pauseRun,
  resumeRun,
  scoreboardStore,
  startRun,
  statusStore,
} from "./stores";

const defaultStatus: StatusPayload = {
  state: "idle",
  error: null,
  config: null,
  started_at: null,
  finished_at: null,
};

class MockWebSocket {
  static instances: MockWebSocket[] = [];
  static readonly CONNECTING = 0;
  static readonly OPEN = 1;
  static readonly CLOSING = 2;
  static readonly CLOSED = 3;

  url: string;
  readyState = MockWebSocket.CONNECTING;
  closed = false;
  onopen: ((event: Event) => void) | null = null;
  onclose: ((event: Event) => void) | null = null;
  onmessage: ((event: MessageEvent<string>) => void) | null = null;
  onerror: ((event: Event) => void) | null = null;

  constructor(url: string) {
    this.url = url;
    MockWebSocket.instances.push(this);
  }

  send(data: string): void {
    void data; // no-op for tests
  }

  close(): void {
    this.closed = true;
    this.readyState = MockWebSocket.CLOSED;
    this.onclose?.({ type: "close" } as Event);
  }

  serverClose(): void {
    this.readyState = MockWebSocket.CLOSED;
    this.onclose?.({ type: "close" } as Event);
  }

  emit<T>(payload: EventPayload<T>): void {
    this.onmessage?.({ data: JSON.stringify(payload) } as MessageEvent<string>);
  }

  emitRaw(data: string): void {
    this.onmessage?.({ data } as MessageEvent<string>);
  }

  triggerError(event: Event = { type: "error" } as Event): void {
    this.onerror?.(event);
  }

  triggerOpen(): void {
    this.readyState = MockWebSocket.OPEN;
    this.onopen?.({ type: "open" } as Event);
  }
}

describe("stores", () => {
  const originalWebSocket = globalThis.WebSocket;
  const originalFetch = globalThis.fetch;
  let cryptoDescriptor = Object.getOwnPropertyDescriptor(globalThis, "crypto");
  let setIntervalSpy!: MockInstance<typeof globalThis.setInterval>;
  let clearIntervalSpy!: MockInstance<typeof globalThis.clearInterval>;
  const useFakeTimersWithTimerSpies = () => {
    setIntervalSpy.mockRestore();
    clearIntervalSpy.mockRestore();
    vi.useFakeTimers();
    setIntervalSpy = vi.spyOn(globalThis, "setInterval");
    clearIntervalSpy = vi.spyOn(globalThis, "clearInterval");
  };

  beforeEach(() => {
    statusStore.set({ ...defaultStatus });
    scoreboardStore.set({});
    messagesStore.set([]);
    artifactsStore.set(null);
    defaultsStore.set(null);
    modelCatalogStore.set([]);
    MockWebSocket.instances = [];
    vi.restoreAllMocks();
    vi.useRealTimers();
    globalThis.WebSocket = MockWebSocket as unknown as typeof WebSocket;
    cryptoDescriptor = Object.getOwnPropertyDescriptor(globalThis, "crypto");
    setIntervalSpy = vi.spyOn(globalThis, "setInterval");
    clearIntervalSpy = vi.spyOn(globalThis, "clearInterval");
  });

  afterEach(() => {
    if (originalWebSocket) {
      globalThis.WebSocket = originalWebSocket;
    } else {
      delete (globalThis as { WebSocket?: typeof globalThis.WebSocket })
        .WebSocket;
    }
    globalThis.fetch = originalFetch;
    if (cryptoDescriptor) {
      Object.defineProperty(globalThis, "crypto", cryptoDescriptor);
    } else {
      delete (globalThis as { crypto?: Crypto }).crypto;
    }
  });

  const summary: Record<string, ModelSummary> = {
    "test/model": {
      total: 3,
      ok: 2,
      issues: 1,
      avg_initial_completeness: 0.5,
      avg_followup_completeness: 0.25,
      initial_refusal_rate: 0.1,
      followup_refusal_rate: 0.2,
      initial_sourcing_counts: { credible: 2, unreliable: 0 },
      followup_sourcing_counts: { credible: 1 },
      asymmetry_counts: { neutral: 3 },
      error_counts: {},
    },
  };

  it("initializes stores with defaults and snapshot payloads", async () => {
    const defaults: DefaultsResponse = {
      models: ["model-a"],
      judge_model: "judge-a",
      limit: 5,
      max_tokens: 1000,
      judge_max_tokens: 500,
      temperature: 0.5,
      judge_temperature: 0,
      sleep_s: 0.1,
      outdir: "out",
      verbose: true,
    };
    const status: StatusPayload = {
      ...defaultStatus,
      state: "running",
      summary,
      artifacts: { csv_path: "file.csv", runs_dir: "dir" },
    };
    const catalog = {
      models: [
        { id: "model-a", name: "Model A" },
        { id: "model-b", name: "Model B" },
      ],
    };

    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce({
        ok: true,
        json: async () => catalog,
      })
      .mockResolvedValueOnce({
        ok: true,
        json: async () => defaults,
      })
      .mockResolvedValueOnce({
        ok: true,
        json: async () => ({ status, history: [] }),
      });
    globalThis.fetch = fetchMock as unknown as typeof fetch;

    await initializeStores();

    expect(fetchMock).toHaveBeenCalledTimes(3);
    expect(get(defaultsStore)).toEqual(defaults);
    expect(get(statusStore)).toMatchObject({ state: "running" });
    expect(get(scoreboardStore)).toEqual(summary);
    expect(get(artifactsStore)).toEqual(status.artifacts);
    expect(get(messagesStore)).toEqual([]);
    expect(get(modelCatalogStore)).toEqual([
      { id: "model-a", name: "Model A" },
      { id: "model-b", name: "Model B" },
    ]);
  });

  it("clears local state when snapshot fetch fails", async () => {
    scoreboardStore.set(summary);
    messagesStore.set([
      {
        id: "msg",
        role: "system",
        content: "existing",
        timestamp: Date.now(),
      },
    ]);
    artifactsStore.set({ csv_path: "existing.csv" });
    modelCatalogStore.set([{ id: "existing", name: "Existing" }]);

    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce({ ok: false, json: async () => ({}) })
      .mockResolvedValueOnce({ ok: false, json: async () => ({}) })
      .mockResolvedValueOnce({ ok: false, json: async () => ({}) });
    globalThis.fetch = fetchMock as unknown as typeof fetch;

    await initializeStores();

    expect(fetchMock).toHaveBeenCalledTimes(3);
    expect(get(defaultsStore)).toBeNull();
    expect(get(scoreboardStore)).toEqual({});
    expect(get(messagesStore)).toEqual([]);
    expect(get(artifactsStore)).toBeNull();
    expect(get(modelCatalogStore)).toEqual([]);
  });

  it("recovers from initialization errors", async () => {
    const error = new Error("network fail");
    const errorSpy = vi.spyOn(console, "error").mockImplementation(() => {});
    globalThis.fetch = vi
      .fn()
      .mockRejectedValue(error) as unknown as typeof fetch;

    await initializeStores();

    expect(errorSpy).toHaveBeenCalledWith(
      "Failed to initialise web UI state",
      error,
    );
    expect(get(scoreboardStore)).toEqual({});
    expect(get(artifactsStore)).toBeNull();
    expect(get(modelCatalogStore)).toEqual([]);
    errorSpy.mockRestore();
  });

  it("handles non-object model catalogs", async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce({ ok: true, json: async () => null })
      .mockResolvedValueOnce({ ok: false, json: async () => ({}) })
      .mockResolvedValueOnce({
        ok: true,
        json: async () => ({ status: { ...defaultStatus }, history: [] }),
      });

    globalThis.fetch = fetchMock as unknown as typeof fetch;

    await initializeStores();

    expect(fetchMock).toHaveBeenCalledTimes(3);
    expect(get(modelCatalogStore)).toEqual([]);
  });

  it("normalises catalog entries and preserves metadata", async () => {
    const catalog = {
      models: [
        null,
        { id: "   " },
        {
          id: "model-c",
          name: " Charlie  ",
          description: "  Insight  ",
          extra: "keep",
        },
        { id: "model-a", name: "Alpha", description: "Leading" },
        { id: "model-b", name: "  beta ", description: "   " },
        { id: "model-a", name: "Duplicate" },
        { id: "model-d", other: "value" },
        { id: "model-e", name: "", description: null },
        { id: "model-f", description: "Desc" },
        { id: "model-g", name: "Zed" },
        { id: "model-h", name: "ZED", description: "Upper" },
      ],
    } satisfies Record<string, unknown>;

    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce({ ok: true, json: async () => catalog })
      .mockResolvedValueOnce({ ok: false, json: async () => ({}) })
      .mockResolvedValueOnce({
        ok: true,
        json: async () => ({ status: { ...defaultStatus }, history: [] }),
      });

    globalThis.fetch = fetchMock as unknown as typeof fetch;

    await initializeStores();

    expect(get(modelCatalogStore)).toEqual([
      { id: "model-a", name: "Alpha", description: "Leading" },
      { id: "model-b", name: "beta" },
      { id: "model-c", name: "Charlie", description: "Insight", extra: "keep" },
      { id: "model-d", other: "value" },
      { id: "model-e" },
      { id: "model-f", description: "Desc" },
      { id: "model-g", name: "Zed" },
      { id: "model-h", name: "ZED", description: "Upper" },
    ]);
  });

  it("returns an empty catalog when the models list is missing", async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce({ ok: true, json: async () => ({}) })
      .mockResolvedValueOnce({ ok: false, json: async () => ({}) })
      .mockResolvedValueOnce({
        ok: true,
        json: async () => ({ status: { ...defaultStatus }, history: [] }),
      });

    globalThis.fetch = fetchMock as unknown as typeof fetch;

    await initializeStores();

    expect(get(modelCatalogStore)).toEqual([]);
  });

  function emit<T>(instance: MockWebSocket, payload: EventPayload<T>) {
    instance.emit(payload);
  }

  it("routes server-sent events into the Svelte stores", () => {
    const warnSpy = vi.spyOn(console, "warn").mockImplementation(() => {});
    const disconnect = connectEvents();
    const instance = MockWebSocket.instances.at(-1)!;

    const statusPayload: StatusPayload = {
      ...defaultStatus,
      state: "running",
      summary,
      artifacts: { csv_path: "status.csv" },
    };

    emit(instance, { type: "status", payload: statusPayload });
    expect(get(statusStore).state).toBe("running");
    expect(get(scoreboardStore)).toEqual(summary);
    expect(get(artifactsStore)).toEqual(statusPayload.artifacts);

    emit(instance, { type: "unknown", payload: {} });

    emit(instance, { type: "run_started", payload: {} });
    expect(get(messagesStore)[0].content).toBe("Run started");

    emit(instance, {
      type: "message",
      payload: {
        role: "assistant",
        content: "Reply",
        model: "m",
        prompt_index: 2,
        step: "analysis",
      },
    });
    expect(get(messagesStore)).toHaveLength(2);

    emit(instance, {
      type: "message",
      payload: {
        role: "assistant",
      },
    });
    const latest = get(messagesStore).at(-1);
    expect(latest?.content).toBe("");

    emit(instance, {
      type: "judge",
      payload: { asymmetry: "positive", notes: "Detailed notes" },
    });
    const judgeEntry = get(messagesStore).find((msg) => msg.role === "judge");
    expect(judgeEntry?.content).toContain("Asymmetry: positive");
    expect(judgeEntry?.content).toContain("Notes: Detailed notes");

    emit(instance, { type: "judge", payload: {} });
    expect(
      get(messagesStore).some(
        (msg) => msg.content === "Judge decision recorded.",
      ),
    ).toBe(true);

    emit(instance, { type: "summary", payload: { summary } });
    expect(get(scoreboardStore)).toEqual(summary);

    const completionArtifacts: ArtifactsInfo = {
      csv_path: "complete.csv",
      runs_dir: "dir",
    };
    emit(instance, {
      type: "run_completed",
      payload: { ...completionArtifacts, summary },
    });
    expect(get(artifactsStore)).toMatchObject(completionArtifacts);

    emit(instance, {
      type: "run_cancelled",
      payload: { csv_path: "cancel.csv", runs_dir: "runs", summary },
    });
    expect(
      get(messagesStore).some((msg) => msg.content === "Run cancelled."),
    ).toBe(true);

    emit(instance, {
      type: "artifacts",
      payload: { csv_path: "direct.csv" },
    });
    expect(get(artifactsStore)).toEqual({ csv_path: "direct.csv" });

    const parseErrorSpy = vi
      .spyOn(console, "error")
      .mockImplementation(() => {});
    instance.emitRaw("not-json");
    expect(parseErrorSpy).toHaveBeenCalledWith(
      "Failed to parse WebSocket message",
      expect.anything(),
    );
    parseErrorSpy.mockRestore();

    instance.triggerError({ type: "error" } as Event);
    expect(warnSpy).toHaveBeenCalled();

    disconnect();
    expect(instance.closed).toBe(true);
    warnSpy.mockRestore();
  });

  it("replaces an existing event source when reconnecting", () => {
    const first = connectEvents();
    const firstInstance = MockWebSocket.instances.at(-1)!;

    const second = connectEvents();
    const secondInstance = MockWebSocket.instances.at(-1)!;

    expect(firstInstance.closed).toBe(true);
    expect(secondInstance).not.toBe(firstInstance);

    first();
    second();
  });

  it("restarts the ping timer on repeated open events", () => {
    const disconnect = connectEvents();
    const instance = MockWebSocket.instances.at(-1)!;

    instance.triggerOpen();
    expect(setIntervalSpy).toHaveBeenCalledTimes(1);
    expect(clearIntervalSpy).not.toHaveBeenCalled();

    instance.triggerOpen();
    expect(clearIntervalSpy).toHaveBeenCalledTimes(1);
    expect(setIntervalSpy).toHaveBeenCalledTimes(2);

    disconnect();
  });

  it("uses wss when the page is served over https", () => {
    const originalLocation = window.location;
    const secureLocation = {
      protocol: "https:",
      host: "secure.example",
    } as Location;
    Object.defineProperty(window, "location", {
      value: secureLocation,
      configurable: true,
      writable: true,
    });

    const disconnect = connectEvents();
    const instance = MockWebSocket.instances.at(-1)!;

    try {
      expect(instance.url).toBe("wss://secure.example/api/ws");
    } finally {
      disconnect();
      Object.defineProperty(window, "location", {
        value: originalLocation,
        configurable: true,
        writable: true,
      });
    }
  });

  it("falls back when crypto.randomUUID is unavailable", () => {
    Object.defineProperty(globalThis, "crypto", {
      value: undefined,
      configurable: true,
      writable: true,
    });

    const disconnect = connectEvents();
    const instance = MockWebSocket.instances.at(-1)!;
    emit(instance, {
      type: "message",
      payload: { content: "no crypto" },
    });
    const [message] = get(messagesStore).slice(-1);
    expect(message.id).toBeTypeOf("string");
    disconnect();
  });

  it("posts JSON payloads for run lifecycle helpers", async () => {
    const fetchSpy = vi
      .fn()
      .mockResolvedValueOnce({
        ok: true,
        json: async () => ({ status: "ok" }),
      })
      .mockResolvedValueOnce({
        ok: true,
        json: async () => ({}),
      })
      .mockResolvedValueOnce({
        ok: false,
        json: async () => ({ error: "bad" }),
      })
      .mockResolvedValueOnce({
        ok: false,
        json: async () => ({}),
      })
      .mockRejectedValueOnce(new Error("offline"));
    globalThis.fetch = fetchSpy as unknown as typeof fetch;
    const errorSpy = vi.spyOn(console, "error").mockImplementation(() => {});

    await expect(
      startRun({
        models: ["a"],
        judge_model: "judge",
        limit: 1,
        max_tokens: 10,
        judge_max_tokens: 10,
        temperature: 0,
        judge_temperature: 0,
        sleep_s: 0.1,
        outdir: "out",
        verbose: false,
      }),
    ).resolves.toEqual({ ok: true, message: "ok" });

    await expect(
      startRun({
        models: ["a"],
        judge_model: "judge",
        limit: 1,
        max_tokens: 10,
        judge_max_tokens: 10,
        temperature: 0,
        judge_temperature: 0,
        sleep_s: 0.1,
        outdir: "out",
        verbose: false,
      }),
    ).resolves.toEqual({ ok: true, message: "ok" });

    await expect(pauseRun()).resolves.toEqual({ ok: false, message: "bad" });

    await expect(resumeRun()).resolves.toEqual({
      ok: false,
      message: "Request failed",
    });

    await expect(cancelRun()).resolves.toEqual({
      ok: false,
      message: "Network error",
    });

    expect(fetchSpy).toHaveBeenCalledWith("/api/run", expect.any(Object));
    expect(fetchSpy).toHaveBeenCalledWith("/api/pause", expect.any(Object));
    expect(fetchSpy).toHaveBeenCalledWith("/api/resume", expect.any(Object));
    expect(fetchSpy).toHaveBeenCalledWith("/api/cancel", expect.any(Object));
    expect(errorSpy).toHaveBeenCalled();
    errorSpy.mockRestore();
  });

  it("identifies active run states", () => {
    expect(isActiveState("running")).toBe(true);
    expect(isActiveState("paused")).toBe(true);
    expect(isActiveState("cancelling")).toBe(true);
    expect(isActiveState("completed")).toBe(false);
  });

  it("manages WebSocket connection states", () => {
    const disconnect = connectEvents();
    const instance = MockWebSocket.instances.at(-1)!;

    expect(get(connectionStore)).toBe("connecting");

    instance.triggerOpen();
    expect(get(connectionStore)).toBe("connected");

    instance.serverClose();
    expect(get(connectionStore)).toBe("error");

    disconnect();
  });

  it("reconnects on server close with exponential backoff", async () => {
    useFakeTimersWithTimerSpies();
    const disconnect = connectEvents();
    const firstInstance = MockWebSocket.instances.at(-1)!;

    firstInstance.triggerOpen();
    expect(get(connectionStore)).toBe("connected");

    firstInstance.serverClose();
    expect(get(connectionStore)).toBe("error");

    // First reconnect attempt after 1 second
    await vi.advanceTimersByTimeAsync(1000);
    const secondInstance = MockWebSocket.instances.at(-1)!;
    expect(secondInstance).not.toBe(firstInstance);

    secondInstance.triggerOpen();
    expect(get(connectionStore)).toBe("connected");

    secondInstance.serverClose();
    expect(get(connectionStore)).toBe("error");

    // Second reconnect attempt after 2 seconds
    await vi.advanceTimersByTimeAsync(2000);
    const thirdInstance = MockWebSocket.instances.at(-1)!;
    expect(thirdInstance).not.toBe(secondInstance);

    disconnect();
    vi.useRealTimers();
  });

  it("stops reconnecting on manual disconnect", () => {
    useFakeTimersWithTimerSpies();
    const disconnect = connectEvents();
    const instance = MockWebSocket.instances.at(-1)!;

    instance.triggerOpen();
    instance.serverClose();

    disconnect(); // Manual disconnect

    // Advance time - should not create new connections
    vi.advanceTimersByTime(1000);
    expect(MockWebSocket.instances).toHaveLength(1);

    vi.useRealTimers();
  });

  it("sends ping messages every 20 seconds when connected", () => {
    useFakeTimersWithTimerSpies();
    const sendSpy = vi.spyOn(MockWebSocket.prototype, "send");
    const disconnect = connectEvents();
    const instance = MockWebSocket.instances.at(-1)!;

    instance.triggerOpen();

    // Advance 20 seconds
    vi.advanceTimersByTime(20000);
    expect(sendSpy).toHaveBeenCalledWith(
      JSON.stringify({ type: "ping", payload: {} }),
    );

    // Advance another 20 seconds
    vi.advanceTimersByTime(20000);
    expect(sendSpy).toHaveBeenCalledTimes(2);

    disconnect();
    vi.useRealTimers();
  });

  it("cleans up ping timer on disconnect", () => {
    useFakeTimersWithTimerSpies();
    const disconnect = connectEvents();
    const instance = MockWebSocket.instances.at(-1)!;

    instance.triggerOpen();
    expect(setIntervalSpy).toHaveBeenCalledTimes(1);

    disconnect();
    expect(clearIntervalSpy).toHaveBeenCalledTimes(1);

    vi.useRealTimers();
  });

  it("cleans up ping timer on connection error", () => {
    useFakeTimersWithTimerSpies();
    const disconnect = connectEvents();
    const instance = MockWebSocket.instances.at(-1)!;

    instance.triggerOpen();
    expect(setIntervalSpy).toHaveBeenCalledTimes(1);

    instance.triggerError();
    expect(clearIntervalSpy).toHaveBeenCalledTimes(1);

    disconnect();
    vi.useRealTimers();
  });

  it("cleans up ping timer on server close", () => {
    useFakeTimersWithTimerSpies();
    const disconnect = connectEvents();
    const instance = MockWebSocket.instances.at(-1)!;

    instance.triggerOpen();
    expect(setIntervalSpy).toHaveBeenCalledTimes(1);

    instance.serverClose();
    expect(clearIntervalSpy).toHaveBeenCalledTimes(1);

    disconnect();
    vi.useRealTimers();
  });

  it("clears a pending reconnect timer when an error occurs", () => {
    useFakeTimersWithTimerSpies();
    const clearTimeoutSpy = vi.spyOn(globalThis, "clearTimeout");
    const disconnect = connectEvents();
    const instance = MockWebSocket.instances.at(-1)!;

    instance.triggerOpen();
    instance.serverClose();

    instance.triggerError();
    expect(clearTimeoutSpy).toHaveBeenCalledTimes(1);

    clearTimeoutSpy.mockRestore();
    disconnect();
    vi.useRealTimers();
  });
});
