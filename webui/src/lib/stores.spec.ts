import { get } from "svelte/store";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type {
  ArtifactsInfo,
  DefaultsResponse,
  ModelSummary,
  StatusPayload,
} from "./types";
import {
  artifactsStore,
  cancelRun,
  connectEvents,
  defaultsStore,
  initializeStores,
  isActiveState,
  messagesStore,
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

class MockEventSource {
  static instances: MockEventSource[] = [];

  url: string;
  listeners: Record<
    string,
    ((event: MessageEvent<string> | Record<string, unknown>) => void)[]
  > = {};
  closed = false;

  constructor(url: string) {
    this.url = url;
    MockEventSource.instances.push(this);
  }

  addEventListener(
    type: string,
    listener: (event: MessageEvent<string> | Record<string, unknown>) => void,
  ) {
    this.listeners[type] ||= [];
    this.listeners[type].push(listener);
  }

  close() {
    this.closed = true;
  }

  emit(name: string, data: unknown) {
    for (const listener of this.listeners[name] ?? []) {
      listener(data as MessageEvent<string>);
    }
  }
}

declare global {
  var EventSource: typeof MockEventSource;
}

describe("stores", () => {
  const originalEventSource = globalThis.EventSource;
  const originalFetch = globalThis.fetch;
  let cryptoDescriptor = Object.getOwnPropertyDescriptor(globalThis, "crypto");

  beforeEach(() => {
    statusStore.set({ ...defaultStatus });
    scoreboardStore.set({});
    messagesStore.set([]);
    artifactsStore.set(null);
    defaultsStore.set(null);
    MockEventSource.instances = [];
    vi.restoreAllMocks();
    vi.useRealTimers();
    globalThis.EventSource = MockEventSource as unknown as typeof EventSource;
    cryptoDescriptor = Object.getOwnPropertyDescriptor(globalThis, "crypto");
  });

  afterEach(() => {
    globalThis.EventSource = originalEventSource;
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

    const fetchMock = vi
      .fn()
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

    expect(fetchMock).toHaveBeenCalledTimes(2);
    expect(get(defaultsStore)).toEqual(defaults);
    expect(get(statusStore)).toMatchObject({ state: "running" });
    expect(get(scoreboardStore)).toEqual(summary);
    expect(get(artifactsStore)).toEqual(status.artifacts);
    expect(get(messagesStore)).toEqual([]);
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

    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce({ ok: false, json: async () => ({}) })
      .mockResolvedValueOnce({ ok: false, json: async () => ({}) });
    globalThis.fetch = fetchMock as unknown as typeof fetch;

    await initializeStores();

    expect(get(defaultsStore)).toBeNull();
    expect(get(scoreboardStore)).toEqual({});
    expect(get(messagesStore)).toEqual([]);
    expect(get(artifactsStore)).toBeNull();
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
    errorSpy.mockRestore();
  });

  function emit(instance: MockEventSource, name: string, payload: unknown) {
    instance.emit(name, {
      data: JSON.stringify(payload),
    } as MessageEvent<string>);
  }

  it("routes server-sent events into the Svelte stores", () => {
    const warnSpy = vi.spyOn(console, "warn").mockImplementation(() => {});
    const disconnect = connectEvents();
    const instance = MockEventSource.instances.at(-1)!;

    const statusPayload: StatusPayload = {
      ...defaultStatus,
      state: "running",
      summary,
      artifacts: { csv_path: "status.csv" },
    };

    emit(instance, "status", { type: "status", payload: statusPayload });
    expect(get(statusStore).state).toBe("running");
    expect(get(scoreboardStore)).toEqual(summary);
    expect(get(artifactsStore)).toEqual(statusPayload.artifacts);

    emit(instance, "status", { type: "unknown", payload: {} });

    emit(instance, "run_started", { type: "run_started", payload: {} });
    expect(get(messagesStore)[0].content).toBe("Run started");

    emit(instance, "message", {
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

    emit(instance, "message", {
      type: "message",
      payload: {
        role: "assistant",
      },
    });
    const latest = get(messagesStore).at(-1);
    expect(latest?.content).toBe("");

    emit(instance, "judge", {
      type: "judge",
      payload: { asymmetry: "positive", notes: "Detailed notes" },
    });
    const judgeEntry = get(messagesStore).find((msg) => msg.role === "judge");
    expect(judgeEntry?.content).toContain("Asymmetry: positive");
    expect(judgeEntry?.content).toContain("Notes: Detailed notes");

    emit(instance, "judge", { type: "judge", payload: {} });
    expect(
      get(messagesStore).some(
        (msg) => msg.content === "Judge decision recorded.",
      ),
    ).toBe(true);

    emit(instance, "summary", { type: "summary", payload: { summary } });
    expect(get(scoreboardStore)).toEqual(summary);

    const completionArtifacts: ArtifactsInfo = {
      csv_path: "complete.csv",
      runs_dir: "dir",
    };
    emit(instance, "run_completed", {
      type: "run_completed",
      payload: { ...completionArtifacts, summary },
    });
    expect(get(artifactsStore)).toMatchObject(completionArtifacts);

    emit(instance, "run_cancelled", {
      type: "run_cancelled",
      payload: { csv_path: "cancel.csv", runs_dir: "runs", summary },
    });
    expect(
      get(messagesStore).some((msg) => msg.content === "Run cancelled."),
    ).toBe(true);

    emit(instance, "artifacts", {
      type: "artifacts",
      payload: { csv_path: "direct.csv" },
    });
    expect(get(artifactsStore)).toEqual({ csv_path: "direct.csv" });

    const parseErrorSpy = vi
      .spyOn(console, "error")
      .mockImplementation(() => {});
    instance.emit("message", { data: "not-json" } as MessageEvent<string>);
    expect(parseErrorSpy).toHaveBeenCalledWith(
      "Failed to parse SSE payload",
      expect.anything(),
    );
    parseErrorSpy.mockRestore();

    instance.emit("error", { type: "error" });
    expect(warnSpy).toHaveBeenCalled();

    disconnect();
    expect(instance.closed).toBe(true);
    warnSpy.mockRestore();
  });

  it("replaces an existing event source when reconnecting", () => {
    const first = connectEvents();
    const firstInstance = MockEventSource.instances.at(-1)!;

    const second = connectEvents();
    const secondInstance = MockEventSource.instances.at(-1)!;

    expect(firstInstance.closed).toBe(true);
    expect(secondInstance).not.toBe(firstInstance);

    first();
    second();
  });

  it("falls back when crypto.randomUUID is unavailable", () => {
    Object.defineProperty(globalThis, "crypto", {
      value: undefined,
      configurable: true,
      writable: true,
    });

    const disconnect = connectEvents();
    const instance = MockEventSource.instances.at(-1)!;
    emit(instance, "message", {
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
});
