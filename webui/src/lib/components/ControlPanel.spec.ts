import { fireEvent, render, screen, waitFor } from "@testing-library/svelte";
import userEvent from "@testing-library/user-event";
import { tick } from "svelte";
import { beforeEach, describe, expect, it, vi } from "vitest";
import ControlPanel from "./ControlPanel.svelte";
import { formatLimit, parseLimit, parseModels } from "./control-panel-helpers";
import * as stores from "@/lib/stores";
import type { RunConfig, RunState } from "@/lib/types";

const { statusStore, defaultsStore } = stores;

const baseStatus = {
  state: "idle" as const,
  error: null,
  config: null,
  started_at: null,
  finished_at: null,
};

describe("ControlPanel", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
    statusStore.set({ ...baseStatus });
    defaultsStore.set(null);
  });

  it("hydrates fields from the defaults store once", async () => {
    defaultsStore.set({
      models: ["first", "second"],
      judge_model: "judge-x",
      limit: 3,
      max_tokens: 10_000,
      judge_max_tokens: 9_000,
      temperature: 0.5,
      judge_temperature: 0.1,
      sleep_s: 0.4,
      outdir: "custom",
      verbose: true,
    });

    render(ControlPanel);

    await waitFor(() => {
      expect(screen.getByLabelText("Models to evaluate")).toHaveValue(
        "first\nsecond",
      );
    });

    await waitFor(() => {
      expect(screen.getByLabelText("Judge model")).toHaveValue("judge-x");
    });
    await waitFor(() => {
      expect(screen.getByLabelText("Prompt limit")).toHaveValue(3);
    });
    await waitFor(() => {
      expect(screen.getByLabelText("Output directory")).toHaveValue("custom");
    });
  });

  it("validates models before starting a run", async () => {
    const user = userEvent.setup();
    const startSpy = vi.spyOn(stores, "startRun");

    render(ControlPanel);

    await user.clear(screen.getByLabelText("Models to evaluate"));
    await user.click(screen.getByRole("button", { name: "Run" }));

    expect(startSpy).not.toHaveBeenCalled();
    expect(
      screen.getByText("Please provide at least one model slug."),
    ).toBeInTheDocument();
  });

  it("submits parsed configuration to the backend", async () => {
    const user = userEvent.setup();
    vi.spyOn(stores, "startRun").mockResolvedValue({ ok: true, message: "ok" });

    render(ControlPanel);

    const modelsTextarea = screen.getByLabelText("Models to evaluate");
    await user.clear(modelsTextarea);
    await user.type(modelsTextarea, "model-a, model-b");

    const limitInput = screen.getByLabelText(
      "Prompt limit",
    ) as HTMLInputElement;
    await user.clear(limitInput);
    await fireEvent.input(limitInput, { target: { value: "   " } });
    expect(limitInput.value).toBe("");

    await user.click(screen.getByRole("button", { name: "Run" }));

    expect(stores.startRun).toHaveBeenCalledWith({
      models: ["model-a", "model-b"],
      judge_model: "x-ai/grok-4-fast",
      limit: null,
      max_tokens: 8000,
      judge_max_tokens: 6000,
      temperature: 0.2,
      judge_temperature: 0.0,
      sleep_s: 0.2,
      outdir: "results",
      verbose: false,
    });

    expect(screen.getByText("Evaluation run triggered.")).toBeInTheDocument();
    expect(screen.getByText("Evaluation run triggered.")).toBeInTheDocument();
  });

  it("exposes parsing helpers for models and limits", () => {
    expect(parseModels("one, two\nthree")).toEqual(["one", "two", "three"]);
    expect(parseLimit("   ")).toBeNull();
    expect(parseLimit("42")).toBe(42);
    expect(formatLimit(5)).toBe("5");
    expect(formatLimit(null)).toBe("");
    expect(formatLimit(undefined)).toBe("");
  });

  it("handles backend errors for lifecycle actions", async () => {
    vi.useFakeTimers();
    const user = userEvent.setup({
      advanceTimers: async (ms) => {
        await vi.advanceTimersByTimeAsync(ms);
      },
    });

    try {
      vi.spyOn(stores, "startRun").mockResolvedValue({ ok: false });
      vi.spyOn(stores, "pauseRun")
        .mockResolvedValueOnce({ ok: false })
        .mockResolvedValueOnce({ ok: true });
      vi.spyOn(stores, "resumeRun")
        .mockResolvedValueOnce({ ok: false })
        .mockResolvedValueOnce({ ok: true });
      vi.spyOn(stores, "cancelRun")
        .mockResolvedValueOnce({ ok: false })
        .mockResolvedValueOnce({ ok: true });

      render(ControlPanel);

      await user.type(screen.getByLabelText("Models to evaluate"), "single");
      await user.click(screen.getByRole("button", { name: "Run" }));
      expect(screen.getByText("Unable to start run")).toBeInTheDocument();

      await vi.advanceTimersByTimeAsync(5000);
      expect(screen.queryByText("Unable to start run")).not.toBeInTheDocument();

      statusStore.set({ ...baseStatus, state: "running" });
      await tick();
      const pauseButton = screen.getByRole("button", { name: "Pause" });
      await user.click(pauseButton);
      expect(screen.getByText("Pause failed")).toBeInTheDocument();

      await user.click(pauseButton);
      expect(screen.getByText("Run paused.")).toBeInTheDocument();

      statusStore.set({ ...baseStatus, state: "paused" });
      await tick();
      const resumeButton = screen.getByRole("button", { name: "Resume" });
      await user.click(resumeButton);
      expect(screen.getByText("Resume failed")).toBeInTheDocument();

      await user.click(resumeButton);
      expect(screen.getByText("Run resumed.")).toBeInTheDocument();

      statusStore.set({ ...baseStatus, state: "running" });
      await tick();
      const cancelButton = screen.getByRole("button", { name: "Cancel" });
      await user.click(cancelButton);
      expect(screen.getByText("Cancel failed")).toBeInTheDocument();

      await user.click(cancelButton);
      expect(screen.getByText("Cancellation requested.")).toBeInTheDocument();
    } finally {
      vi.useRealTimers();
    }
  });

  it("disables buttons according to the current run state", async () => {
    statusStore.set({ ...baseStatus, state: "running" });
    render(ControlPanel);

    expect(screen.getByRole("button", { name: "Run" })).toBeDisabled();
    expect(screen.getByRole("button", { name: "Pause" })).not.toBeDisabled();
    expect(screen.getByRole("button", { name: "Resume" })).toBeDisabled();
    expect(screen.getByRole("button", { name: "Cancel" })).not.toBeDisabled();

    statusStore.set({
      ...baseStatus,
      state: "paused",
      config: { models: ["one"] } as RunConfig,
    });
    await tick();
    expect(screen.getByRole("button", { name: "Run" })).toBeDisabled();
    expect(screen.getByRole("button", { name: "Pause" })).toBeDisabled();
    expect(screen.getByRole("button", { name: "Resume" })).not.toBeDisabled();
    expect(screen.getByRole("button", { name: "Cancel" })).not.toBeDisabled();
  });

  it("falls back to the idle state when status data is missing", async () => {
    statusStore.set({ ...baseStatus, state: undefined as unknown as RunState });
    render(ControlPanel);
    expect(screen.getByText("idle")).toBeInTheDocument();
  });
});
