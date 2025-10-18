import { fireEvent, render, screen, waitFor } from "@testing-library/svelte";
import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import userEvent from "@testing-library/user-event";
import { tick } from "svelte";
import { within } from "@testing-library/dom";
import { beforeEach, describe, expect, it, vi } from "vitest";
import ControlPanel from "./ControlPanel.svelte";
import {
  deriveJudgeOptionsRendered,
  displayModelName,
  formatLimit,
  judgeOptionKey,
  modelCheckboxValue,
  modelKey,
  parseLimit,
  coerceJudgeModelId,
} from "./control-panel-helpers";
import * as stores from "@/lib/stores";
import type { ModelInfo, RunConfig, RunState } from "@/lib/types";

const { statusStore, defaultsStore, modelCatalogStore } = stores;

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
const controlPanelSourcePath = resolve(__dirname, "./ControlPanel.svelte");

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
    modelCatalogStore.set([
      {
        id: "qwen/qwen3-next-80b-a3b-instruct",
        name: "Qwen Next 80B",
      },
      { id: "first", name: "First" },
      { id: "second", name: "Second" },
      { id: "judge-x", name: "Judge X" },
    ]);
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
      expect(screen.getByRole("checkbox", { name: /First/ })).toBeChecked();
      expect(screen.getByRole("checkbox", { name: /Second/ })).toBeChecked();
    });

    await waitFor(() => {
      expect(
        screen.getByLabelText("Judge model") as HTMLSelectElement,
      ).toHaveValue("judge-x");
    });
    await waitFor(() => {
      expect(screen.getByLabelText("Max rounds")).toHaveValue(3);
    });
    await waitFor(() => {
      expect(screen.getByLabelText("Output directory")).toHaveValue("custom");
      const selectedGroup = screen.getByRole("group", {
        name: "Selected models",
      });
      const selectedItems = within(selectedGroup).getAllByText(/First|Second/);
      expect(
        selectedItems.map((node) =>
          node.textContent?.replace(/^[X×]\s*/, "").trim(),
        ),
      ).toEqual(["First", "Second"]);
    });
  });

  it("validates models before starting a run", async () => {
    const user = userEvent.setup();
    const startSpy = vi.spyOn(stores, "startRun");

    render(ControlPanel);

    await user.click(screen.getByRole("checkbox", { name: /Qwen Next 80B/ }));
    await user.click(screen.getByRole("button", { name: "Run" }));

    expect(startSpy).not.toHaveBeenCalled();
    expect(
      screen.getByText("Please provide at least one model slug."),
    ).toBeInTheDocument();
    expect(screen.getByText("No models selected.")).toBeInTheDocument();
  });

  it("submits parsed configuration to the backend", async () => {
    const user = userEvent.setup();
    vi.spyOn(stores, "startRun").mockResolvedValue({ ok: true, message: "ok" });

    render(ControlPanel);

    await user.click(screen.getByRole("checkbox", { name: /First/ }));

    const limitInput = screen.getByLabelText("Max rounds") as HTMLInputElement;
    await user.clear(limitInput);
    await fireEvent.input(limitInput, { target: { value: "   " } });
    expect(limitInput.value).toBe("");

    await user.click(screen.getByRole("button", { name: "Run" }));

    expect(stores.startRun).toHaveBeenCalledWith({
      models: ["qwen/qwen3-next-80b-a3b-instruct", "first"],
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

  it("keeps judge and prompt inputs within layout bounds", () => {
    render(ControlPanel);

    const source = readFileSync(controlPanelSourcePath, "utf-8");
    const rowRule = source.match(
      /\.row\s*\{[^}]*grid-template-columns:[^}]*\}/,
    );
    expect(rowRule).toBeTruthy();
    expect(rowRule?.[0]).toMatch(
      /grid-template-columns:\s*minmax\(0,\s*1.5fr\)\s*minmax\(0,\s*0.75fr\)/,
    );

    const fieldRule = source.match(/\.row select,[^}]*\.row input\s*\{[^}]*\}/);
    expect(fieldRule).toBeTruthy();
    expect(fieldRule?.[0]).toMatch(/width:\s*100%/);
  });

  it("allows removing selected models via the chip list", async () => {
    const user = userEvent.setup();
    render(ControlPanel);

    const removeButton = screen.getByRole("button", {
      name: /Deselect Qwen Next 80B/i,
    });
    await user.click(removeButton);

    expect(screen.getByText("No models selected.")).toBeInTheDocument();
    expect(
      screen.getByRole("checkbox", { name: /Qwen Next 80B/ }),
    ).not.toBeChecked();
  });

  it("shows a fallback notice when the catalog is empty", () => {
    modelCatalogStore.set([]);
    render(ControlPanel);
    expect(
      screen.getByText(
        /Model catalog unavailable\. Verify OpenRouter connectivity\./,
      ),
    ).toBeInTheDocument();
  });

  it("renders a fallback judge option when none are available", async () => {
    defaultsStore.set({
      models: [],
      judge_model: "",
      limit: 1,
      max_tokens: 8000,
      judge_max_tokens: 6000,
      temperature: 0.2,
      judge_temperature: 0.0,
      sleep_s: 0.2,
      outdir: "results",
      verbose: false,
    });
    modelCatalogStore.set([]);

    render(ControlPanel);

    const judgeSelect = screen.getByLabelText(
      "Judge model",
    ) as HTMLSelectElement;
    await waitFor(() => {
      expect(judgeSelect.options).toHaveLength(1);
      expect(judgeSelect.value).toBe("");
    });
  });

  it("reconciles judge options as the catalog changes", async () => {
    const user = userEvent.setup();
    defaultsStore.set({
      models: [],
      judge_model: "",
      limit: 1,
      max_tokens: 8000,
      judge_max_tokens: 6000,
      temperature: 0.2,
      judge_temperature: 0.0,
      sleep_s: 0.2,
      outdir: "results",
      verbose: false,
    });
    modelCatalogStore.set([]);

    render(ControlPanel);

    const judgeSelect = screen.getByLabelText(
      "Judge model",
    ) as HTMLSelectElement;
    await waitFor(() => {
      expect(judgeSelect.options).toHaveLength(1);
      expect(judgeSelect.value).toBe("");
    });

    modelCatalogStore.set([
      { id: "judge/a", name: "Judge A" },
      { id: "judge/b", name: "Judge B" },
    ]);
    await tick();
    await waitFor(() => {
      expect(judgeSelect.options).toHaveLength(2);
      expect(
        Array.from(judgeSelect.options).map((option) => option.value),
      ).toEqual(["judge/a", "judge/b"]);
    });

    modelCatalogStore.set([
      { id: "judge/b", name: "Judge B" },
      { id: "judge/a", name: "Judge A" },
    ]);
    await tick();
    await waitFor(() => {
      expect(
        Array.from(judgeSelect.options).map((option) => option.value),
      ).toEqual(["judge/b", "judge/a"]);
    });

    modelCatalogStore.set([
      { id: "judge/b", name: "Judge B" },
      { id: "judge/a", name: "Judge A" },
    ]);
    await tick();
    await waitFor(() => {
      expect(
        Array.from(judgeSelect.options).map((option) => option.value),
      ).toEqual(["judge/b", "judge/a"]);
    });

    modelCatalogStore.set([
      { id: "judge/b", name: "Judge B updated" },
      { id: "judge/a", name: "Judge A" },
    ]);
    await tick();
    await waitFor(() => {
      expect(
        Array.from(judgeSelect.options).map((option) =>
          option.textContent?.trim(),
        ),
      ).toEqual(["Judge B updated", "Judge A"]);
      expect(
        Array.from(judgeSelect.options).map((option) => option.value),
      ).toEqual(["judge/b", "judge/a"]);
    });

    const rotationSequences: ModelInfo[][] = [
      [
        { id: "judge/c", name: "Judge C" },
        { id: "judge/b", name: "Judge B updated" },
        { id: "judge/a", name: "Judge A" },
      ],
      [
        { id: "judge/a", name: "Judge A" },
        { id: "judge/c", name: "Judge C" },
        { id: "judge/b", name: "Judge B updated" },
      ],
      [
        { id: "judge/b", name: "Judge B updated" },
        { id: "judge/a", name: "Judge A" },
        { id: "judge/c", name: "Judge C" },
      ],
    ];

    for (const sequence of rotationSequences) {
      modelCatalogStore.set(sequence);
      await tick();
      await waitFor(() => {
        expect(
          Array.from(judgeSelect.options).map((option) => option.value),
        ).toEqual(sequence.map((model) => model.id));
      });
    }

    const flagsModulePath = resolve(
      __dirname,
      "../../../node_modules/svelte/src/internal/flags/index.js",
    );
    const flags = await import(pathToFileURL(flagsModulePath).href);
    flags.enable_async_mode_flag();
    try {
      const asyncSequence: ModelInfo[] = [
        { id: "judge/a", name: "Judge A" },
        { id: "judge/d", name: "Judge D" },
        { id: "judge/b", name: "Judge B updated" },
      ];
      modelCatalogStore.set(asyncSequence);
      await tick();
      await waitFor(() => {
        expect(
          Array.from(judgeSelect.options).map((option) => option.value),
        ).toEqual(asyncSequence.map((model) => model.id));
      });
    } finally {
      flags.disable_async_mode_flag();
    }

    await user.selectOptions(judgeSelect, "judge/b");
    expect(judgeSelect.value).toBe("judge/b");

    modelCatalogStore.set([{ id: "judge/b", name: "Judge B" }]);
    await tick();
    await waitFor(() => {
      expect(judgeSelect.options).toHaveLength(1);
      expect(judgeSelect.value).toBe("judge/b");
    });

    modelCatalogStore.set([]);
    await tick();
    await waitFor(() => {
      expect(judgeSelect.options).toHaveLength(1);
      expect(judgeSelect.value).toBe("judge/b");
      const [onlyOption] = Array.from(judgeSelect.options);
      expect(onlyOption.value).toBe("judge/b");
    });

    const catalogWithMissingId = [
      { name: "Anonymous" } as unknown as ModelInfo,
    ];
    modelCatalogStore.set(catalogWithMissingId);
    await tick();
    await waitFor(() => {
      expect(
        Array.from(judgeSelect.options).map((option) => option.value),
      ).toEqual(["judge/b", ""]);
      expect(judgeSelect.options[1].textContent).toBe("Anonymous");
    });

    modelCatalogStore.set([{ name: "Anonymous" } as unknown as ModelInfo]);
    await tick();
    await waitFor(() => {
      expect(
        Array.from(judgeSelect.options).map((option) => option.value),
      ).toEqual(["judge/b", ""]);
    });
  });

  it("falls back to model identifiers when names are missing", async () => {
    defaultsStore.set({
      models: ["orphan"],
      judge_model: "judge-x",
      limit: 1,
      max_tokens: 8000,
      judge_max_tokens: 6000,
      temperature: 0.2,
      judge_temperature: 0.0,
      sleep_s: 0.2,
      outdir: "results",
      verbose: false,
    });
    modelCatalogStore.set([{ id: "orphan" }]);

    render(ControlPanel);

    await waitFor(() => {
      expect(
        screen.getByRole("button", { name: "Deselect orphan" }),
      ).toBeInTheDocument();
    });

    const selectedGroup = screen.getByRole("group", {
      name: "Selected models",
    });
    expect(within(selectedGroup).getAllByText("orphan")).not.toHaveLength(0);

    const availableGroup = screen.getByRole("group", {
      name: "Available models",
    });
    expect(within(availableGroup).getAllByText("orphan")).not.toHaveLength(0);
  });

  it("positions model checkboxes inline with their labels", () => {
    render(ControlPanel);
    const source = readFileSync(controlPanelSourcePath, "utf-8");
    const modelItemRule = source.match(/\.model-item\s*\{[^}]*\}/);
    expect(modelItemRule).toBeTruthy();
    expect(modelItemRule?.[0]).toMatch(/align-items:\s*center;/);
    expect(modelItemRule?.[0]).toMatch(/flex-direction:\s*row;/);

    const checkboxRule = source.match(
      /\.model-item\s*input\[type="checkbox"\]\s*\{[^}]*\}/,
    );
    expect(checkboxRule).toBeTruthy();
    expect(checkboxRule?.[0]).toMatch(/margin:\s*0;/);
  });

  it("exposes parsing helpers and mapping utilities", () => {
    expect(parseLimit("   ")).toBeNull();
    expect(parseLimit("42")).toBe(42);
    expect(formatLimit(5)).toBe("5");
    expect(formatLimit(null)).toBe("");
    expect(formatLimit(undefined)).toBe("");
    expect(displayModelName({ id: "plain" })).toBe("plain");
    expect(displayModelName({ id: "with-name", name: "Named" })).toBe("Named");
    expect(
      deriveJudgeOptionsRendered([{ id: "m", name: "Model" }], "fallback"),
    ).toEqual([{ id: "m", name: "Model" }]);
    expect(deriveJudgeOptionsRendered([], "fallback")).toEqual([
      { id: "fallback", name: "fallback" },
    ]);
    expect(modelKey({ id: "mid" }, 2)).toBe("mid");
    expect(modelKey({ id: "" }, 3)).toBe("model-3");
    expect(judgeOptionKey({ id: "judge" }, 0)).toBe("judge-judge");
    expect(judgeOptionKey({ id: "", name: "Judge" }, 1)).toBe(
      "judge-name-Judge",
    );
    expect(judgeOptionKey({ id: "", name: "" }, 2)).toBe("judge-fallback-2");
    expect(modelCheckboxValue({ id: "abc" })).toBe("abc");
    expect(modelCheckboxValue({ id: "", name: "Readable" })).toBe("Readable");
    expect(coerceJudgeModelId("judge/a")).toBe("judge/a");
    expect(coerceJudgeModelId(undefined)).toBe("");
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

  it("renders judge options while async mode defers appends", async () => {
    const flagsModulePath = resolve(
      __dirname,
      "../../../node_modules/svelte/src/internal/flags/index.js",
    );
    const flags = await import(pathToFileURL(flagsModulePath).href);
    flags.enable_async_mode_flag();
    try {
      modelCatalogStore.set([
        { id: "judge/a", name: "Judge A" },
        { id: "judge/b", name: "Judge B" },
      ]);
      const AsyncControlPanel = (
        (await import(
          /* @vite-ignore */ "./ControlPanel.svelte?async-render"
        )) as unknown as { default: typeof ControlPanel }
      ).default;
      const { unmount } = render(AsyncControlPanel);
      await waitFor(() => {
        expect(screen.getByLabelText("Judge model")).toBeInTheDocument();
      });
      unmount();
    } finally {
      flags.disable_async_mode_flag();
    }
  });

  it("manually exercises the deferred append branch", async () => {
    const base = "../../../node_modules/svelte/src/internal";
    const load = async (rel: string) =>
      import(pathToFileURL(resolve(__dirname, `${base}/${rel}`)).href);

    const flags = await load("flags/index.js");
    const runtime = await load("client/runtime.js");
    const constants = await load("client/constants.js");
    const operations = await load("client/dom/operations.js");

    flags.enable_async_mode_flag();
    try {
      const previousEffect = runtime.active_effect;
      runtime.set_active_effect({ f: constants.EFFECT_RAN });
      expect(operations.should_defer_append()).toBe(true);
      runtime.set_active_effect(previousEffect);
    } finally {
      flags.disable_async_mode_flag();
    }
  });
});
