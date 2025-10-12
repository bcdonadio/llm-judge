import { render, screen, waitFor } from "@testing-library/svelte";
import { beforeEach, describe, expect, it } from "vitest";
import Scoreboard from "./Scoreboard.svelte";
import { scoreboardStore, statusStore } from "@/lib/stores";
import type { ModelSummary, StatusPayload } from "@/lib/types";

const baseStatus: StatusPayload = {
  state: "idle",
  error: null,
  config: null,
  started_at: null,
  finished_at: null,
};

describe("Scoreboard", () => {
  beforeEach(() => {
    statusStore.set({ ...baseStatus });
    scoreboardStore.set({});
  });

  it("shows an empty message when there is no summary data", () => {
    render(Scoreboard);
    expect(
      screen.getByText("No judgments yet. Launch a run to populate results."),
    ).toBeInTheDocument();
    expect(screen.getByText("idle")).toBeInTheDocument();
  });

  it("renders summary cards with formatted metrics", async () => {
    const summary: ModelSummary = {
      total: 3,
      ok: 2,
      issues: 1,
      avg_initial_completeness: 0.87,
      avg_followup_completeness: 0.45,
      initial_refusal_rate: 0.2,
      followup_refusal_rate: 0.4,
      initial_sourcing_counts: {
        credible: 2,
        unknown: 1,
        skipped: 0,
      },
      followup_sourcing_counts: {
        credible: 1,
      },
      asymmetry_counts: {
        positive: 3,
      },
      error_counts: {},
    };

    statusStore.set({
      ...baseStatus,
      state: "running",
    });

    render(Scoreboard);
    scoreboardStore.set({
      "openrouter/test-model": summary,
    });

    await waitFor(() => {
      expect(screen.getByText("openrouter/test-model")).toBeInTheDocument();
    });

    expect(screen.getByText("running")).toBeInTheDocument();
    expect(screen.getByText("3 prompts")).toBeInTheDocument();
    expect(screen.getByText("2 ok / 1 issues")).toBeInTheDocument();
    expect(screen.getByText("0.87")).toBeInTheDocument();
    expect(screen.getByText("0.45")).toBeInTheDocument();
    expect(screen.getByText("20%")).toBeInTheDocument();
    expect(screen.getByText("40%")).toBeInTheDocument();
    expect(screen.getByText("credible×2, unknown×1")).toBeInTheDocument();
    expect(screen.queryByText("skipped×0")).not.toBeInTheDocument();
    expect(screen.getByText("credible×1")).toBeInTheDocument();
    expect(screen.getByText("positive×3")).toBeInTheDocument();

    const updatedSummary: ModelSummary = {
      ...summary,
      total: 4,
      ok: 3,
      issues: 1,
      initial_refusal_rate: 0.1,
      followup_refusal_rate: 0.5,
    };

    scoreboardStore.set({
      "openrouter/test-model": updatedSummary,
    });

    await waitFor(() => {
      expect(screen.getByText("4 prompts")).toBeInTheDocument();
    });
    expect(screen.getByText("3 ok / 1 issues")).toBeInTheDocument();
    expect(screen.getByText("10%")).toBeInTheDocument();
    expect(screen.getByText("50%")).toBeInTheDocument();

    const secondUpdate: ModelSummary = {
      ...updatedSummary,
      issues: 2,
    };

    scoreboardStore.set({
      "openrouter/test-model": secondUpdate,
    });

    await waitFor(() => {
      expect(screen.getByText("3 ok / 2 issues")).toBeInTheDocument();
    });

    scoreboardStore.set({});
    await waitFor(() => {
      expect(
        screen.getByText("No judgments yet. Launch a run to populate results."),
      ).toBeInTheDocument();
    });
  });

  it("falls back to safe defaults for missing metrics", async () => {
    const summary: ModelSummary = {
      total: 1,
      ok: 0,
      issues: 1,
      avg_initial_completeness: Number.NaN,
      avg_followup_completeness: Number.NaN,
      initial_refusal_rate: Number.NaN,
      followup_refusal_rate: Number.NaN,
      initial_sourcing_counts: {},
      followup_sourcing_counts: undefined as unknown as Record<string, number>,
      asymmetry_counts: {},
      error_counts: {},
    };

    statusStore.set({
      ...baseStatus,
      state: "completed",
    });

    render(Scoreboard);
    scoreboardStore.set({ failing: summary });

    await waitFor(() => {
      expect(screen.getByText("failing")).toBeInTheDocument();
    });

    expect(screen.getByText("1 prompts")).toBeInTheDocument();
    expect(screen.getByText("0 ok / 1 issues")).toBeInTheDocument();
    expect(screen.queryAllByText("0.00")).not.toHaveLength(0);
    expect(screen.getAllByText("0%")).toHaveLength(2);
    expect(screen.getAllByText("n/a")).not.toHaveLength(0);
  });
});
