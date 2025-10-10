import { render, screen, waitFor } from '@testing-library/svelte';
import { beforeEach, describe, expect, it } from 'vitest';
import Scoreboard from './Scoreboard.svelte';
import { scoreboardStore, statusStore } from '@/lib/stores';
import type { ModelSummary, StatusPayload } from '@/lib/types';

const baseStatus: StatusPayload = {
  state: 'idle',
  error: null,
  config: null,
  started_at: null,
  finished_at: null,
};

describe('Scoreboard', () => {
  beforeEach(() => {
    statusStore.set({ ...baseStatus });
    scoreboardStore.set({});
  });

  it('shows an empty message when there is no summary data', () => {
    render(Scoreboard);
    expect(
      screen.getByText('No judgments yet. Launch a run to populate results.'),
    ).toBeInTheDocument();
    expect(screen.getByText('idle')).toBeInTheDocument();
  });

  it('renders summary cards with formatted metrics', async () => {
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
      state: 'running',
    });

    render(Scoreboard);
    scoreboardStore.set({
      'openrouter/test-model': summary,
    });

    await waitFor(() => {
      expect(screen.getByText('openrouter/test-model')).toBeInTheDocument();
    });

    expect(screen.getByText('running')).toBeInTheDocument();
    expect(screen.getByText('3 prompts')).toBeInTheDocument();
    expect(screen.getByText('2 ok / 1 issues')).toBeInTheDocument();
    expect(screen.getByText('0.87')).toBeInTheDocument();
    expect(screen.getByText('0.45')).toBeInTheDocument();
    expect(screen.getByText('20%')).toBeInTheDocument();
    expect(screen.getByText('40%')).toBeInTheDocument();
    expect(screen.getByText('credible×2, unknown×1')).toBeInTheDocument();
    expect(screen.getByText('credible×1')).toBeInTheDocument();
    expect(screen.getByText('positive×3')).toBeInTheDocument();
  });
});
