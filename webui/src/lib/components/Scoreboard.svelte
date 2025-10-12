<script lang="ts">
  import { derived } from "svelte/store";
  import { scoreboardStore, statusStore } from "@/lib/stores";
  import type { ModelSummary } from "@/lib/types";

  interface ScoreboardEntry {
    model: string;
    totalLabel: string;
    successLabel: string;
    initialCompleteness: string;
    followupCompleteness: string;
    initialRefusal: string;
    followupRefusal: string;
    initialSourcing: string;
    followupSourcing: string;
    asymmetry: string;
  }

  const scoreboardEntries = derived(scoreboardStore, ($store) =>
    Object.entries($store).map<ScoreboardEntry>(([model, summary]) =>
      formatSummary(model, summary),
    ),
  );
  const status = derived(statusStore, ($store) => $store.state);

  const formatPercent = (value: number | undefined): string => {
    if (typeof value !== "number" || Number.isNaN(value)) {
      return "0%";
    }
    return `${Math.round(value * 100)}%`;
  };

  const formatAverage = (value: number | undefined): string => {
    if (typeof value !== "number" || Number.isNaN(value)) {
      return "0.00";
    }
    return value.toFixed(2);
  };

  const formatCounts = (counts: Record<string, number> | undefined): string => {
    if (!counts) {
      return "n/a";
    }
    const entries = Object.entries(counts)
      .filter(([, value]) => value)
      .sort((a, b) => b[1] - a[1]);
    if (!entries.length) {
      return "n/a";
    }
    return entries
      .slice(0, 3)
      .map(([label, count]) => `${label}×${count}`)
      .join(", ");
  };

  const formatSummary = (
    model: string,
    summary: ModelSummary,
  ): ScoreboardEntry => ({
    model,
    totalLabel: `${summary.total} prompts`,
    successLabel: `${summary.ok} ok / ${summary.issues} issues`,
    initialCompleteness: formatAverage(summary.avg_initial_completeness),
    followupCompleteness: formatAverage(summary.avg_followup_completeness),
    initialRefusal: formatPercent(summary.initial_refusal_rate),
    followupRefusal: formatPercent(summary.followup_refusal_rate),
    initialSourcing: formatCounts(summary.initial_sourcing_counts),
    followupSourcing: formatCounts(summary.followup_sourcing_counts),
    asymmetry: formatCounts(summary.asymmetry_counts),
  });
</script>

<aside class="scoreboard">
  <header>
    <h2>Scoreboard</h2>
    <span class="state-pill">{$status}</span>
  </header>

  {#if $scoreboardEntries.length === 0}
    <p class="empty">No judgments yet. Launch a run to populate results.</p>
  {:else}
    <div class="grid">
      {#each $scoreboardEntries as entry (entry.model)}
        <article class="card">
          <header>
            <h3>{entry.model}</h3>
            <span class="badge">{entry.totalLabel}</span>
          </header>
          <dl>
            <div>
              <dt>Success</dt>
              <dd>{entry.successLabel}</dd>
            </div>
            <div>
              <dt>Initial completeness</dt>
              <dd>{entry.initialCompleteness}</dd>
            </div>
            <div>
              <dt>Follow-up completeness</dt>
              <dd>{entry.followupCompleteness}</dd>
            </div>
            <div>
              <dt>Initial refusal</dt>
              <dd>{entry.initialRefusal}</dd>
            </div>
            <div>
              <dt>Follow-up refusal</dt>
              <dd>{entry.followupRefusal}</dd>
            </div>
            <div>
              <dt>Sourcing (initial)</dt>
              <dd>{entry.initialSourcing}</dd>
            </div>
            <div>
              <dt>Sourcing (follow)</dt>
              <dd>{entry.followupSourcing}</dd>
            </div>
            <div>
              <dt>Asymmetry</dt>
              <dd>{entry.asymmetry}</dd>
            </div>
          </dl>
        </article>
      {/each}
    </div>
  {/if}
</aside>

<style>
  .scoreboard {
    display: flex;
    flex-direction: column;
    gap: 1rem;
    background: rgba(21, 37, 43, 0.7);
    border: 1px solid var(--color-border);
    border-radius: 24px;
    padding: 1.5rem;
    min-height: 0;
  }

  header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 1rem;
  }

  h2 {
    margin: 0;
    font-size: 1.2rem;
  }

  .state-pill {
    text-transform: capitalize;
    background: rgba(45, 204, 202, 0.22);
    padding: 0.35rem 0.9rem;
    border-radius: 999px;
    font-size: 0.85rem;
    font-weight: 600;
  }

  .empty {
    color: var(--color-text-muted);
    margin: 0;
    font-size: 0.9rem;
  }

  .grid {
    display: flex;
    flex-direction: column;
    gap: 1rem;
  }

  .card {
    background: rgba(15, 20, 27, 0.65);
    border: 1px solid rgba(70, 85, 93, 0.4);
    border-radius: 20px;
    padding: 1rem;
    display: flex;
    flex-direction: column;
    gap: 0.75rem;
  }

  .card header {
    display: flex;
    justify-content: space-between;
    align-items: center;
  }

  .card h3 {
    margin: 0;
    font-size: 1rem;
  }

  .badge {
    font-size: 0.8rem;
    color: var(--color-text-muted);
    background: rgba(70, 85, 93, 0.35);
    padding: 0.3rem 0.6rem;
    border-radius: 8px;
  }

  dl {
    display: grid;
    gap: 0.6rem;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    margin: 0;
  }

  dt {
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    color: var(--color-text-muted);
  }

  dd {
    margin: 0.15rem 0 0;
    font-weight: 600;
    color: var(--color-text);
  }

  @media (max-width: 900px) {
    dl {
      grid-template-columns: 1fr;
    }
  }
</style>
