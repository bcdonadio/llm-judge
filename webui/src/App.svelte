<script lang="ts">
  import ControlPanel from '@/lib/components/ControlPanel.svelte';
  import Scoreboard from '@/lib/components/Scoreboard.svelte';
  import ChatWindow from '@/lib/components/ChatWindow.svelte';
  import { artifactsStore, connectEvents, initializeStores } from '@/lib/stores';

  let disconnect = $state<(() => void) | null>(null);
  let loaded = $state(false);

  $effect(() => {
    let cancelled = false;
    (async () => {
      try {
        await initializeStores();
      } catch (error) {
        console.error('Failed to initialise stores', error);
      } finally {
        if (!cancelled) {
          disconnect = connectEvents();
          loaded = true;
        }
      }
    })();

    return () => {
      cancelled = true;
      disconnect?.();
    };
  });

  const artifacts = $derived($artifactsStore);
</script>

<div class="app-grid">
  <aside class="sidebar">
    <Scoreboard />
  </aside>
  <main class="main">
    <ControlPanel />
    <ChatWindow />
    {#if artifacts?.csv_path}
      <section class="artifacts">
        <h3>Artifacts</h3>
        <p>
          <span>CSV:</span>
          <code>{artifacts.csv_path}</code>
        </p>
        <p>
          <span>Raw JSON:</span>
          <code>{artifacts.runs_dir}</code>
        </p>
      </section>
    {/if}
    {#if !loaded}
      <div class="loading-overlay">
        <div class="spinner"></div>
        <p>Loading dashboard…</p>
      </div>
    {/if}
  </main>
</div>

<style>
  .app-grid {
    display: grid;
    grid-template-columns: minmax(260px, 320px) 1fr;
    gap: 1.5rem;
    padding: 2rem;
    min-height: 100vh;
    box-sizing: border-box;
  }

  .sidebar {
    display: flex;
    flex-direction: column;
    gap: 1.5rem;
    min-height: 0;
  }

  .main {
    display: grid;
    grid-template-rows: auto 1fr auto;
    gap: 1.5rem;
    position: relative;
  }

  .artifacts {
    background: rgba(15, 20, 27, 0.7);
    border: 1px solid var(--color-border);
    border-radius: 20px;
    padding: 1rem 1.25rem;
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
  }

  .artifacts h3 {
    margin: 0 0 0.3rem;
    font-size: 1rem;
  }

  .artifacts p {
    margin: 0;
    display: flex;
    gap: 0.5rem;
    align-items: baseline;
    font-size: 0.85rem;
    color: var(--color-text-muted);
  }

  .artifacts span {
    font-weight: 600;
    color: var(--color-text);
  }

  code {
    background: rgba(21, 37, 43, 0.85);
    padding: 0.25rem 0.5rem;
    border-radius: 6px;
    font-family: 'Fira Code', 'SFMono-Regular', monospace;
    font-size: 0.8rem;
    color: var(--color-accent);
    overflow-wrap: anywhere;
  }

  .loading-overlay {
    position: absolute;
    inset: 0;
    background: rgba(15, 20, 27, 0.85);
    border-radius: 24px;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 0.75rem;
    pointer-events: none;
  }

  .spinner {
    width: 32px;
    height: 32px;
    border-radius: 50%;
    border: 3px solid rgba(45, 204, 202, 0.2);
    border-top-color: var(--color-accent);
    animation: spin 1s linear infinite;
  }

  .loading-overlay p {
    margin: 0;
    color: var(--color-text-muted);
    font-size: 0.9rem;
  }

  @keyframes spin {
    from {
      transform: rotate(0deg);
    }
    to {
      transform: rotate(360deg);
    }
  }

  @media (max-width: 1080px) {
    .app-grid {
      grid-template-columns: 1fr;
    }

    .sidebar {
      order: 2;
    }

    .main {
      order: 1;
    }
  }
</style>
