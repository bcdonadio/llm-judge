<script lang="ts">
  import { cancelRun, defaultsStore, isActiveState, pauseRun, resumeRun, startRun, statusStore } from '@/lib/stores';
  import { parseLimit, parseModels } from './control-panel-helpers';
  import type { RunConfig, RunState } from '@/lib/types';

  const baseConfig: RunConfig = {
    models: ['qwen/qwen3-next-80b-a3b-instruct'],
    judge_model: 'x-ai/grok-4-fast',
    limit: 1,
    max_tokens: 8000,
    judge_max_tokens: 6000,
    temperature: 0.2,
    judge_temperature: 0.0,
    sleep_s: 0.2,
    outdir: 'results',
    verbose: false,
  };

  let config: RunConfig = { ...baseConfig };
  let modelsText = baseConfig.models.join('\n');
  let limitText = baseConfig.limit != null ? String(baseConfig.limit) : '';
  let message = '';
  let messageKind: 'info' | 'error' = 'info';
  let submitting = false;
  let defaultsBootstrapped = false;
  let state: RunState = 'idle';

  $: currentStatus = $statusStore;
  $: state = currentStatus.state ?? 'idle';
  $: disableRun = submitting || isActiveState(state);
  $: canPause = state === 'running';
  $: canResume = state === 'paused';
  $: canCancel = isActiveState(state);

  $: if (!defaultsBootstrapped && $defaultsStore) {
    defaultsBootstrapped = true;
    config = { ...baseConfig, ...$defaultsStore };
    modelsText = $defaultsStore.models.join('\n');
    limitText = $defaultsStore.limit != null ? String($defaultsStore.limit) : '';
  }

  function showMessage(text: string, kind: 'info' | 'error' = 'info') {
    message = text;
    messageKind = kind;
    if (text) {
      setTimeout(() => {
        message = '';
      }, 5000);
    }
  }

  async function handleRun() {
    submitting = true;
    const models = parseModels(modelsText);
    if (!models.length) {
      showMessage('Please provide at least one model slug.', 'error');
      submitting = false;
      return;
    }

    const nextConfig: RunConfig = {
      ...config,
      models,
      limit: parseLimit(limitText),
    };
    const result = await startRun(nextConfig);
    submitting = false;
    if (!result.ok) {
      showMessage(result.message ?? 'Unable to start run', 'error');
      return;
    }
    showMessage('Evaluation run triggered.', 'info');
  }

  async function handlePause() {
    const result = await pauseRun();
    if (!result.ok) {
      showMessage(result.message ?? 'Pause failed', 'error');
      return;
    }
    showMessage('Run paused.');
  }

  async function handleResume() {
    const result = await resumeRun();
    if (!result.ok) {
      showMessage(result.message ?? 'Resume failed', 'error');
      return;
    }
    showMessage('Run resumed.');
  }

  async function handleCancel() {
    const result = await cancelRun();
    if (!result.ok) {
      showMessage(result.message ?? 'Cancel failed', 'error');
      return;
    }
    showMessage('Cancellation requested.');
  }
</script>

<section class="control-panel">
  <header class="header">
    <img src="/llm-judge.png" alt="LLM Judge logo" />
    <div>
      <h1>LLM Judge</h1>
      <p class="subtitle">Real-time evaluation dashboard</p>
    </div>
  </header>

  <form class="form" on:submit|preventDefault={handleRun}>
    <label>
      <span>Models to evaluate</span>
      <textarea
        rows="3"
        placeholder="openrouter/model-1&#10;openrouter/model-2"
        bind:value={modelsText}
      ></textarea>
    </label>

    <div class="row">
      <label>
        <span>Judge model</span>
        <input type="text" bind:value={config.judge_model} />
      </label>
      <label>
        <span>Prompt limit</span>
        <input type="number" min="0" step="1" bind:value={limitText} placeholder="Leave blank for all prompts" />
      </label>
    </div>

    <label>
      <span>Output directory</span>
      <input type="text" bind:value={config.outdir} />
    </label>

    <details class="advanced">
      <summary>Advanced parameters</summary>
      <div class="advanced-grid">
        <label>
          <span>Max tokens (response)</span>
          <input type="number" min="1" bind:value={config.max_tokens} />
        </label>
        <label>
          <span>Judge max tokens</span>
          <input type="number" min="1" bind:value={config.judge_max_tokens} />
        </label>
        <label>
          <span>Temperature</span>
          <input type="number" min="0" step="0.1" bind:value={config.temperature} />
        </label>
        <label>
          <span>Judge temperature</span>
          <input type="number" min="0" step="0.1" bind:value={config.judge_temperature} />
        </label>
        <label>
          <span>Inter-request sleep (s)</span>
          <input type="number" min="0" step="0.1" bind:value={config.sleep_s} />
        </label>
      </div>
    </details>

    <div class="actions">
      <button type="submit" disabled={disableRun}>Run</button>
      <button type="button" disabled={!canPause} on:click={handlePause}>Pause</button>
      <button type="button" disabled={!canResume} on:click={handleResume}>Resume</button>
      <button type="button" disabled={!canCancel} class="secondary" on:click={handleCancel}>Cancel</button>
    </div>
  </form>

  <footer class="status">
    <span class={`state state-${state}`}>{state}</span>
    {#if currentStatus.config?.models?.length}
      <span class="models-count">{currentStatus.config.models.length} models queued</span>
    {/if}
    {#if message}
      <span class={`message ${messageKind}`}>{message}</span>
    {/if}
  </footer>
</section>

<style>
  .control-panel {
    background: rgba(15, 20, 27, 0.65);
    border: 1px solid var(--color-border);
    border-radius: 24px;
    padding: 1.5rem;
    backdrop-filter: blur(18px);
    display: flex;
    flex-direction: column;
    gap: 1.5rem;
  }

  .header {
    display: flex;
    align-items: center;
    gap: 1rem;
  }

  .header img {
    width: 56px;
    height: 56px;
    border-radius: 16px;
    border: 2px solid rgba(45, 204, 202, 0.4);
  }

  .header h1 {
    margin: 0;
    font-size: 1.5rem;
  }

  .subtitle {
    margin: 0;
    color: var(--color-text-muted);
    font-size: 0.9rem;
  }

  .form {
    display: flex;
    flex-direction: column;
    gap: 1rem;
  }

  label {
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
    font-size: 0.95rem;
  }

  textarea {
    min-height: 96px;
    resize: vertical;
  }

  .row {
    display: grid;
    gap: 1rem;
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
  }

  .advanced {
    border: 1px dashed rgba(70, 85, 93, 0.5);
    border-radius: 16px;
    padding: 0.75rem 1rem;
  }

  .advanced summary {
    cursor: pointer;
    font-weight: 600;
    color: var(--color-text-muted);
  }

  .advanced-grid {
    margin-top: 0.75rem;
    display: grid;
    gap: 1rem;
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
  }

  .actions {
    display: flex;
    flex-wrap: wrap;
    gap: 0.75rem;
  }

  .actions button.secondary {
    background: rgba(70, 85, 93, 0.65);
    color: var(--color-text);
  }

  .status {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 1rem;
    font-size: 0.85rem;
    color: var(--color-text-muted);
  }

  .state {
    padding: 0.35rem 0.8rem;
    border-radius: 999px;
    background: rgba(45, 204, 202, 0.18);
    font-weight: 600;
    text-transform: capitalize;
  }

  .state-paused {
    background: rgba(244, 190, 79, 0.25);
  }

  .state-cancelled,
  .state-error {
    background: rgba(200, 80, 80, 0.25);
  }

  .message {
    font-weight: 500;
  }

  .message.error {
    color: #f08080;
  }
</style>
