<script lang="ts">
  import {
    cancelRun,
    defaultsStore,
    isActiveState,
    modelCatalogStore,
    pauseRun,
    resumeRun,
    startRun,
    statusStore,
  } from "@/lib/stores";
  import {
    deriveJudgeOptionsRendered,
    displayModelName,
    formatLimit,
    modelCheckboxValue,
    modelKey,
    parseLimit,
    coerceJudgeModelId,
  } from "./control-panel-helpers";
  import type { ModelInfo, RunConfig, RunState } from "@/lib/types";

  const baseConfig: RunConfig = {
    models: ["qwen/qwen3-next-80b-a3b-instruct"],
    judge_model: "x-ai/grok-4-fast",
    limit: 1,
    max_tokens: 8000,
    judge_max_tokens: 6000,
    temperature: 0.2,
    judge_temperature: 0.0,
    sleep_s: 0.2,
    outdir: "results",
    verbose: false,
  };

  let config: RunConfig = { ...baseConfig };
  let selectedModelIds = [...baseConfig.models];
  let limitText = formatLimit(baseConfig.limit);
  let message = "";
  let messageKind: "info" | "error" = "info";
  let submitting = false;
  let defaultsBootstrapped = false;
  let state: RunState = "idle";
  let catalog: ModelInfo[] = [];
  let judgeOptions: ModelInfo[] = [];
  let selectedModels: ModelInfo[] = [];

  $: currentStatus = $statusStore;
  $: state = currentStatus.state ?? "idle";
  $: catalog = $modelCatalogStore;
  $: disableRun = submitting || isActiveState(state);
  $: canPause = state === "running";
  $: canResume = state === "paused";
  $: canCancel = isActiveState(state);

  $: if (!defaultsBootstrapped && $defaultsStore) {
    defaultsBootstrapped = true;
    config = { ...baseConfig, ...$defaultsStore };
    selectedModelIds = [...config.models];
    limitText = formatLimit($defaultsStore.limit);
  }

  function showMessage(text: string, kind: "info" | "error" = "info") {
    message = text;
    messageKind = kind;
    if (text) {
      setTimeout(() => {
        message = "";
      }, 5000);
    }
  }

  function unique(values: string[]): string[] {
    return Array.from(new Set(values.filter(Boolean)));
  }

  function arraysEqual(a: string[], b: string[]): boolean {
    if (a.length !== b.length) {
      return false;
    }
    return a.every((value, index) => value === b[index]);
  }

  $: dedupedSelected = unique(selectedModelIds);

  $: if (!arraysEqual(config.models, dedupedSelected)) {
    config = { ...config, models: dedupedSelected };
  }

  function resolveModelName(id: string): ModelInfo {
    const match = catalog.find((entry) => entry.id === id);
    if (match) {
      return match;
    }
    return { id, name: id };
  }

  function deselectModel(id: string): void {
    selectedModelIds = selectedModelIds.filter((value) => value !== id);
  }

  $: selectedModels = dedupedSelected.map(resolveModelName);

  let judgeSelectElement: HTMLSelectElement | undefined;

  $: judgeOptions = (() => {
    const seen: Record<string, boolean> = {};
    const options: ModelInfo[] = [];
    if (config.judge_model && config.judge_model.trim()) {
      if (!catalog.some((model) => model.id === config.judge_model)) {
        options.push({ id: config.judge_model, name: config.judge_model });
        seen[config.judge_model] = true;
      }
    }
    for (const model of catalog) {
      if (!seen[model.id]) {
        seen[model.id] = true;
        options.push(model);
      }
    }
    return options;
  })();

  // Workaround for Svelte 5 select binding issue: explicitly restore value after options change
  $: if (judgeSelectElement && judgeOptions) {
    // Schedule value restoration on next tick to ensure options are rendered
    setTimeout(() => {
      const currentValue = config.judge_model;
      if (
        currentValue &&
        judgeSelectElement &&
        judgeSelectElement.value !== currentValue
      ) {
        const optionExists = Array.from(judgeSelectElement.options).some(
          (opt) => opt.value === currentValue,
        );
        if (optionExists) {
          // eslint-disable-next-line svelte/infinite-reactive-loop
          judgeSelectElement.value = currentValue;
        }
      }
    }, 0);
  }

  async function handleRun() {
    submitting = true;
    const models = dedupedSelected;
    if (!models.length) {
      showMessage("Please provide at least one model slug.", "error");
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
      showMessage(result.message ?? "Unable to start run", "error");
      return;
    }
    showMessage("Evaluation run triggered.", "info");
  }

  async function handlePause() {
    const result = await pauseRun();
    if (!result.ok) {
      showMessage(result.message ?? "Pause failed", "error");
      return;
    }
    showMessage("Run paused.");
  }

  async function handleResume() {
    const result = await resumeRun();
    if (!result.ok) {
      showMessage(result.message ?? "Resume failed", "error");
      return;
    }
    showMessage("Run resumed.");
  }

  async function handleCancel() {
    const result = await cancelRun();
    if (!result.ok) {
      showMessage(result.message ?? "Cancel failed", "error");
      return;
    }
    showMessage("Cancellation requested.");
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
    <div class="field selected-field">
      <span>Selected models</span>
      <div class="selected-list" role="group" aria-label="Selected models">
        {#if selectedModels.length === 0}
          <p class="selected-empty">No models selected.</p>
        {:else}
          {#each selectedModels as model (model.id)}
            <span class="selected-pill" title={model.id}>
              <button
                type="button"
                class="pill-remove"
                on:click={() => deselectModel(model.id)}
                aria-label={`Deselect ${displayModelName(model)}`}
              >
                X
              </button>
              {displayModelName(model)}
            </span>
          {/each}
        {/if}
      </div>
    </div>

    <div class="field field-models">
      <span>Available models</span>
      <div class="models-list" role="group" aria-label="Available models">
        <p
          class="catalog-empty"
          hidden={catalog.length !== 0}
          aria-live="polite"
        >
          Model catalog unavailable. Verify OpenRouter connectivity.
        </p>
        {#each catalog as model, index (modelKey(model, index))}
          <label class="model-item">
            <input
              type="checkbox"
              value={modelCheckboxValue(model)}
              bind:group={selectedModelIds}
            />
            <span class="model-name" title={model.id}
              >{displayModelName(model)}</span
            >
          </label>
        {/each}
      </div>
    </div>

    <div class="row">
      <label>
        <span>Judge model</span>
        <select bind:value={config.judge_model} bind:this={judgeSelectElement}>
          <!-- c8 ignore start -->
          <!-- Svelte 5.43+ has a binding issue with keyed each blocks in selects, so we omit the key -->
          <!-- eslint-disable-next-line svelte/require-each-key -->
          {#each deriveJudgeOptionsRendered(judgeOptions, coerceJudgeModelId(config.judge_model)) as option}
            <option value={option.id}>{displayModelName(option)}</option>
          {/each}
          <!-- c8 ignore end -->
        </select>
      </label>
      <label>
        <span>Max rounds</span>
        <input
          type="number"
          min="0"
          step="1"
          bind:value={limitText}
          placeholder="Leave blank for all prompts"
        />
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
          <input
            type="number"
            min="0"
            step="0.1"
            bind:value={config.temperature}
          />
        </label>
        <label>
          <span>Judge temperature</span>
          <input
            type="number"
            min="0"
            step="0.1"
            bind:value={config.judge_temperature}
          />
        </label>
        <label>
          <span>Inter-request sleep (s)</span>
          <input type="number" min="0" step="0.1" bind:value={config.sleep_s} />
        </label>
      </div>
    </details>

    <div class="actions">
      <button type="submit" disabled={disableRun}>Run</button>
      <button type="button" disabled={!canPause} on:click={handlePause}
        >Pause</button
      >
      <button type="button" disabled={!canResume} on:click={handleResume}
        >Resume</button
      >
      <button
        type="button"
        disabled={!canCancel}
        class="secondary"
        on:click={handleCancel}>Cancel</button
      >
    </div>
  </form>

  <footer class="status">
    <span class={`state state-${state}`}>{state}</span>
    {#if currentStatus.config?.models?.length}
      <span class="models-count"
        >{currentStatus.config.models.length} models queued</span
      >
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

  label,
  .field {
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
    font-size: 0.95rem;
  }

  .selected-field {
    gap: 0.65rem;
  }

  .selected-list {
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
    padding: 0.5rem 0;
    min-height: 2.25rem;
  }

  .selected-pill {
    display: inline-flex;
    align-items: center;
    padding: 0.25rem 0.6rem 0.25rem 0.4rem;
    border-radius: 999px;
    background: rgba(45, 204, 202, 0.18);
    border: 1px solid rgba(45, 204, 202, 0.35);
    font-size: 0.85rem;
    gap: 0.35rem;
  }

  .pill-remove {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 1.15rem;
    height: 1.15rem;
    border-radius: 999px;
    border: 1px solid rgba(255, 255, 255, 0.8);
    background: transparent;
    color: #ffffff;
    font-size: 0.75rem;
    line-height: 1;
    cursor: pointer;
    padding: 0;
  }

  .pill-remove:hover,
  .pill-remove:focus-visible {
    background: rgba(255, 255, 255, 0.2);
    outline: none;
  }

  .model-item input[type="checkbox"] {
    margin: 0;
  }
  .selected-empty {
    margin: 0;
    font-size: 0.85rem;
    color: var(--color-text-muted);
  }

  .models-list {
    display: flex;
    flex-direction: column;
    gap: 0.4rem;
    max-height: 288px;
    overflow-y: auto;
    padding: 0.5rem 0.65rem;
    border: 1px solid rgba(70, 85, 93, 0.4);
    border-radius: 14px;
    background: rgba(15, 20, 27, 0.45);
  }

  .model-item {
    display: flex;
    flex-direction: row;
    align-items: center;
    gap: 0.6rem;
    font-size: 0.9rem;
    cursor: pointer;
  }

  .model-name {
    font-weight: 600;
    line-height: 1.2;
    word-break: break-word;
  }

  .catalog-empty {
    margin: 0;
    color: var(--color-text-muted);
    font-size: 0.9rem;
  }

  .row {
    display: grid;
    gap: 1rem;
    grid-template-columns: minmax(0, 1.5fr) minmax(0, 0.75fr);
    align-items: end;
  }

  @media (max-width: 800px) {
    .row {
      grid-template-columns: 1fr;
    }
  }

  .row select,
  .row input {
    width: 100%;
    box-sizing: border-box;
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
