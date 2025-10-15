<script lang="ts">
  import { onDestroy, tick } from "svelte";
  import { messagesStore } from "@/lib/stores";
  import type { MessageEntry } from "@/lib/types";

  let messages: MessageEntry[] = [];
  let container: HTMLDivElement | null = null;

  const unsubscribe = messagesStore.subscribe(async (value) => {
    messages = value;
    await tick();
    if (container) {
      container.scrollTop = container.scrollHeight;
    }
  });

  onDestroy(() => {
    unsubscribe();
  });

  const formatMeta = (message: MessageEntry): string | null => {
    if (!message.model && message.promptIndex == null) {
      return null;
    }
    const parts: string[] = [];
    if (message.model) {
      parts.push(message.model);
    }
    if (Number.isFinite(message.promptIndex)) {
      parts.push(`#${message.promptIndex}`);
    }
    return parts.join(" · ");
  };
</script>

<section class="chat-window">
  <header>
    <h2>Conversation</h2>
    <p>Live feed of prompts, responses, and judge notes.</p>
  </header>
  <div class="messages" bind:this={container}>
    {#if messages.length === 0}
      <div class="placeholder">
        <p>Run the suite to populate the conversation feed.</p>
      </div>
    {:else}
      {#each messages as message (message.id)}
        <article class={`message ${message.role}`}>
          {#if formatMeta(message)}
            <span class="meta">{formatMeta(message)}</span>
          {/if}
          <div class="bubble">
            <p>{message.content || "No content returned."}</p>
            {#if message.step}
              <span class="step">{message.step}</span>
            {/if}
          </div>
        </article>
      {/each}
    {/if}
  </div>
</section>

<style>
  .chat-window {
    display: flex;
    flex-direction: column;
    gap: 1rem;
    background: rgba(21, 37, 43, 0.75);
    border: 1px solid var(--color-border);
    border-radius: 24px;
    padding: 1.5rem;
    height: 100%;
    min-height: 0;
    box-sizing: border-box;
  }

  header h2 {
    margin: 0;
    font-size: 1.2rem;
  }

  header p {
    margin: 0.2rem 0 0;
    color: var(--color-text-muted);
    font-size: 0.9rem;
  }

  .messages {
    flex: 1;
    overflow-y: auto;
    display: flex;
    flex-direction: column;
    gap: 1rem;
    padding-right: 0.5rem;
  }

  .placeholder {
    margin: auto;
    text-align: center;
    color: var(--color-text-muted);
    font-size: 0.95rem;
  }

  .message {
    display: flex;
    flex-direction: column;
    max-width: 70%;
  }

  .message.user {
    align-self: flex-start;
  }

  .message.assistant {
    align-self: flex-end;
  }

  .message.judge,
  .message.system {
    align-self: center;
    max-width: 80%;
  }

  .meta {
    font-size: 0.75rem;
    color: var(--color-text-muted);
    margin-bottom: 0.3rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
  }

  .bubble {
    position: relative;
    padding: 0.85rem 1rem;
    border-radius: 16px;
    background: rgba(15, 20, 27, 0.8);
    border: 1px solid rgba(70, 85, 93, 0.45);
  }

  .message.user .bubble {
    background: rgba(70, 85, 93, 0.35);
    border-color: rgba(70, 85, 93, 0.5);
  }

  .message.assistant .bubble {
    background: rgba(45, 204, 202, 0.2);
    border-color: rgba(45, 204, 202, 0.45);
  }

  .message.judge .bubble {
    background: rgba(244, 190, 79, 0.18);
    border-color: rgba(244, 190, 79, 0.4);
  }

  .bubble p {
    margin: 0;
    white-space: pre-wrap;
    line-height: 1.5;
  }

  .step {
    margin-top: 0.4rem;
    display: inline-block;
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--color-text-muted);
  }
</style>
