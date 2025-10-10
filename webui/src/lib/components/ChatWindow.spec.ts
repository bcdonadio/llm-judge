import { render, screen, waitFor } from '@testing-library/svelte';
import { beforeEach, describe, expect, it } from 'vitest';
import ChatWindow from './ChatWindow.svelte';
import { messagesStore } from '@/lib/stores';
import type { MessageEntry } from '@/lib/types';

describe('ChatWindow', () => {
  beforeEach(() => {
    messagesStore.set([]);
  });

  it('shows placeholder content when there are no messages', () => {
    render(ChatWindow);
    expect(
      screen.getByText('Run the suite to populate the conversation feed.'),
    ).toBeInTheDocument();
  });

  it('renders messages with metadata and step details', async () => {
    render(ChatWindow);

    const message: MessageEntry = {
      id: 'example-id',
      role: 'assistant',
      content: 'Response body',
      timestamp: Date.now(),
      model: 'openrouter/model',
      promptIndex: 5,
      step: 'analysis',
    };

    messagesStore.set([message]);

    await waitFor(() => {
      expect(screen.getByText('openrouter/model · #5')).toBeInTheDocument();
    });

    expect(screen.getByText('Response body')).toBeInTheDocument();
    expect(screen.getByText('analysis')).toBeInTheDocument();
  });

  it('falls back to default text when content is missing', async () => {
    render(ChatWindow);

    const message: MessageEntry = {
      id: 'no-content',
      role: 'user',
      content: '',
      timestamp: Date.now(),
    };

    messagesStore.set([message]);

    await waitFor(() => {
      expect(screen.getByText('No content returned.')).toBeInTheDocument();
    });
  });
});
