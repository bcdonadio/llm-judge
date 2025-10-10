import { render, screen, waitFor } from '@testing-library/svelte';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import App from './App.svelte';
import * as stores from '@/lib/stores';

const { artifactsStore } = stores;

describe('App', () => {
  beforeEach(() => {
    artifactsStore.set(null);
    vi.restoreAllMocks();
  });

  afterEach(() => {
    artifactsStore.set(null);
  });

  it('boots the dashboard and shows artifacts from the store', async () => {
    const disconnectMock = vi.fn();
    const initSpy = vi.spyOn(stores, 'initializeStores').mockResolvedValue();
    const connectSpy = vi.spyOn(stores, 'connectEvents').mockReturnValue(disconnectMock);

    const { unmount } = render(App);

    await waitFor(() => {
      expect(initSpy).toHaveBeenCalled();
      expect(connectSpy).toHaveBeenCalled();
    });

    await waitFor(() => {
      expect(screen.queryByText('Loading dashboard…')).not.toBeInTheDocument();
    });

    expect(screen.getByText('LLM Judge')).toBeInTheDocument();
    expect(screen.getByText('Scoreboard')).toBeInTheDocument();

    artifactsStore.set({ csv_path: 'runs.csv', runs_dir: 'runs' });

    await waitFor(() => {
      expect(screen.getByText('Artifacts')).toBeInTheDocument();
      expect(screen.getByText('runs.csv')).toBeInTheDocument();
      expect(screen.getByText('runs')).toBeInTheDocument();
    });

    unmount();
    expect(disconnectMock).toHaveBeenCalled();
  });

  it('logs initialization errors but still connects to events', async () => {
    const disconnectMock = vi.fn();
    const error = new Error('failed');
    const errorSpy = vi.spyOn(console, 'error').mockImplementation(() => {});
    vi.spyOn(stores, 'initializeStores').mockRejectedValue(error);
    const connectSpy = vi.spyOn(stores, 'connectEvents').mockReturnValue(disconnectMock);

    const { unmount } = render(App);

    await waitFor(() => {
      expect(connectSpy).toHaveBeenCalled();
    });

    expect(errorSpy).toHaveBeenCalledWith('Failed to initialise stores', error);
    await waitFor(() => {
      expect(screen.queryByText('Loading dashboard…')).not.toBeInTheDocument();
    });

    unmount();
    expect(disconnectMock).toHaveBeenCalled();
    errorSpy.mockRestore();
  });
});
