import { render, screen, waitFor } from "@testing-library/svelte";
import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import App from "./App.svelte";
import * as stores from "@/lib/stores";
import type { ConnectionState } from "@/lib/stores";

const { artifactsStore, connectionStore } = stores;

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
const appSourcePath = resolve(__dirname, "./App.svelte");
const chatWindowSourcePath = resolve(
  __dirname,
  "./lib/components/ChatWindow.svelte",
);

describe("App", () => {
  beforeEach(() => {
    artifactsStore.set(null);
    vi.restoreAllMocks();
  });

  afterEach(() => {
    artifactsStore.set(null);
  });

  it("boots the dashboard and shows artifacts from the store", async () => {
    const disconnectMock = vi.fn();
    const initSpy = vi.spyOn(stores, "initializeStores").mockResolvedValue();
    const connectSpy = vi
      .spyOn(stores, "connectEvents")
      .mockReturnValue(disconnectMock);

    const { unmount } = render(App);

    await waitFor(() => {
      expect(initSpy).toHaveBeenCalled();
      expect(connectSpy).toHaveBeenCalled();
    });

    await waitFor(() => {
      expect(screen.queryByText("Loading dashboard…")).not.toBeInTheDocument();
    });

    expect(screen.getByText("LLM Judge")).toBeInTheDocument();
    expect(screen.getByText("Scoreboard")).toBeInTheDocument();

    connectionStore.set("connected");

    await waitFor(() => {
      expect(screen.getByText("Connection: connected")).toBeInTheDocument();
    });

    connectionStore.set(undefined as unknown as ConnectionState);

    await waitFor(() => {
      expect(screen.getByText(/^Connection:\s*$/)).toBeInTheDocument();
    });

    connectionStore.set("connected");

    await waitFor(() => {
      expect(screen.getByText("Connection: connected")).toBeInTheDocument();
    });

    artifactsStore.set({ csv_path: "runs.csv", runs_dir: "runs" });

    await waitFor(() => {
      expect(screen.getByText("Artifacts")).toBeInTheDocument();
      expect(screen.getByText("runs.csv")).toBeInTheDocument();
      expect(screen.getByText("runs")).toBeInTheDocument();
    });

    unmount();
    expect(disconnectMock).toHaveBeenCalled();
  });

  it("logs initialization errors but still connects to events", async () => {
    const disconnectMock = vi.fn();
    const error = new Error("failed");
    const errorSpy = vi.spyOn(console, "error").mockImplementation(() => {});
    vi.spyOn(stores, "initializeStores").mockRejectedValue(error);
    const connectSpy = vi
      .spyOn(stores, "connectEvents")
      .mockReturnValue(disconnectMock);

    const { unmount } = render(App);

    await waitFor(() => {
      expect(connectSpy).toHaveBeenCalled();
    });

    expect(errorSpy).toHaveBeenCalledWith("Failed to initialise stores", error);
    await waitFor(() => {
      expect(screen.queryByText("Loading dashboard…")).not.toBeInTheDocument();
    });

    unmount();
    expect(disconnectMock).toHaveBeenCalled();
    errorSpy.mockRestore();
  });

  it("shows a loading indicator until initialization completes", async () => {
    const disconnectMock = vi.fn();
    let resolveInit!: () => void;
    const initPromise = new Promise<void>((resolve) => {
      resolveInit = resolve;
    });
    vi.spyOn(stores, "initializeStores").mockReturnValue(initPromise);
    const connectSpy = vi
      .spyOn(stores, "connectEvents")
      .mockReturnValue(disconnectMock);

    const { unmount } = render(App);

    expect(screen.getByText("Loading dashboard…")).toBeInTheDocument();

    resolveInit();

    await waitFor(() => {
      expect(connectSpy).toHaveBeenCalled();
    });
    await waitFor(() => {
      expect(screen.queryByText("Loading dashboard…")).not.toBeInTheDocument();
    });

    unmount();
    expect(disconnectMock).toHaveBeenCalled();
  });

  it("skips disconnect cleanup when initialization has not finished", async () => {
    let resolveInit!: () => void;
    const initPromise = new Promise<void>((resolve) => {
      resolveInit = resolve;
    });
    const connectSpy = vi
      .spyOn(stores, "connectEvents")
      .mockReturnValue(vi.fn());
    vi.spyOn(stores, "initializeStores").mockReturnValue(initPromise);

    const { unmount } = render(App);
    unmount();

    expect(connectSpy).not.toHaveBeenCalled();

    resolveInit();
    await waitFor(() => {
      expect(connectSpy).not.toHaveBeenCalled();
    });
  });

  it("keeps the conversation panel scrollable without clipping", async () => {
    const disconnectMock = vi.fn();
    vi.spyOn(stores, "initializeStores").mockResolvedValue();
    vi.spyOn(stores, "connectEvents").mockReturnValue(disconnectMock);

    const { unmount } = render(App);

    await waitFor(() => {
      expect(screen.queryByText("Loading dashboard…")).not.toBeInTheDocument();
    });

    const appSource = readFileSync(appSourcePath, "utf-8");
    const chatSource = readFileSync(chatWindowSourcePath, "utf-8");

    const chatWrapperBlock = appSource.match(/\.chat-wrapper\s*\{[^}]*\}/);
    const chatWrapperBlockText = chatWrapperBlock?.[0] ?? "";
    expect(chatWrapperBlockText).not.toEqual("");
    expect(chatWrapperBlockText).toMatch(/min-height:\s*0;/);
    expect(chatWrapperBlockText).toMatch(/display:\s*flex;/);
    expect(chatWrapperBlockText).not.toMatch(/overflow\s*:\s*hidden/);

    const chatWrapperChildBlock = appSource.match(
      /\.chat-wrapper\s*:global\(\.chat-window\)\s*\{[^}]*\}/,
    );
    const chatWrapperChildText = chatWrapperChildBlock?.[0] ?? "";
    expect(chatWrapperChildText).not.toEqual("");
    expect(chatWrapperChildText).toMatch(/flex:\s*1;/);
    expect(chatWrapperChildText).toMatch(/min-height:\s*0;/);

    const chatWindowBlock = chatSource.match(/\.chat-window\s*\{[^}]*\}/);
    const chatWindowBlockText = chatWindowBlock?.[0] ?? "";
    expect(chatWindowBlockText).not.toEqual("");
    expect(chatWindowBlockText).toMatch(/box-sizing:\s*border-box;/);

    const messagesBlock = chatSource.match(/\.messages\s*\{[^}]*\}/);
    const messagesBlockText = messagesBlock?.[0] ?? "";
    expect(messagesBlockText).not.toEqual("");
    expect(messagesBlockText).toMatch(/overflow-y:\s*auto;/);

    unmount();
    expect(disconnectMock).toHaveBeenCalled();
  });
});
