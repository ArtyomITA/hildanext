import { create } from "zustand";
import {
  ChatRunConfig,
  ChatThread,
  ChatTurn,
  SavedPreset,
} from "../domain/types";
import {
  CHAT_DEFAULTS_REVISION,
  CHAT_STORAGE_KEY,
  CHAT_STORAGE_KEY_V1,
  createDefaultStorage,
  parseChatStorage,
  serializeChatStorage,
} from "../features/chat/storage";
import { DEFAULT_CHAT_CONFIG } from "../features/chat/catalog";
import { createInitialThread } from "../features/chat/orchestrator";

interface ChatState {
  hydrated: boolean;
  filter: string;
  running: boolean;
  advancedOpen: boolean;
  config: ChatRunConfig;
  threads: ChatThread[];
  selectedThreadId: string | null;
  presets: SavedPreset[];
  hydrate: () => void;
  setFilter: (value: string) => void;
  setRunning: (value: boolean) => void;
  setAdvancedOpen: (value: boolean) => void;
  updateConfig: <K extends keyof ChatRunConfig>(key: K, value: ChatRunConfig[K]) => void;
  resetConfig: () => void;
  ensureThread: () => string;
  selectThread: (id: string) => void;
  createThread: () => string;
  renameThread: (id: string, title: string) => void;
  deleteThread: (id: string) => void;
  addTurn: (threadId: string, turn: ChatTurn) => void;
  updateTurn: (threadId: string, turnId: string, patch: Partial<ChatTurn>) => void;
  savePreset: (name: string) => void;
  applyPreset: (id: string) => void;
  deletePreset: (id: string) => void;
}

function persist(state: ChatState) {
  const data = {
    version: 2 as const,
    configDefaultsRevision: CHAT_DEFAULTS_REVISION,
    threads: state.threads,
    selectedThreadId: state.selectedThreadId,
    presets: state.presets,
    lastConfig: state.config,
    advancedOpen: state.advancedOpen,
  };
  localStorage.setItem(CHAT_STORAGE_KEY, serializeChatStorage(data));
}

function updateThread(
  threads: ChatThread[],
  threadId: string,
  updater: (thread: ChatThread) => ChatThread,
): ChatThread[] {
  return threads.map((thread) => {
    if (thread.id !== threadId) return thread;
    return updater(thread);
  });
}

export const useChatStore = create<ChatState>((set, get) => ({
  hydrated: false,
  filter: "",
  running: false,
  advancedOpen: false,
  config: { ...DEFAULT_CHAT_CONFIG },
  threads: [],
  selectedThreadId: null,
  presets: [],

  hydrate() {
    if (get().hydrated) return;
    const rawV2 = localStorage.getItem(CHAT_STORAGE_KEY);
    const rawV1 = localStorage.getItem(CHAT_STORAGE_KEY_V1);
    const parsed = parseChatStorage(rawV2 ?? rawV1);
    set({
      hydrated: true,
      threads: parsed.threads,
      selectedThreadId: parsed.selectedThreadId,
      presets: parsed.presets,
      config: parsed.lastConfig,
      advancedOpen: parsed.advancedOpen,
    });
    localStorage.setItem(CHAT_STORAGE_KEY, serializeChatStorage(parsed));
  },

  setFilter(value) {
    set({ filter: value });
  },

  setRunning(value) {
    set({ running: value });
  },

  setAdvancedOpen(value) {
    set({ advancedOpen: value });
    persist(get());
  },

  updateConfig(key, value) {
    set((state) => ({ config: { ...state.config, [key]: value } }));
    persist(get());
  },

  resetConfig() {
    set({ config: { ...DEFAULT_CHAT_CONFIG } });
    persist(get());
  },

  ensureThread() {
    const state = get();
    if (state.selectedThreadId) return state.selectedThreadId;
    return state.createThread();
  },

  selectThread(id) {
    set({ selectedThreadId: id });
    persist(get());
  },

  createThread() {
    const thread = createInitialThread();
    set((state) => ({
      threads: [thread, ...state.threads],
      selectedThreadId: thread.id,
    }));
    persist(get());
    return thread.id;
  },

  renameThread(id, title) {
    set((state) => ({
      threads: updateThread(state.threads, id, (thread) => ({ ...thread, title })),
    }));
    persist(get());
  },

  deleteThread(id) {
    set((state) => {
      const next = state.threads.filter((thread) => thread.id !== id);
      return {
        threads: next,
        selectedThreadId: state.selectedThreadId === id ? next[0]?.id ?? null : state.selectedThreadId,
      };
    });
    persist(get());
  },

  addTurn(threadId, turn) {
    set((state) => ({
      threads: updateThread(state.threads, threadId, (thread) => ({
        ...thread,
        title: thread.turns.length === 0 ? turn.prompt.slice(0, 48) || "Nuova chat" : thread.title,
        updatedAt: turn.createdAt,
        turns: [...thread.turns, turn],
      })),
    }));
    persist(get());
  },

  updateTurn(threadId, turnId, patch) {
    set((state) => ({
      threads: updateThread(state.threads, threadId, (thread) => ({
        ...thread,
        updatedAt: new Date().toISOString(),
        turns: thread.turns.map((turn) => (turn.id === turnId ? { ...turn, ...patch } : turn)),
      })),
    }));
    persist(get());
  },

  savePreset(name) {
    if (!name.trim()) return;
    const preset: SavedPreset = {
      id: crypto.randomUUID(),
      name: name.trim(),
      config: { ...get().config },
    };
    set((state) => ({ presets: [preset, ...state.presets] }));
    persist(get());
  },

  applyPreset(id) {
    const preset = get().presets.find((item) => item.id === id);
    if (!preset) return;
    set({ config: { ...preset.config } });
    persist(get());
  },

  deletePreset(id) {
    set((state) => ({ presets: state.presets.filter((preset) => preset.id !== id) }));
    persist(get());
  },
}));

export function clearChatStorageForTests() {
  localStorage.removeItem(CHAT_STORAGE_KEY);
  localStorage.removeItem(CHAT_STORAGE_KEY_V1);
  const reset = createDefaultStorage();
  useChatStore.setState({
    hydrated: false,
    filter: "",
    running: false,
    advancedOpen: reset.advancedOpen,
    config: reset.lastConfig,
    threads: reset.threads,
    selectedThreadId: reset.selectedThreadId,
    presets: reset.presets,
  });
}
