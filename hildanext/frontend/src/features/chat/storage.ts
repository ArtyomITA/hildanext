import { ChatRunConfig, ChatThread, SavedPreset } from "../../domain/types";
import { DEFAULT_CHAT_CONFIG } from "./catalog";

export const CHAT_STORAGE_KEY = "hildanext.chat_studio.v2";
export const CHAT_STORAGE_KEY_V1 = "hildanext.chat_studio.v1";
export const CHAT_DEFAULTS_REVISION = 2;

export interface ChatStorageV2 {
  version: 2;
  configDefaultsRevision: number;
  threads: ChatThread[];
  selectedThreadId: string | null;
  presets: SavedPreset[];
  lastConfig: ChatRunConfig;
  advancedOpen: boolean;
}

interface ChatStorageV1Like {
  version: 1;
  threads?: ChatThread[];
  selectedThreadId?: string | null;
  presets?: SavedPreset[];
  lastConfig?: Partial<ChatRunConfig>;
  advancedOpen?: boolean;
}

export function createDefaultStorage(): ChatStorageV2 {
  return {
    version: 2,
    configDefaultsRevision: CHAT_DEFAULTS_REVISION,
    threads: [],
    selectedThreadId: null,
    presets: [],
    lastConfig: { ...DEFAULT_CHAT_CONFIG },
    advancedOpen: false,
  };
}

function isObject(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}

function hasOwn(raw: Record<string, unknown>, key: string): boolean {
  return Object.prototype.hasOwnProperty.call(raw, key);
}

function normalizeConfig(raw: unknown, upgradeLegacyDefaults = false): ChatRunConfig {
  if (!isObject(raw)) return { ...DEFAULT_CHAT_CONFIG };
  const merged = {
    ...DEFAULT_CHAT_CONFIG,
    ...(raw as Partial<ChatRunConfig>),
  };
  if (!upgradeLegacyDefaults) return merged;
  const legacy = raw as Record<string, unknown>;
  if (!hasOwn(legacy, "systemPrompt")) merged.systemPrompt = DEFAULT_CHAT_CONFIG.systemPrompt;
  if (!hasOwn(legacy, "thinkingMode")) merged.thinkingMode = DEFAULT_CHAT_CONFIG.thinkingMode;
  if (!hasOwn(legacy, "contextWindowTokens"))
    merged.contextWindowTokens = DEFAULT_CHAT_CONFIG.contextWindowTokens;
  if (!hasOwn(legacy, "arDecodeMode")) merged.arDecodeMode = DEFAULT_CHAT_CONFIG.arDecodeMode;
  if (!hasOwn(legacy, "topK")) merged.topK = DEFAULT_CHAT_CONFIG.topK;
  if (!hasOwn(legacy, "presencePenalty")) merged.presencePenalty = DEFAULT_CHAT_CONFIG.presencePenalty;
  if (!hasOwn(legacy, "repetitionPenalty")) merged.repetitionPenalty = DEFAULT_CHAT_CONFIG.repetitionPenalty;
  if (!hasOwn(legacy, "maxNewTokens") || Number(legacy.maxNewTokens) === 96) {
    merged.maxNewTokens = DEFAULT_CHAT_CONFIG.maxNewTokens;
  }
  return merged;
}

function fromV1(parsed: ChatStorageV1Like): ChatStorageV2 {
  const fallback = createDefaultStorage();
  return {
    version: 2,
    configDefaultsRevision: CHAT_DEFAULTS_REVISION,
    threads: Array.isArray(parsed.threads) ? parsed.threads : fallback.threads,
    selectedThreadId:
      typeof parsed.selectedThreadId === "string" || parsed.selectedThreadId === null
        ? parsed.selectedThreadId
        : fallback.selectedThreadId,
    presets: Array.isArray(parsed.presets)
      ? parsed.presets.map((preset) => ({ ...preset, config: normalizeConfig(preset.config, true) }))
      : fallback.presets,
    lastConfig: normalizeConfig(parsed.lastConfig, true),
    advancedOpen: typeof parsed.advancedOpen === "boolean" ? parsed.advancedOpen : fallback.advancedOpen,
  };
}

export function parseChatStorage(raw: string | null | undefined): ChatStorageV2 {
  if (!raw) return createDefaultStorage();
  try {
    const parsed: unknown = JSON.parse(raw);
    if (!isObject(parsed)) return createDefaultStorage();
    if (parsed.version === 2) {
      const fallback = createDefaultStorage();
      const revision =
        typeof parsed.configDefaultsRevision === "number" ? parsed.configDefaultsRevision : 1;
      const mustUpgradeLegacyDefaults = revision < CHAT_DEFAULTS_REVISION;
      return {
        version: 2,
        configDefaultsRevision: CHAT_DEFAULTS_REVISION,
        threads: Array.isArray(parsed.threads) ? (parsed.threads as ChatThread[]) : fallback.threads,
        selectedThreadId:
          typeof parsed.selectedThreadId === "string" || parsed.selectedThreadId === null
            ? parsed.selectedThreadId
            : fallback.selectedThreadId,
        presets: Array.isArray(parsed.presets)
          ? (parsed.presets as SavedPreset[]).map((preset) => ({
              ...preset,
              config: normalizeConfig(preset.config, mustUpgradeLegacyDefaults),
            }))
          : fallback.presets,
        lastConfig: normalizeConfig(parsed.lastConfig, mustUpgradeLegacyDefaults),
        advancedOpen: typeof parsed.advancedOpen === "boolean" ? parsed.advancedOpen : fallback.advancedOpen,
      };
    }
    if (parsed.version === 1) {
      return fromV1(parsed as unknown as ChatStorageV1Like);
    }
    return createDefaultStorage();
  } catch {
    return createDefaultStorage();
  }
}

export function serializeChatStorage(data: ChatStorageV2): string {
  return JSON.stringify(data);
}
