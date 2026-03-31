import { describe, expect, it } from "vitest";
import {
  createDefaultStorage,
  parseChatStorage,
  serializeChatStorage,
} from "../features/chat/storage";

describe("chat storage", () => {
  it("falls back to defaults on empty payload", () => {
    const storage = parseChatStorage(null);
    expect(storage.version).toBe(2);
    expect(storage.threads).toHaveLength(0);
    expect(storage.lastConfig.engineMode).toBe("BOTH");
    expect(storage.lastConfig.maxNewTokens).toBe(256);
  });

  it("serializes and parses roundtrip", () => {
    const base = createDefaultStorage();
    base.selectedThreadId = "t1";
    base.advancedOpen = true;
    const raw = serializeChatStorage(base);
    const parsed = parseChatStorage(raw);
    expect(parsed.selectedThreadId).toBe("t1");
    expect(parsed.advancedOpen).toBe(true);
  });

  it("returns defaults for malformed json", () => {
    const parsed = parseChatStorage("{not-json");
    expect(parsed).toEqual(createDefaultStorage());
  });

  it("migrates v1 payload into v2 schema", () => {
    const legacy = JSON.stringify({
      version: 1,
      threads: [],
      selectedThreadId: null,
      presets: [],
      lastConfig: {
        engineMode: "AR",
        maxNewTokens: 96,
      },
      advancedOpen: true,
    });
    const parsed = parseChatStorage(legacy);
    expect(parsed.version).toBe(2);
    expect(parsed.advancedOpen).toBe(true);
    expect(parsed.lastConfig.engineMode).toBe("AR");
    expect(parsed.lastConfig.systemPrompt.length).toBeGreaterThan(0);
    expect(parsed.lastConfig.maxNewTokens).toBe(256);
  });

  it("upgrades legacy v2 defaults when revision marker is missing", () => {
    const legacyV2 = JSON.stringify({
      version: 2,
      threads: [],
      selectedThreadId: null,
      presets: [],
      lastConfig: {
        engineMode: "AR",
        maxNewTokens: 96,
      },
      advancedOpen: false,
    });
    const parsed = parseChatStorage(legacyV2);
    expect(parsed.version).toBe(2);
    expect(parsed.configDefaultsRevision).toBe(2);
    expect(parsed.lastConfig.maxNewTokens).toBe(256);
    expect(parsed.lastConfig.contextWindowTokens).toBe(4096);
    expect(parsed.lastConfig.arDecodeMode).toBe("sampling");
  });
});
