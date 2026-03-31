import { ChatRunConfig, ModelCatalogEntry } from "../../domain/types";

export const MODEL_CATALOG: ModelCatalogEntry[] = [
  {
    id: "ar_qwen3_0_6b",
    lane: "ar",
    label: "Qwen3-0.6B",
    description: "AR locale via /api/generate/ar",
  },
  {
    id: "dllm_hilda_default",
    lane: "dllm",
    label: "Hilda dLLM",
    description: "Diffusion decode via /api/generate",
  },
];

export const DEFAULT_CHAT_CONFIG: ChatRunConfig = {
  engineMode: "BOTH",
  arModelId: "ar_qwen3_0_6b",
  dllmModelId: "dllm_hilda_default",
  decodeMode: "S_MODE",
  effort: "medium",
  maxNewTokens: 256,
  seed: 42,
  temperature: 0.6,
  topP: 0.95,
  topK: 20,
  presencePenalty: 1.5,
  repetitionPenalty: 1.1,
  tauMask: 0.08,
  tauEdit: 0.08,
  systemPrompt:
    "Sei un assistente tecnico per coding su Qwen3 small. Rispondi in italiano tecnico, con output pratico, evitando ripetizioni. Fornisci codice solo quando richiesto.",
  thinkingMode: "auto",
  contextWindowTokens: 4096,
  arDecodeMode: "sampling",
};
