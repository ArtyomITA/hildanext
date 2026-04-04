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
    label: "Hilda dLLM (base)",
    description: "Qwen3-0.6B base — models/qwen3-0.6b",
  },
  {
    id: "dllm_wsd_step_04000",
    lane: "dllm",
    label: "WSD step 4000",
    description: "WSD checkpoint — runs/checkpoints/cpt/step_04000",
  },
  {
    id: "dllm_wsd_step_03800",
    lane: "dllm",
    label: "WSD step 3800",
    description: "WSD checkpoint — runs/checkpoints/cpt/step_03800",
  },
  {
    id: "dllm_wsd_step_03600",
    lane: "dllm",
    label: "WSD step 3600",
    description: "WSD checkpoint — runs/checkpoints/cpt/step_03600",
  },
];

/** Map catalog IDs to relative model paths (from hildanext root). */
export const DLLM_MODEL_PATHS: Record<string, string> = {
  dllm_hilda_default: "models/qwen3-0.6b",
  dllm_wsd_step_04000: "runs/checkpoints/cpt/step_04000",
  dllm_wsd_step_03800: "runs/checkpoints/cpt/step_03800",
  dllm_wsd_step_03600: "runs/checkpoints/cpt/step_03600",
};

export const DEFAULT_CHAT_CONFIG: ChatRunConfig = {
  engineMode: "BOTH",
  arModelId: "ar_qwen3_0_6b",
  dllmModelId: "dllm_wsd_step_04000",
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
