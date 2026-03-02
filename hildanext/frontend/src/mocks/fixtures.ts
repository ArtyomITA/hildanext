/** Single live WSD run served by the FastAPI backend. */
export const scenarioManifest = {
  wsd: ["live_wsd_run"] as const,
  /** Inference scenarios — backend endpoint not yet wired; PromptLab local generation always works. */
  inference: ["ar_vs_diffusion_compare"] as const,
} as const;
