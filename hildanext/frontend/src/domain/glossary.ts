import { GlossaryTerm } from "./types";

export const glossaryTerms: GlossaryTerm[] = [
  {
    key: "gamma",
    shortLabel: "Gamma",
    english: "Gamma = committed unmask operations.",
    italianHint: "Quanti token passano da MASK a token vero nello step.",
    explanation:
      "High gamma means the diffusion loop is still drafting aggressively. If gamma stalls at zero while mask ratio stays high, decoding is degenerating.",
    relatedKeys: ["mask_ratio", "steps_to_converge", "q_mode"],
  },
  {
    key: "delta",
    shortLabel: "Delta",
    english: "Delta = token-to-token edits on already filled positions.",
    italianHint: "Correzioni vere su token gia' emessi.",
    explanation:
      "Delta shows how much the model is revising itself instead of just filling blanks. Useful for reading quality-vs-speed behavior.",
    relatedKeys: ["gamma", "s_mode", "remask"],
  },
  {
    key: "mask_ratio",
    shortLabel: "Mask ratio",
    english: "Share of generated positions still masked at the current step.",
    italianHint: "Quanto output e' ancora incompleto in quel momento.",
    explanation:
      "A falling mask ratio means convergence. Plateaus indicate slow refinement or bad thresholds.",
    relatedKeys: ["steps_to_converge", "gamma", "remask"],
  },
  {
    key: "steps_to_converge",
    shortLabel: "Converge",
    english: "How many denoising passes are needed before the mask ratio reaches zero.",
    italianHint: "Numero di step per chiudere la generazione diffusion.",
    explanation:
      "Lower is usually faster, but too low can mean the loop is being over-constrained and quality may collapse.",
    relatedKeys: ["mask_ratio", "q_mode", "s_mode"],
  },
  {
    key: "bidirectional_only_stable",
    shortLabel: "Stable-only bidir",
    english: "Bidirectional attention only during the stable WSD phase.",
    italianHint: "Warmup e decay restano causali, stable diventa bidirezionale.",
    explanation:
      "This is the repo default because it gives the model a denser denoising regime only where the schedule is most stable.",
    relatedKeys: ["block_size", "warmup", "stable", "decay"],
  },
  {
    key: "loss_weighting_inv_t",
    shortLabel: "inv_t",
    english: "Inverse-time loss weighting emphasizes harder, low-noise positions.",
    italianHint: "Pesa di piu' gli step con poco rumore e target piu' precisi.",
    explanation:
      "Useful for stabilizing the continuous-time objective when easy noisy positions dominate the batch.",
    relatedKeys: ["t_bucket", "mask_ratio"],
  },
  {
    key: "shift_mode",
    shortLabel: "preserve_left_shift",
    english: "Keeps standard AR label alignment without injecting BOS noise.",
    italianHint: "Mantiene lo shift sinistro senza sporcare il pretraining con BOS artificiali.",
    explanation:
      "This is important for Qwen3-Base because the repo explicitly avoids synthetic BOS insertion in diffusion conversion.",
    relatedKeys: ["loss_weighting_inv_t", "bidirectional_only_stable"],
  },
  {
    key: "remask",
    shortLabel: "Remask",
    english: "Controlled re-masking between passes to keep diffusion editing alive.",
    italianHint: "Rimette una parte dei token in MASK per far continuare la correzione.",
    explanation:
      "Without remask, the loop may freeze early and lose the point of editable diffusion.",
    relatedKeys: ["delta", "mask_ratio", "steps_to_converge"],
  },
  {
    key: "q_mode",
    shortLabel: "Q_MODE",
    english: "Faster, more conservative decode thresholds.",
    italianHint: "Preset rapido con meno correzioni e throughput piu' alto.",
    explanation:
      "Use when you want speed and can accept fewer correction passes.",
    relatedKeys: ["s_mode", "gamma"],
  },
  {
    key: "s_mode",
    shortLabel: "S_MODE",
    english: "Quality mode with lower thresholds and more correction passes.",
    italianHint: "Preset qualita' con piu' editing e convergenza piu' lenta.",
    explanation:
      "Use when you care about cleaner final text and richer token revision dynamics.",
    relatedKeys: ["q_mode", "delta"],
  },
];
