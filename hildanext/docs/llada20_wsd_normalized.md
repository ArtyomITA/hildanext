# LLaDA 2.0 -- WSD Training Reference (Normalized from Paper)

## 1. WSD Training Schedule

Warmup-Stable-Decay converts a pre-trained AR model into a diffusion LLM via 3 phases:

| Phase   | Purpose                                    | Attention         | Block behavior              |
|---------|--------------------------------------------|--------------------|------------------------------|
| Warmup  | Ramp from AR (block=1) to full-seq MDLM    | Causal (hybrid)    | Ladder: 1 → 4 → 32 → 64 → 4096 |
| Stable  | Full-sequence MDLM training                | **Bidirectional**  | Fixed at seq_len (one block) |
| Decay   | Shrink blocks for inference efficiency     | Causal (hybrid)    | 4096 → 2048 → ... → 32      |

Base models: Ling-mini-2.0 (16B), Ling-flash-2.0 (100B).
Starting point: AR model = degenerate BDLM with block_size=1.

## 2. Block Size Logic

### Warmup ladder
`1 → 4 → 32 → 64 → 4096` -- progressively expands joint denoising receptive field.

### Stable phase
Block = seq_len (e.g. 4096). Entire input is a single block = classical MDLM (K=1).

### Decay phase
Step-by-step reduction: `4096 → 2048 → ... → 32`.
Gradual reduction preserves semantic understanding while regaining BDLM's KV-cache reuse.

### Inference
- Native context: 32k tokens
- Optimal inference block: 32
- Denoising threshold: 0.95

## 3. Attention Masking

### BDLM phases (Warmup + Decay)
**Block-wise document-level attention mask** M applied. Sequence doubled: `x_full = [x_t || x_0]` (noisy || clean prefix).

Rules:
- `xt_i → xt_j`: bidirectional WITHIN same block (`b(i)==b(j)`)
- `xt_i → x0_j`: causal -- masked block attends only to PRECEDING clean blocks (`b(i) > b(j-L)`)
- `x0_i → x0_j`: causal (standard AR-style, `b(i-L) >= b(j-L)`)
- `x0 → xt`: always 0 (no attention from clean to masked)

### MDLM phase (Stable)
Flat document-level mask: `M_ij = 1 iff same document`. Pure bidirectional within each document.

### Summary
AR-style causal → hybrid causal+bidir (warmup) → fully bidirectional (stable) → hybrid (decay).

## 4. Loss Functions

### BDLM Loss (Warmup + Decay)
```
L_BDLM = -E[alpha'_t/(1-alpha_t) * sum_k sum_i 1[x_{t,k}^i=MASK] * log p(x_{0,k}^i | x_{0,<k}, x_{t,k})]
```
- K = blocks per sequence, L_B = block size
- Conditions on all preceding clean blocks x_{0,<k}

### MDLM Loss (Stable, K=1)
```
L_MDLM = -E[alpha'_t/(1-alpha_t) * sum_i 1[x_t^i=MASK] * log p(x_0^i | x_t)]
```
No causal conditioning. Full (partially masked) sequence visible.

### ELBO Weighting
`weight(t) = alpha'_t / (1 - alpha_t) = 1/t` for linear noise schedule.
Earlier timesteps (more masking) → lower weight. Later (less masking) → higher weight.

## 5. LR Schedule (Per Paper)

| Phase   | LR behavior                           |
|---------|----------------------------------------|
| Warmup  | Linear ramp: 0 → lr_base               |
| Stable  | **Constant**: lr_base                   |
| Decay   | Cosine decay: lr_base → lr_base × min_ratio |

**Critical**: LR is CONSTANT during stable phase. Not cosine decay.

## 6. Document-Level Attention Mask

Applied throughout ALL training phases.
- Sequences formed by packing heterogeneous documents into fixed-length segments
- Without mask, attention incorrectly crosses document boundaries
- BDLM variant: block-diagonal with causal cross-block structure
- MDLM variant: purely block-diagonal per document

## 7. Embedding Noise for Gradient Stability

Problem: During AR pretraining, `[MASK]` token is never observed. Its embedding decays
toward zero. When AR model is loaded for CPT, this causes gradient explosion at high mask ratios.

Fix: During initial iterations of CPT, add independent Gaussian noise to embedding output
for masked tokens:
```
e_mask_tilde = e_mask + epsilon,  epsilon ~ N(0, sigma^2 I)
```
Ensures mask token's L2 norm remains significant. Applied during warmup, linearly decayed.

## 8. Optimizer & Hyperparameters

### Production scale (paper)
- Optimizer: AdamW, beta1=0.9, beta2=0.95, eps=1e-8
- Weight decay: 0.1
- LR (100B): ~1e-4 peak
- Batch: ~4M tokens (multi-node)
- Backend: Megatron-LM
- DPO LR: initialized from SFT final LR

## 9. CPT vs Pretraining

| Aspect        | CPT (AR→dLLM)        | SFT                    |
|---------------|----------------------|------------------------|
| Structure     | WSD 3-phase          | Single phase BDLM      |
| Loss          | BDLM/MDLM uncondit.  | BDLM conditioned on c  |
| Masking       | Full alpha_t ~ U[0,1] | Restricted bandwidth  |
| Data util     | ~50% tokens/sample   | ~100% via complementary |
| Embed noise   | Yes, initial iters   | N/A                    |
| Doc mask      | Applied throughout   | Applied throughout     |

## 10. Dataset Preparation

- Packing: heterogeneous documents packed into fixed-length segments
- Cross-doc prevention: document-level attention mask
- SFT padding: quantized to nearest multiple of block size
- Tokenizer: inherited from AR base model (not changed during conversion)
- Inference: 32k native; 64k via YaRN RoPE scaling

## Key Equations

| Symbol            | Meaning                                    |
|-------------------|--------------------------------------------|
| L_B               | Block size                                 |
| K                 | Number of blocks per sequence              |
| alpha_t           | Keep-probability at time t (1=clean, 0=masked) |
| alpha'_t/(1-alpha_t) | ELBO weight = 1/t for linear schedule    |
| b(i)              | Document-block index of position i         |
| x_t               | Noisy (masked) view                        |
| x_0               | Clean sequence                             |
| x_{0,<k}          | Clean tokens of all blocks preceding k     |
