# LLaDA 2.1 -- Training & Inference Reference (Normalized from Paper)

## 1. Relationship to LLaDA 2.0

LLaDA 2.1 delegates pretraining infrastructure to LLaDA 2.0 (Section 4.1):
> "For both CPT and SFT, we adopt the same training infrastructure as LLaDA2.0,
> leveraging dFactory, except that we introduce a dedicated optimized implementation
> for the multi-turn forward (MTF) stage."

LLaDA 2.1's contribution: **inference-time editing scheme** (M2T + T2T), not a new training schedule.

## 2. WSD vs eWSD

| Aspect       | WSD                   | eWSD                          |
|--------------|----------------------|-------------------------------|
| Start point  | Random init          | Stable-phase checkpoint       |
| Warmup       | Full ladder warmup   | Skipped (already warmed)      |
| Stable       | Long stable on PT data | Short stable on new domain  |
| Decay        | Standard cosine      | Aggressive (shorter, steeper) |
| Use case     | Base pretraining     | CPT / domain adaptation       |
| LR at start  | 0 → ramp up          | Resume at stable LR           |

LLaDA 2.1 CPT uses **eWSD** (inherits stable-phase checkpoint from LLaDA 2.0 base).

## 3. Loss Functions

### M2T (Mask-to-Token) -- Absorbing-state MDLM objective
```
t ~ Uniform(t_min=0.001, t_max=1.0)
mask_prob = t  (per-token i.i.d.)
x_t = x_0 with each token replaced by [MASK] with probability t
L_m2t = CrossEntropy(model(x_t), x_0) over masked positions
```

### ELBO 1/t Weighting
```
weight(t) = alpha'_t / (1 - alpha_t) = 1/t  (linear schedule)
L_m2t_weighted = L_m2t_raw / max(t, 0.001)
```
t_min=0.001 → max amplification = 1000x.

### T2T (Token-to-Token) -- LLaDA 2.1 addition
Non-masked positions corrupted with random tokens (not [MASK]):
```
Randomly replace token with uniform random token != original
L_t2t = CrossEntropy(model(x_corrupt), x_0) over T2T positions
```
M2T and T2T positions are **disjoint**. Single forward pass computes both.

### Combined Loss
```
L = m2t_weight * L_m2t_weighted + t2t_weight * L_t2t
```

## 4. Multi-Turn Forward (MTF) -- LLaDA 2.1 Innovation

Forces model to learn editing its own predictions (Draft-and-Edit scenario):

- **Turn 0**: Corrupt x_0 with M2T + T2T → get x_t, compute loss, get x_hat (predictions)
- **Turn 1**: Use x_hat as input, corrupt again, compute loss against original x_0

Labels always come from ground-truth x_0, not the model's own predictions.
`multi_turn_t2t = 2` means 2 passes per example.

## 5. Attention Strategy (CPT)

| Phase   | Attention      | Notes                         |
|---------|---------------|-------------------------------|
| Warmup  | Causal         | Small blocks, AR-like         |
| Stable  | Bidirectional  | Full-seq MDLM                 |
| Decay   | Causal         | Re-introduces block structure |

CPT for editing: `bidirectional_always` may be preferred (editing needs bidir for T2T).

## 6. Document-Level Masking

Three modes:
- `simple_blockdiag`: Block-diagonal per document. Optionally causal. Used in stable phase.
- `composite_llada20`: Double-length `[x_t | x_0]` with block-causal over x_0, bidir within blocks.
  Used in warmup/decay (BDLM phases).
- Document IDs tracked per token. Padding tokens get doc_id=-1.

## 7. Dataset

- Tokenizer inherited from base AR model
- Documents packed end-to-end with EOS separator
- Fixed-length sequences (seq_len)
- 1% random truncation for variable-length exposure
- doc_ids tensor tracks document boundaries
- Padding with pad_token, attention_mask=0 for padding

## 8. Optimizer (Production)

- AdamW, beta1=0.9, beta2=0.95
- Weight decay: 0.1
- Gradient clipping: 1.0
- LR: WSD-schedule-aware (linear warmup, constant stable, cosine decay)

## 9. Embedding Noise

During WSD warmup: Gaussian noise (sigma=0.1) added to [MASK] token embeddings.
Prevents gradient explosion from near-zero mask embeddings.
Linearly decayed to 0 over warmup phase.
