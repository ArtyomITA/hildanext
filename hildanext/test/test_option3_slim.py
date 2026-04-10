"""Test Option 3: slim lm_head (only on x_t positions) at S=1024 composite=2048.
Verifies VRAM savings vs the old full-model call."""
import sys, time, gc, torch
sys.path.insert(0, 'backend/src')

torch.cuda.set_per_process_memory_fraction(0.85)
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_math_sdp(True)

from transformers import AutoModelForCausalLM
from hildanext.masks import batch_doc_attention_mask
from hildanext.diffusion import _attn_for_model, _forward
from types import SimpleNamespace

model = AutoModelForCausalLM.from_pretrained(
    '../Qwen3-0.6B', dtype=torch.float16, attn_implementation='sdpa'
).cuda()
model.gradient_checkpointing_enable(
    gradient_checkpointing_kwargs={'use_reentrant': False, 'preserve_rng_state': False}
)
print(f'model loaded: {torch.cuda.memory_allocated()/1024**2:.0f} MB', flush=True)
print(f'  model.model:   {type(model.model).__name__}', flush=True)
print(f'  model.lm_head: {type(model.lm_head).__name__} '
      f'{list(model.lm_head.weight.shape)}', flush=True)

S = 1024
vocab = 151936
input_ids = torch.randint(0, vocab, (1, S), device='cuda')
clean_ids = input_ids.clone()
doc_ids = torch.ones(1, S, dtype=torch.long, device='cuda')
attn_1d = torch.ones(1, S, dtype=torch.long, device='cuda')

# Corrupt 30%
mask_pos = torch.rand(1, S, device='cuda') < 0.3
mixed = input_ids.clone()
mixed[mask_pos] = vocab - 1

print(f'\n=== Forward no_grad (Option 3: slim lm_head) ===', flush=True)
gc.collect(); torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()
torch.cuda.synchronize()
t0 = time.time()
with torch.no_grad():
    out = _forward(model, mixed, attn_1d, doc_ids,
                   mask_mode='composite_llada20', clean_ids=clean_ids,
                   composite_block_size=None, bidirectional=False)
    logits = out.logits
torch.cuda.synchronize()
dt = time.time() - t0
peak = torch.cuda.max_memory_allocated() / 1024**2
print(f'  logits: {list(logits.shape)}  time={dt*1000:.0f}ms  peak={peak:.0f}MB', flush=True)
del logits, out

print(f'\n=== Forward + Backward (Option 3: slim lm_head) ===', flush=True)
gc.collect(); torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()
model.zero_grad()
torch.cuda.synchronize()
t0 = time.time()

out = _forward(model, mixed, attn_1d, doc_ids,
               mask_mode='composite_llada20', clean_ids=clean_ids,
               composite_block_size=None, bidirectional=False)
logits = out.logits
targets = clean_ids
loss = torch.nn.functional.cross_entropy(
    logits.view(-1, logits.size(-1)), targets.view(-1)
)
loss.backward()
torch.cuda.synchronize()
dt = time.time() - t0
peak_bwd = torch.cuda.max_memory_allocated() / 1024**2
print(f'  loss={loss.item():.4f}  time={dt*1000:.0f}ms  peak={peak_bwd:.0f}MB', flush=True)
del logits, out, loss

print(f'\n=== MTF 2-turn (Option 3: slim lm_head) ===', flush=True)
gc.collect(); torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()
model.zero_grad()
torch.cuda.synchronize()
t0 = time.time()
current = mixed.clone()
for turn in range(2):
    out = _forward(model, current, attn_1d, doc_ids,
                   mask_mode='composite_llada20', clean_ids=clean_ids,
                   composite_block_size=None, bidirectional=False)
    logits = out.logits
    loss = torch.nn.functional.cross_entropy(
        logits.view(-1, logits.size(-1)), clean_ids.view(-1)
    )
    (loss / 2.0 / 8.0).backward()
    if turn == 0:
        with torch.no_grad():
            preds = logits.argmax(-1)
            current = clean_ids.clone()
            current[:, 1:] = preds[:, :-1]
    del logits, out, loss
    torch.cuda.synchronize()
    print(f'  turn {turn}: alloc={torch.cuda.memory_allocated()/1024**2:.0f} MB', flush=True)

torch.cuda.synchronize()
dt = time.time() - t0
peak_mtf = torch.cuda.max_memory_allocated() / 1024**2
print(f'  MTF total: time={dt*1000:.0f}ms  peak={peak_mtf:.0f}MB', flush=True)

print(f'\n{"="*60}')
print(f'SUMMARY')
print(f'{"="*60}')
print(f'  Forward no_grad peak:  {peak:.0f} MB')
print(f'  Fwd+Bwd peak:         {peak_bwd:.0f} MB')
print(f'  MTF 2-turn peak:      {peak_mtf:.0f} MB')
limit_85 = 8192 * 0.85
limit_70 = 8192 * 0.70
ok = "OK" if peak_mtf < limit_85 else "OVER"
ok70 = "OK" if peak_mtf < limit_70 else "OVER"
print(f'  vs 85% ({limit_85:.0f} MB):     {ok}')
print(f'  vs 70% ({limit_70:.0f} MB):     {ok70}')
if peak_mtf < limit_85:
    print(f'  Headroom:              {limit_85 - peak_mtf:.0f} MB (at 85%)')
print()

del model
gc.collect(); torch.cuda.empty_cache()
print('cleanup done', flush=True)
