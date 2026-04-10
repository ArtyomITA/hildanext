"""Test combined Option 3 + Option 2: slim lm_head + S=512 for composite phases.
Tests at 85% VRAM cap (production). Must pass MTF 2-turn."""
import sys, time, gc, torch
sys.path.insert(0, 'backend/src')

torch.cuda.set_per_process_memory_fraction(0.85)
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_math_sdp(True)

from transformers import AutoModelForCausalLM
from hildanext.masks import batch_doc_attention_mask
from hildanext.diffusion import _attn_for_model, _forward

model = AutoModelForCausalLM.from_pretrained(
    '../Qwen3-0.6B', dtype=torch.float16, attn_implementation='sdpa'
).cuda()
model.gradient_checkpointing_enable(
    gradient_checkpointing_kwargs={'use_reentrant': False, 'preserve_rng_state': False}
)
print(f'model loaded: {torch.cuda.memory_allocated()/1024**2:.0f} MB', flush=True)

# === Test 1: Composite warmup/decay (S=512, composite=1024) ===
print(f'\n{"="*60}')
print(f'TEST 1: Composite (warmup/decay) S=512 → composite=1024')
print(f'{"="*60}')

S = 512  # halved by Option 2
vocab = 151936
input_ids = torch.randint(0, vocab, (1, S), device='cuda')
clean_ids = input_ids.clone()
doc_ids = torch.ones(1, S, dtype=torch.long, device='cuda')
attn_1d = torch.ones(1, S, dtype=torch.long, device='cuda')
mask_pos = torch.rand(1, S, device='cuda') < 0.3
mixed = input_ids.clone()
mixed[mask_pos] = vocab - 1

# MTF 2-turn
model.zero_grad()
gc.collect(); torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()
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
peak_composite = torch.cuda.max_memory_allocated() / 1024**2
print(f'  MTF 2-turn: time={dt*1000:.0f}ms  peak={peak_composite:.0f}MB')

# === Test 2: Stable phase (S=1024, simple_blockdiag, bidirectional) ===
print(f'\n{"="*60}')
print(f'TEST 2: Stable phase S=1024 simple_blockdiag bidirectional')
print(f'{"="*60}')

S2 = 1024
input_ids2 = torch.randint(0, vocab, (1, S2), device='cuda')
clean_ids2 = input_ids2.clone()
doc_ids2 = torch.ones(1, S2, dtype=torch.long, device='cuda')
attn_1d2 = torch.ones(1, S2, dtype=torch.long, device='cuda')
mixed2 = input_ids2.clone()
m2 = torch.rand(1, S2, device='cuda') < 0.3
mixed2[m2] = vocab - 1

model.zero_grad()
gc.collect(); torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()
torch.cuda.synchronize()
t0 = time.time()
current2 = mixed2.clone()
for turn in range(2):
    out = _forward(model, current2, attn_1d2, doc_ids2,
                   mask_mode='simple_blockdiag', clean_ids=None,
                   composite_block_size=None, bidirectional=True)
    logits = out.logits
    loss = torch.nn.functional.cross_entropy(
        logits.view(-1, logits.size(-1)), clean_ids2.view(-1)
    )
    (loss / 2.0 / 8.0).backward()
    if turn == 0:
        with torch.no_grad():
            preds = logits.argmax(-1)
            current2 = clean_ids2.clone()
            current2[:, 1:] = preds[:, :-1]
    del logits, out, loss
    torch.cuda.synchronize()
    print(f'  turn {turn}: alloc={torch.cuda.memory_allocated()/1024**2:.0f} MB', flush=True)

torch.cuda.synchronize()
dt = time.time() - t0
peak_stable = torch.cuda.max_memory_allocated() / 1024**2
print(f'  MTF 2-turn: time={dt*1000:.0f}ms  peak={peak_stable:.0f}MB')

# === Summary ===
print(f'\n{"="*60}')
print(f'SUMMARY (Option 3 + Option 2)')
print(f'{"="*60}')
limit_85 = 8192 * 0.85
limit_70 = 8192 * 0.70
print(f'  Composite (warmup/decay) peak:  {peak_composite:.0f} MB')
print(f'  Stable (bidirectional) peak:    {peak_stable:.0f} MB')
peak_max = max(peak_composite, peak_stable)
ok85 = "OK" if peak_max < limit_85 else "OVER"
ok70 = "OK" if peak_max < limit_70 else "OVER"
print(f'  Max peak:                       {peak_max:.0f} MB')
print(f'  vs 85% ({limit_85:.0f} MB):           {ok85}  (headroom: {limit_85-peak_max:.0f} MB)')
print(f'  vs 70% ({limit_70:.0f} MB):           {ok70}  (headroom: {limit_70-peak_max:.0f} MB)')
print()

del model
gc.collect(); torch.cuda.empty_cache()
print('cleanup done', flush=True)
