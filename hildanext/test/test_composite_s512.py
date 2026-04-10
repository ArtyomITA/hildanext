"""Test composite path at S=512 (composite=1024) with full fwd+bwd+MTF."""
import sys, time, gc, torch
sys.path.insert(0, 'backend/src')

torch.cuda.set_per_process_memory_fraction(0.70)
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_math_sdp(True)

from transformers import AutoModelForCausalLM
from hildanext.masks import batch_doc_attention_mask
from hildanext.diffusion import _attn_for_model

model = AutoModelForCausalLM.from_pretrained(
    '../Qwen3-0.6B', dtype=torch.float16, attn_implementation='sdpa'
).cuda()
model.gradient_checkpointing_enable(
    gradient_checkpointing_kwargs={'use_reentrant': False, 'preserve_rng_state': False}
)
print(f'model loaded: {torch.cuda.memory_allocated()/1024**2:.0f} MB', flush=True)

S = 512
vocab = 151936
input_ids = torch.randint(0, vocab, (1, S), device='cuda')
clean_ids = input_ids.clone()
doc_ids = torch.ones(1, S, dtype=torch.long, device='cuda')
ids2 = torch.cat([input_ids, clean_ids], dim=1)
docs2 = torch.cat([doc_ids, doc_ids], dim=1)

mask_bool = batch_doc_attention_mask(
    docs2, causal=False, mask_mode='composite_llada20', block_size=None, base_len=S
)
attn4d = _attn_for_model(mask_bool, model)
del mask_bool
gc.collect(); torch.cuda.empty_cache()
print(f'after mask [1,1,{2*S},{2*S}]: {torch.cuda.memory_allocated()/1024**2:.0f} MB', flush=True)

# --- Phase A: Forward + Backward (single turn) ---
torch.cuda.synchronize()
t0 = time.time()
model.zero_grad()
out = model(input_ids=ids2, attention_mask=attn4d)
logits = out.logits[:, :S, :].contiguous()
del out
targets = torch.randint(0, vocab, (1, S), device='cuda')
loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
loss.backward()
torch.cuda.synchronize()
dt = time.time() - t0
peak = torch.cuda.max_memory_allocated() / 1024**2
print(f'fwd+bwd S={S} composite={2*S}: loss={loss.item():.4f} '
      f'time={dt*1000:.0f}ms peak={peak:.0f}MB', flush=True)
del logits, loss, targets
gc.collect(); torch.cuda.empty_cache()

# --- Phase B: MTF 2 turns ---
model.zero_grad()
torch.cuda.reset_peak_memory_stats()
torch.cuda.synchronize()
t0 = time.time()
current = input_ids.clone()
for turn in range(2):
    ids2t = torch.cat([current, clean_ids], dim=1)
    docs2t = torch.cat([doc_ids, doc_ids], dim=1)
    mk = batch_doc_attention_mask(
        docs2t, causal=False, mask_mode='composite_llada20', block_size=None, base_len=S
    )
    a4d = _attn_for_model(mk, model)
    del mk
    out = model(input_ids=ids2t, attention_mask=a4d)
    logits = out.logits[:, :S, :].contiguous()
    del out, ids2t, docs2t, a4d
    loss = torch.nn.functional.cross_entropy(
        logits.view(-1, logits.size(-1)), clean_ids.view(-1)
    )
    (loss / 2.0 / 8.0).backward()
    if turn == 0:
        with torch.no_grad():
            preds = logits.argmax(-1)
            current = clean_ids.clone()
            current[:, 1:] = preds[:, :-1]
    del logits, loss
    torch.cuda.synchronize()
    print(f'  turn {turn}: alloc={torch.cuda.memory_allocated()/1024**2:.0f} MB', flush=True)

torch.cuda.synchronize()
dt = time.time() - t0
peak = torch.cuda.max_memory_allocated() / 1024**2
print(f'MTF 2-turn S={S} composite={2*S}: time={dt*1000:.0f}ms peak={peak:.0f}MB', flush=True)
print()

limit_85 = 8192 * 0.85
limit_70 = 8192 * 0.70
ok_85 = "OK" if peak < limit_85 else "OVER"
ok_70 = "OK" if peak < limit_70 else "OVER"
print(f'CAPACITY CHECK: peak {peak:.0f} MB vs 85%={limit_85:.0f} MB => {ok_85}')
print(f'CAPACITY CHECK: peak {peak:.0f} MB vs 70%={limit_70:.0f} MB => {ok_70}')
del model
gc.collect(); torch.cuda.empty_cache()
print('cleanup done', flush=True)
