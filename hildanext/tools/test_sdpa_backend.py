# SDPA Backend Diagnostic — verifies which backend PyTorch dispatches to
# on this hardware with HildaNext's composite attention masks.
# Usage: conda activate mdm && python tools/test_sdpa_backend.py
import sys,time
import torch
import torch.nn.functional as F

def main():
    if not torch.cuda.is_available():
        print("CUDA not available — cannot test SDPA backends"); return
    dev=torch.device("cuda")
    props=torch.cuda.get_device_properties(0)
    sm=props.major*10+props.minor
    _tmem=getattr(props,"total_memory",None) or getattr(props,"total_mem",0)
    print(f"GPU: {props.name}  sm_{sm}  VRAM: {_tmem//1024//1024}MB")
    print(f"PyTorch: {torch.__version__}  CUDA: {torch.version.cuda}")
    print()
    # Qwen3-0.6B dimensions
    B,H,S,D=1,16,2048,128  # composite mode doubles seq to 2048
    H_kv=8  # GQA: 8 KV heads, 16 Q heads (HF repeat_kv expands to H=16)
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.enable_math_sdp(True)
    print("Backend config: flash=OFF  mem_efficient=ON  math=ON (fallback)")
    print()
    tests=[
        ("no_mask",None,False),
        ("causal_only",None,True),
        ("2d_padding",torch.ones(B,S,device=dev,dtype=torch.float16),False),
        ("4d_simple_block_diag","simple",False),
        ("4d_composite_llada20","composite",False),
    ]
    for name,mask_spec,is_causal in tests:
        q=torch.randn(B,H,S,D,device=dev,dtype=torch.float16)
        k=torch.randn(B,H,S,D,device=dev,dtype=torch.float16)
        v=torch.randn(B,H,S,D,device=dev,dtype=torch.float16)
        mask=None
        if mask_spec=="simple":
            mask=torch.zeros(B,1,S,S,device=dev,dtype=torch.float16)
            bs=32
            for i in range(0,S,bs):
                end=min(i+bs,S)
                mask[:,:,i:end,i:end]=0.0
            mask[mask==0.0]=0.0
            neg=torch.finfo(torch.float16).min
            for i in range(0,S,bs):
                for j in range(0,S,bs):
                    if i!=j:
                        ie,je=min(i+bs,S),min(j+bs,S)
                        mask[:,:,i:ie,j:je]=neg
        elif mask_spec=="composite":
            half=S//2
            mask=torch.zeros(B,1,S,S,device=dev,dtype=torch.float16)
            neg=torch.finfo(torch.float16).min
            mask[:,:,half:,:half]=neg
            bs=32
            for i in range(0,half,bs):
                for j in range(0,half,bs):
                    if i!=j:
                        ie,je=min(i+bs,half),min(j+bs,half)
                        mask[:,:,i:ie,j:je]=neg
            for i in range(0,half,bs):
                for j in range(half,S,bs):
                    bi,bj=i//bs,(j-half)//bs
                    if bi<=bj:
                        continue
                    ie,je=min(i+bs,half),min(j+bs,S)
                    mask[:,:,i:ie,j:je]=neg
            for i in range(half,S,bs):
                for j in range(half,S,bs):
                    bi,bj=(i-half)//bs,(j-half)//bs
                    if bi<bj:
                        ie,je=min(i+bs,S),min(j+bs,S)
                        mask[:,:,i:ie,j:je]=neg
        elif isinstance(mask_spec,torch.Tensor):
            mask=mask_spec
        try:
            backend_id=torch._fused_sdp_choice(q,k,v,attn_mask=mask,dropout_p=0.0,is_causal=is_causal)
            from torch.nn.attention import SDPBackend
            backend=SDPBackend(backend_id)
            tag="OK" if backend==SDPBackend.EFFICIENT_ATTENTION else "FALLBACK"
        except Exception as e:
            backend=None
            tag=f"ERROR: {e}"
        # Benchmark
        torch.cuda.synchronize()
        t0=time.perf_counter()
        N=20
        for _ in range(N):
            _=F.scaled_dot_product_attention(q,k,v,attn_mask=mask,dropout_p=0.0,is_causal=is_causal)
        torch.cuda.synchronize()
        ms=(time.perf_counter()-t0)/N*1000
        bname=backend.name if backend else "???"
        print(f"  [{tag:8s}] {name:30s} -> {bname:25s}  {ms:.2f} ms/call")
        # Debug info for failures
        if backend and backend!=SDPBackend.EFFICIENT_ATTENTION:
            try:
                from torch.backends.cuda import SDPAParams,can_use_efficient_attention
                params=SDPAParams(q,k,v,mask,0.0,is_causal)
                can_use_efficient_attention(params,debug=True)
            except Exception:
                pass
        del q,k,v,mask
        torch.cuda.empty_cache()
    # Compare speed: mem_efficient vs math on composite mask
    print()
    print("--- Speed comparison: mem_efficient vs math (composite mask, 20 iterations) ---")
    q=torch.randn(B,H,S,D,device=dev,dtype=torch.float16)
    k=torch.randn(B,H,S,D,device=dev,dtype=torch.float16)
    v=torch.randn(B,H,S,D,device=dev,dtype=torch.float16)
    half=S//2
    mask=torch.zeros(B,1,S,S,device=dev,dtype=torch.float16)
    neg=torch.finfo(torch.float16).min
    mask[:,:,half:,:half]=neg
    from torch.nn.attention import sdpa_kernel,SDPBackend
    for backend_choice,label in [(SDPBackend.EFFICIENT_ATTENTION,"mem_efficient"),(SDPBackend.MATH,"math")]:
        try:
            torch.cuda.synchronize()
            with sdpa_kernel(backend_choice):
                _=F.scaled_dot_product_attention(q,k,v,attn_mask=mask,dropout_p=0.0)
            torch.cuda.synchronize()
            t0=time.perf_counter()
            N=20
            with sdpa_kernel(backend_choice):
                for _ in range(N):
                    _=F.scaled_dot_product_attention(q,k,v,attn_mask=mask,dropout_p=0.0)
            torch.cuda.synchronize()
            ms=(time.perf_counter()-t0)/N*1000
            print(f"  {label:20s}: {ms:.2f} ms/call")
        except Exception as e:
            print(f"  {label:20s}: FAILED — {e}")
    speedup_note="If mem_efficient < math → our enable_mem_efficient_sdp(True) is genuinely faster."
    print(f"\n  {speedup_note}")

if __name__=="__main__":
    main()
