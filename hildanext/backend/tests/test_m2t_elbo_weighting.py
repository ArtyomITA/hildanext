# M2T ELBO weighting (loss_weighting="inv_t").
# Must equal LLaDA's sum over masked tokens of CE/t divided by all candidate positions,
# not the masked-token mean divided by t (which over-weights low-t batches up to 1000x).
from types import SimpleNamespace
import torch
import torch.nn.functional as F
from hildanext.config import TrainConfig
from hildanext.diffusion import compute_m2t_t2t_losses

V=32

class _Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.emb=torch.nn.Embedding(V+1,8)
        self.head=torch.nn.Linear(8,V)
    def forward(self,input_ids,attention_mask=None):
        return SimpleNamespace(logits=self.head(self.emb(input_ids)))

def _reference(model,ids,corrupted,t):
    mixed=ids.clone()
    mixed[corrupted]=V
    logits=model(mixed).logits[:,:-1,:]
    labels=ids[:,1:]
    masked=corrupted[:,1:]
    ce=F.cross_entropy(logits[masked],labels[masked],reduction="sum")
    return ce/t/labels.numel()

def test_inv_t_matches_llada_elbo():
    torch.manual_seed(0)
    model=_Tiny()
    cfg=TrainConfig(m2t_weight=1.0,t2t_weight=0.0,t2t_noise_ratio=0.0)
    for _ in range(20):
        ids=torch.randint(0,V,(4,64))
        out=compute_m2t_t2t_losses(model,ids,torch.ones_like(ids),torch.zeros_like(ids),None,mask_id=V,vocab_size=V,cfg=cfg,
                                   focus_response=False,time_param="continuous_time",loss_weighting="inv_t")
        ref=_reference(model,ids,out["corrupted_positions"],out["t_sampled"])
        assert torch.allclose(out["loss_m2t_scaled"],ref,rtol=1e-4,atol=1e-6),(out["t_sampled"],float(out["loss_m2t_scaled"]),float(ref))
        # the scaled loss stays on the scale of the raw masked mean (ratio ~ mask fraction / t ~ 1), never ~1/t
        assert float(out["loss_m2t_scaled"])<10*float(out["loss_m2t"])+1e-6
