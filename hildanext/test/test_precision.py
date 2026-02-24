# Precision tests for fp16/fp32 mapping and tiny forward.
# Entrypoints: unittest test methods.
# Verifies dtype routing and numerical path validity.
from __future__ import annotations
import unittest
import torch
from hildanext.utils import choose_device,dtype_from_name,TinyCausalLM
from reporting import emit_payload

class PrecisionTests(unittest.TestCase):
    def test_dtype_mapping_cpu(self):
        d=torch.device("cpu")
        emit_payload(
            "test_dtype_mapping_cpu",
            "Checks dtype routing on CPU for fp32/fp16/bf16 inputs.",
            {
                "float32":str(dtype_from_name("float32",d)),
                "fp16":str(dtype_from_name("fp16",d)),
                "bfloat16":str(dtype_from_name("bfloat16",d))
            }
        )
        self.assertEqual(dtype_from_name("float32",d),torch.float32)
        self.assertEqual(dtype_from_name("fp16",d),torch.float32)
        self.assertEqual(dtype_from_name("bfloat16",d),torch.float32)
    def test_tiny_forward_fp32(self):
        m=TinyCausalLM(vocab_size=128,hidden_size=32)
        x=torch.randint(0,128,(2,16),dtype=torch.long)
        y=m(input_ids=x).logits
        emit_payload(
            "test_tiny_forward_fp32",
            "Runs TinyCausalLM forward in fp32.",
            {"shape":list(y.shape),"dtype":str(y.dtype),"finite":bool(torch.isfinite(y).all().item())}
        )
        self.assertEqual(tuple(y.shape),(2,16,128))
        self.assertTrue(torch.isfinite(y).all().item())
    def test_tiny_forward_fp16_if_cuda(self):
        if not torch.cuda.is_available():
            self.skipTest("cuda not available")
        dev=torch.device("cuda")
        m=TinyCausalLM(vocab_size=256,hidden_size=64).to(dev).half()
        x=torch.randint(0,256,(1,8),dtype=torch.long,device=dev)
        y=m(input_ids=x).logits
        emit_payload(
            "test_tiny_forward_fp16_if_cuda",
            "Runs TinyCausalLM forward in fp16 on CUDA.",
            {"device":str(dev),"shape":list(y.shape),"dtype":str(y.dtype),"finite":bool(torch.isfinite(y).all().item())}
        )
        self.assertEqual(y.dtype,torch.float16)
        self.assertTrue(torch.isfinite(y).all().item())

if __name__=="__main__":
    unittest.main()
