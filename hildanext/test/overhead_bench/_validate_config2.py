"""Quick validation that all Config 2 fixes are working."""
import os
os.environ["CUDA_MODULE_LOADING"] = "LAZY"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:512"

import torch

print(f"CUDA_MODULE_LOADING = {os.environ.get('CUDA_MODULE_LOADING')}")
print(f"ALLOC_CONF = {os.environ.get('PYTORCH_CUDA_ALLOC_CONF')}")
print(f"CUDA available: {torch.cuda.is_available()}")

# Test memory fraction cap (P0)
torch.cuda.set_per_process_memory_fraction(0.85, device=0)
props = torch.cuda.get_device_properties(0)
cap_mb = int(props.total_memory * 0.85) // (1024 * 1024)
print(f"GPU: {props.name}, VRAM: {props.total_memory // (1024*1024)} MB, cap: {cap_mb} MB")

# Test pynvml (P1)
import pynvml
pynvml.nvmlInit()
h = pynvml.nvmlDeviceGetHandleByIndex(0)
t = pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU)
print(f"pynvml temp: {t} C")

# Test bitsandbytes PagedAdamW8bit
import bitsandbytes as bnb
has_paged = hasattr(bnb.optim, "PagedAdamW8bit")
print(f"PagedAdamW8bit available: {has_paged}")

# Test _select_optimizer_name
from hildanext.wsd_stage0 import _select_optimizer_name
opt_name = _select_optimizer_name()
print(f"_select_optimizer_name() = {opt_name}")

# Test _gpu_temp_celsius via pynvml
from hildanext.training import _gpu_temp_celsius
gpu_t = _gpu_temp_celsius()
print(f"_gpu_temp_celsius() = {gpu_t} C")

assert has_paged, "PagedAdamW8bit not available!"
assert opt_name == "bnb_paged_adamw8bit", f"Wrong optimizer: {opt_name}"
assert gpu_t is not None, "pynvml failed in _gpu_temp_celsius"
print("\nALL CHECKS PASSED")
