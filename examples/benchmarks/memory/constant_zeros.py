import torch

import torch_xla
import torch_xla.core.xla_model as xm

N = 16 * 1024
K = 4 * 1024
M = 16 * 1024

print(f"Running matmul of {N} x {K} and {K} x {M} matrices on XLA device")
print(f"NxK: {N}x{K} (size: {N*K*2/1e6} MB)")
print(f"KxM: {K}x{M} (size: {K*M*2/1e6} MB)")
print(f"NxM: {N}x{M} (size: {N*M*2/1e6} MB)")

dtype = torch.bfloat16
device = torch_xla.device()

# The difference between the two ways of creating the acc tensor is that
# the second way does not create a `constant` in the compiled artifact
# acc = torch.zeros((N, M), device=device, dtype=dtype)

# acc = torch.zeros((N, M), dtype=dtype).to(device) # This does not create a constant

# base = torch.empty((N, M), dtype=dtype, device="meta")
# acc = torch.zeros_like(base, device=device)
torch_xla.sync()
for _ in range(10):
    a = torch.empty((N, K), device=device, dtype=dtype)
    b = torch.empty((K, M), device=device, dtype=dtype)
    acc += torch.matmul(a, b)
    torch_xla.sync()

input("Press Enter to exit...")
