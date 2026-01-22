import torch
import torch_xla
import torch_xla.core.xla_model as xm
import torch.utils.benchmark as benchmark

# B, S, H, use_affine = 1, 128, 768, False
# B, S, H, use_affine = 1, 128, 768, True
# B, S, H, use_affine = 4, 512, 1024, False
# B, S, H, use_affine = 4, 512, 1024, True
# B, S, H, use_affine = 8, 1024, 2048, False
# B, S, H, use_affine = 8, 1024, 2048, True
# B, S, H, use_affine = 16, 1024, 4096, False
# B, S, H, use_affine = 16, 1024, 4096, True
# B, S, H, use_affine = 32, 2048, 8192, False
# B, S, H, use_affine = 32, 2048, 8192, True
# B, S, H, use_affine = 64, 2048, 16384, False
B, S, H, use_affine = 64, 2048, 16384, True

dtype = torch.float16
device = 'xla:0'
warmup_iters = 50
measure_iters = 10000
eps = 1e-5


x = torch.randn(B, S, H, dtype=dtype, device=device)
if use_affine:
    weight = torch.randn(H, dtype=dtype, device=device)
    bias = torch.randn(H, dtype=dtype, device=device)
else:
    weight = bias = None

xm.mark_step()

if use_affine:
    def layernorm_op():
        with torch.no_grad():
            y = torch.nn.functional.layer_norm(x, (H,), weight=weight, bias=bias, eps=eps)
        xm.mark_step()
else:
    def layernorm_op():
        with torch.no_grad():
            y = torch.nn.functional.layer_norm(x, (H,), eps=eps)
        xm.mark_step()

print(f"Warm-up ({warmup_iters} iterations)...")
for _ in range(warmup_iters):
    layernorm_op()
xm.mark_step()


num_elements = B * S * H
if use_affine:
    workload_per_iter = num_elements * 8.0
else:
    workload_per_iter = num_elements * 6.0


print(f"\nBenchmarking {measure_iters} iterations of LayerNorm({B}, {S}, {H}) [{dtype}]...")
t = benchmark.Timer(
    stmt="layernorm_op()",
    globals={"layernorm_op": layernorm_op},
    num_threads=1
)

all_tflops = []

print("\nStreaming TFLOPs per iteration:")
try:
    for i in range(1, measure_iters + 1):
        t_iter = t.timeit(number=1).mean
        tflops_iter = workload_per_iter / (t_iter * 1e12)
        print(f"Iter {i:2d}: {tflops_iter:.4f} TFLOPs, {t_iter * 1000:.3f} ms")
        all_tflops.append(tflops_iter)
except KeyboardInterrupt:
    print("\nBenchmarking interrupted by user.")
finally:
    print("\nFinal Results:")
    avg_tflops = sum(all_tflops) / len(all_tflops)
    print(f"Average TFLOPs over {len(all_tflops)} iterations: {avg_tflops:.4f} TFLOPs")
