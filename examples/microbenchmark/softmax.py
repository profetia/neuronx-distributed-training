import torch
import torch_xla
import torch_xla.core.xla_model as xm
import torch.utils.benchmark as benchmark

# B, S, H = 1, 128, 768
# B, S, H = 4, 512, 1024
# B, S, H = 8, 1024, 2048
# B, S, H = 16, 1024, 4096
B, S, H = 32, 2048, 8192

dtype = torch.float16
device = 'xla:0'
warmup_iters = 50
measure_iters = 10000
eps = 1e-5


x = torch.randn(B, S, H, dtype=dtype, device=device)

xm.mark_step()

def softmax_op():
    with torch.no_grad():
        y = torch.nn.functional.softmax(x, dim=-1)
    xm.mark_step()

print(f"Warm-up ({warmup_iters} iterations)...")
for _ in range(warmup_iters):
    softmax_op()
xm.mark_step()


num_elements = B * S * H
workload_per_iter = num_elements * 5.0


print(f"\nBenchmarking {measure_iters} iterations of Softmax({B}, {S}, {H}) [{dtype}]...")
t = benchmark.Timer(
    stmt="softmax_op()",
    globals={"softmax_op": softmax_op},
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
