import torch
import torch_xla
import torch_xla.core.xla_model as xm
import torch.utils.benchmark as benchmark

# M, N = 1024, 1024
# M, N = 2048, 2048
# M, N = 4096, 4096
# M, N = 8192, 8192
M, N = 16384, 16384
# M, N = 32768, 32768

dtype = torch.float16
device = 'xla:0'
warmup_iters = 50
measure_iters = 10000

A = torch.randn(M, N, dtype=dtype, device=device)
xm.mark_step()

def relu_op():
    C = torch.relu(A)
    xm.mark_step()

print(f"Warm-up ({warmup_iters} iterations)...")
for _ in range(warmup_iters):
    relu_op()
xm.mark_step()


num_elements = M * N
workload_per_iter = 1.0 * num_elements


print(f"\nBenchmarking {measure_iters} iterations of ReLU({M}x{N}) [{dtype}]...")
t = benchmark.Timer(
    stmt="relu_op()",
    globals={"relu_op": relu_op},
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
