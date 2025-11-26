import torch
import torch_xla
import torch_xla.core.xla_model as xm
import torch.utils.benchmark as benchmark

# M, K, N, use_bias = 256, 256, 256, False
# M, K, N, use_bias = 256, 256, 256, True
# M, K, N, use_bias = 512, 512, 512, False
# M, K, N, use_bias = 512, 512, 512, True
# M, K, N, use_bias = 1024, 1024, 1024, False
# M, K, N, use_bias = 1024, 1024, 1024, True
# M, K, N, use_bias = 2048, 2048, 2048, False
# M, K, N, use_bias = 2048, 2048, 2048, True
M, K, N, use_bias = 4096, 4096, 4096, False
# M, K, N, use_bias = 4096, 4096, 4096, True
# M, K, N, use_bias = 8192, 8192, 8192, False
# M, K, N, use_bias = 8192, 8192, 8192, True
# M, K, N, use_bias = 16384, 16384, 16384, False
# M, K, N, use_bias = 16384, 16384, 16384, True
dtype = torch.float16
device = 'xla:0'
warmup_iters = 50
measure_iters = 10000

A = torch.randn(M, K, dtype=dtype, device=device)
B = torch.randn(K, N, dtype=dtype, device=device)
bias = torch.randn(N, dtype=dtype, device=device)

xm.mark_step()

if use_bias:
    def gemm_op():
        C = torch.mm(A, B) + bias
        xm.mark_step()
else:
    def gemm_op():
        C = torch.mm(A, B)
        xm.mark_step()

print(f"Warm-up ({warmup_iters} iterations)...")
for _ in range(warmup_iters):
    gemm_op()

xm.mark_step()

print(f"\nBenchmarking {measure_iters} iterations of {M}x{K} * {K}x{N} GEMM ({dtype})...")
t = benchmark.Timer(
    stmt="gemm_op()",
    globals={"gemm_op": gemm_op},
    num_threads=1
)


all_tflops = []

if use_bias:
    workload_per_iter = 2.0 * M * K * N + 2.0 * M * N
else:
    workload_per_iter = 2.0 * M * K * N

print("\nStreaming TFLOPs per iteration:")
try:
    for i in range(1, measure_iters + 1):
        t_iter = t.timeit(number=1).mean
        tflops_iter = workload_per_iter / (t_iter * 1e12)
        print(f"Iter {i:2d}: {tflops_iter:.2f} TFLOPs, {t_iter * 1000:.3f} ms")
        all_tflops.append(tflops_iter)
except KeyboardInterrupt:
    print("\nBenchmarking interrupted by user.")
finally:
    print("\nFinal Results:")
    avg_tflops = sum(all_tflops) / len(all_tflops)
    print(f"Average TFLOPs over {measure_iters} iterations: {avg_tflops:.2f} TFLOPs")