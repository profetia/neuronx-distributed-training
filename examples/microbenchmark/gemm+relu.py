import torch
import torch_xla
import torch_xla.core.xla_model as xm
import torch.utils.benchmark as benchmark


M, K, N, use_bias = 256, 256, 256, False
# M, K, N, use_bias = 256, 256, 256, True
# M, K, N, use_bias = 512, 512, 512, False
# M, K, N, use_bias = 512, 512, 512, True
# M, K, N, use_bias = 1024, 1024, 1024, False
# M, K, N, use_bias = 1024, 1024, 1024, True
# M, K, N, use_bias = 2048, 2048, 2048, False
# M, K, N, use_bias = 2048, 2048, 2048, True
# M, K, N, use_bias = 4096, 4096, 4096, False
# M, K, N, use_bias = 4096, 4096, 4096, True
# M, K, N, use_bias = 8192, 8192, 8192, False
# M, K, N, use_bias = 8192, 8192, 8192, True
# M, K, N, use_bias = 16384, 16384, 16384, False
# M, K, N, use_bias = 16384, 16384, 16384, True
dtype = torch.float16
device = 'xla:0'
warmup_iters = 50
measure_iters = 10000

gl_pre_H = K 
gl_post_H = N

A = torch.randn(M, K, dtype=dtype, device=device)
B = torch.randn(K, N, dtype=dtype, device=device)
bias = torch.randn(N, dtype=dtype, device=device) if use_bias else None

xm.mark_step()

def composite_op():
    x = torch.nn.functional.relu(A)
    y = torch.mm(x, B)
    if use_bias:
        y = y + bias
    z = torch.nn.functional.relu(y)
    xm.mark_step()
    return z

print(f"Warm-up ({warmup_iters} iterations)...")
for _ in range(warmup_iters):
    composite_op()
xm.mark_step()


if use_bias:
    gemm_flops = 2.0 * M * K * N + 2.0 * M * N
else:
    gemm_flops = 2.0 * M * K * N

rl_pre_flops = M * K
rl_post_flops = M * N

workload_per_iter = gemm_flops + rl_pre_flops + rl_post_flops
workload_only_gemm_per_iter = gemm_flops

print(f"\nBenchmarking {measure_iters} iterations of GEMM + Relu [{dtype}]...")
t = benchmark.Timer(
    stmt="composite_op()",
    globals={"composite_op": composite_op},
    num_threads=1
)

all_tflops = []
all_tflops_only_gemm = []

print("\nStreaming TFLOPs per iteration:")
try:
    for i in range(1, measure_iters + 1):
        t_iter = t.timeit(number=1).mean
        tflops_iter = workload_per_iter / (t_iter * 1e12)
        tflops_iter_only_gemm = workload_only_gemm_per_iter / (t_iter * 1e12)
        print(f"Iter {i:2d}: {tflops_iter:.2f} TFLOPs, {t_iter*1000:.3f} ms")
        all_tflops.append(tflops_iter)
        all_tflops_only_gemm.append(tflops_iter_only_gemm)
except KeyboardInterrupt:
    print("\nBenchmarking interrupted by user.")
finally:
    avg_tflops = sum(all_tflops) / len(all_tflops)
    print(f"\nAverage TFLOPs over {len(all_tflops)} iterations: {avg_tflops:.2f} TFLOPs")
    avg_tflops_only_gemm = sum(all_tflops_only_gemm) / len(all_tflops_only_gemm)
    print(f"Average TFLOPs (only GEMM) over {len(all_tflops_only_gemm)} iterations: {avg_tflops_only_gemm:.2f} TFLOPs")
