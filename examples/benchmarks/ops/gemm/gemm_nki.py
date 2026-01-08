import argparse
import csv
import os

import torch
import torch.distributed as dist
import torch.utils.benchmark as benchmark
import torch_xla as xla

from neuronxcc import nki
import neuronxcc.nki.language as nl
import neuronxcc.nki.isa as ni


def parse_int_list(s: str):
    # Accept comma-separated ints, e.g. "256,512,1024"
    return [int(x) for x in s.split(",") if x.strip()]


# pmax = 128
# gemm_stationary_fmax = 128
# gemm_moving_fmax = 512
# TILES_IN_BLOCK_K=8, TILES_IN_BLOCK_M=4, TILES_IN_BLOCK_N=4
@nki.jit
def matmul(A_DRAM, B_DRAM, TILES_IN_BLOCK_K=8, TILES_IN_BLOCK_M=4, TILES_IN_BLOCK_N=2):
  """
  Optimized matrix multiplication kernel

   Args:

      A_DRAM: an input tensor of shape [K, M], where K is a multiple of 1024
      and M is a multiple of 512.  It is the left-hand-side argument of the
      matrix multiplication, delivered transposed for optimal performance.

      B_DRAM: an input tensor of shape [K, N],  where K is a multiple of 1024
        and N is a multiple of 2048.  It is the right-hand-side argument of
        the matrix multiplication.

      Z_DRAM: the resulting output tensor of shape [M, N]

  """
  K, M = A_DRAM.shape
  _, N = B_DRAM.shape

  Z_DRAM = nl.ndarray([M, N], dtype=A_DRAM.dtype, buffer=nl.shared_hbm)

  TILE_K = nl.tile_size.pmax
  TILE_M = nl.tile_size.gemm_stationary_fmax
  TILE_N = nl.tile_size.gemm_moving_fmax

  NUM_BLOCK_K = K // (TILES_IN_BLOCK_K * TILE_K)
  NUM_BLOCK_M = M // (TILES_IN_BLOCK_M * TILE_M)
  NUM_BLOCK_N = N // (TILES_IN_BLOCK_N * TILE_N)

  assert NUM_BLOCK_K * TILES_IN_BLOCK_K * TILE_K == K
  assert NUM_BLOCK_M * TILES_IN_BLOCK_M * TILE_M == M
  assert NUM_BLOCK_N * TILES_IN_BLOCK_N * TILE_N == N

  for n2 in nl.affine_range(NUM_BLOCK_N):
    for m2 in nl.affine_range(NUM_BLOCK_M):

      # Partition Z and then ensure that we are Z-block stationary
      # This way, no matter how large K, M, and N are, Z is never spilled/loaded
      # We only need to store once
      Z_SBUF = nl.zeros((TILES_IN_BLOCK_M, nl.par_dim(TILE_M), TILES_IN_BLOCK_N * TILE_N), dtype=Z_DRAM.dtype, buffer=nl.sbuf)

      for k2 in nl.affine_range(NUM_BLOCK_K):
        A_SBUF = nl.ndarray((TILES_IN_BLOCK_K, nl.par_dim(TILE_K), TILES_IN_BLOCK_M * TILE_M), dtype=A_DRAM.dtype, buffer=nl.sbuf)
        B_SBUF = nl.ndarray((TILES_IN_BLOCK_K, nl.par_dim(TILE_K), TILES_IN_BLOCK_N * TILE_N), dtype=B_DRAM.dtype, buffer=nl.sbuf)

        # Load in a block of A and a block of B
        for k1 in nl.affine_range(TILES_IN_BLOCK_K):
          k_start = k2 * TILES_IN_BLOCK_K * TILE_K + k1 * TILE_K
          k_end = k_start + TILE_K

          m_start = m2 * TILES_IN_BLOCK_M * TILE_M
          m_end = m_start + TILES_IN_BLOCK_M * TILE_M

          n_start = n2 * TILES_IN_BLOCK_N * TILE_N
          n_end = n_start + TILES_IN_BLOCK_N * TILE_N

          # We coalesce memory accesses by loading TILES_IN_BLOCK_M * TILE_M
          # values of A at a time. We cannot coalesce across K because K gets
          # split across the partition dimension
          A_SBUF[k1] = nl.load(A_DRAM[k_start:k_end, m_start:m_end])

          # We coalesce memory accesses by loading TILES_IN_BLOCK_N * TILE_N
          # values of B at a time. We cannot coalesce across K because K gets
          # split across the partition dimension
          B_SBUF[k1] = nl.load(B_DRAM[k_start:k_end, n_start:n_end])

        for m1 in nl.affine_range(TILES_IN_BLOCK_M):
          for n1 in nl.affine_range(TILES_IN_BLOCK_N):
            # Keep the tile of Z stationary in the PSUM buffer to minimize the
            # number of calls to nl.loop_reduce
            Z_PSUM = nl.zeros((TILE_M, TILE_N), dtype=nl.float32, buffer=nl.psum)

            m_start = m1 * TILE_M
            m_end = m_start + TILE_M

            n_start = n1 * TILE_N
            n_end = n_start + TILE_N

            for k1 in nl.affine_range(TILES_IN_BLOCK_K):
              Z_PSUM += ni.nc_matmul(A_SBUF[k1, :, m_start:m_end], B_SBUF[k1, :, n_start:n_end])

            Z_SBUF[m1, :, n_start:n_end] = nl.loop_reduce(Z_PSUM, op=nl.add, loop_indices=[k2], dtype=Z_DRAM.dtype)

      for m1 in nl.affine_range(TILES_IN_BLOCK_M):
        m_start = m2 * TILES_IN_BLOCK_M * TILE_M + m1 * TILE_M
        m_end = m_start + TILE_M

        n_start = n2 * TILES_IN_BLOCK_N * TILE_N
        n_end = n_start + TILES_IN_BLOCK_N * TILE_N

        # We coalesce memory accesses by storing TILES_IN_BLOCK_N * TILE_N
        # values of Z at a time. We cannot coalesce across M because M gets
        # split across the partition dimension
        nl.store(Z_DRAM[m_start:m_end, n_start:n_end], value=Z_SBUF[m1])

  return Z_DRAM


def run_one(M: int, K: int, dtype, device: str, warmup_iters: int, measure_iters: int):
    N = M  # M = N as requested

    A = torch.empty(M, K, dtype=dtype, device=device)
    B = torch.empty(K, N, dtype=dtype, device=device)

    xla.sync(wait=True)

    def gemm_op():
        C = matmul(A.T, B)
        xla.sync(wait=True)

    for _ in range(warmup_iters):
        gemm_op()

    xla.sync(wait=True)

    t = benchmark.Timer(
        stmt="gemm_op()",
        globals={"gemm_op": gemm_op},
        num_threads=1
    )

    workload_per_iter = 2.0 * M * K * N

    measured = t.timeit(measure_iters)
    avg_ms = measured.mean * 1000.0  # seconds to milliseconds
    avg_tflops = (workload_per_iter / 1.0e12) / (avg_ms / 1000.0)  # TFLOPs
    return avg_tflops, avg_ms


def main():
    parser = argparse.ArgumentParser()
    # Sweep ranges (inclusive end)
    parser.add_argument("--m_start", type=int, required=True)
    parser.add_argument("--m_end", type=int, required=True)
    parser.add_argument("--m_step", type=int, default=16)
    parser.add_argument("--k_start", type=int, required=True)
    parser.add_argument("--k_end", type=int, required=True)
    parser.add_argument("--k_step", type=int, default=16)

    args = parser.parse_args()

    if args.m_step <= 0 or args.k_step <= 0:
        raise ValueError("m_step and k_step must be > 0")
    if args.m_end < args.m_start or args.k_end < args.k_start:
        raise ValueError("end must be >= start")

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))

    os.environ["NEURON_RT_VISIBLE_CORES"] = str(rank * 2) # Avoid sharing cores on the same chip

    dist.init_process_group(backend="xla")

    dtype = torch.float16
    device = "xla"
    warmup_iters = 5
    measure_iters = 25

    flush_interval = 10

    # partition by m
    m_step = (args.m_end - args.m_start + world_size - 1) // world_size
    m_start = args.m_start + rank * m_step
    m_end = args.m_start + (rank + 1) * m_step

    if m_start == 0:
        m_start = args.m_step  # avoid zero dimension

    if args.k_start == 0:
        args.k_start = args.k_step  # avoid zero dimension

    output_csv = f"gemm_nki_benchmark_k{args.k_start}-{args.k_end}-{args.k_step}_m{m_start}-{m_end}-{args.m_step}_rank{rank}.csv"
    if os.path.exists(output_csv):
        # Recover from previous run
        with open(output_csv, "r", newline="") as f:
            reader = csv.reader(f)
            existing_rows = list(reader)

            try:
                last_m = int(existing_rows[-1][0])
                last_k = int(existing_rows[-1][1])
                should_resume = True
            except Exception:
                last_m = m_start - args.m_step
                should_resume = False
            
        if should_resume:
            print(f"Resuming from M={last_m}, K={last_k}...")
            with open(output_csv, "a", newline="") as f:
                writer = csv.writer(f)

                # Resume last unfinished k for last_m
                for i, K in enumerate(range(last_k + args.k_step, args.k_end + 1, args.k_step)):
                    avg_tflops, avg_ms = run_one(
                        M=last_m,
                        K=K,
                        dtype=dtype,
                        device=device,
                        warmup_iters=warmup_iters,
                        measure_iters=measure_iters,
                    )
                    N = last_m
                    writer.writerow([last_m, K, N, str(dtype), f"{avg_tflops:.6f}", f"{avg_ms:.6f}"])
                    print(f"M=N={last_m}, K={K} -> {avg_tflops:.2f} TFLOPs, {avg_ms:.3f} ms")
                    if (i + 1) % flush_interval == 0:
                        f.flush()

        m_start = last_m + args.m_step
        print(f"Continuing from M={m_start}...")

    m_values = range(m_start, m_end, args.m_step)
    if rank == world_size - 1:
        # Last rank includes the end value
        m_values = range(m_start, m_end + 1, args.m_step)
    k_values = range(args.k_start, args.k_end + 1, args.k_step)

    with open(output_csv, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["M", "K", "N", "dtype", "avg_tflops", "avg_ms"])

        for M in m_values:
            for i, K in enumerate(k_values):
                avg_tflops, avg_ms = run_one(
                    M=M,
                    K=K,
                    dtype=dtype,
                    device=device,
                    warmup_iters=warmup_iters,
                    measure_iters=measure_iters,
                )
                N = M
                writer.writerow([M, K, N, str(dtype), f"{avg_tflops:.6f}", f"{avg_ms:.6f}"])
                print(f"M=N={M}, K={K} -> {avg_tflops:.2f} TFLOPs, {avg_ms:.3f} ms")
                if (i + 1) % flush_interval == 0:
                    f.flush()

    print(f"\nWrote results to: {output_csv}")


if __name__ == "__main__":
    main()
