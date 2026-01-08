import argparse
import csv
import itertools
import os

import torch
import torch.distributed as dist
import torch.utils.benchmark as benchmark
import torch_xla as xla


def parse_int_list(s: str):
    # Accept comma-separated ints, e.g. "256,512,1024"
    return [int(x) for x in s.split(",") if x.strip()]


def run_one(M: int, K: int, use_bias: bool, dtype, device: str, warmup_iters: int, measure_iters: int):
    N = M  # M = N as requested

    A = torch.empty(M, K, dtype=dtype, device=device)
    B = torch.empty(K, N, dtype=dtype, device=device)

    if use_bias:
        bias = torch.empty(N, dtype=dtype, device=device)

    xla.sync(wait=True)

    if use_bias:
        def gemm_op():
            C = torch.mm(A, B) + bias
            xla.sync(wait=True)
    else:
        def gemm_op():
            C = torch.mm(A, B)
            xla.sync(wait=True)

    for _ in range(warmup_iters):
        gemm_op()

    xla.sync(wait=True)

    t = benchmark.Timer(
        stmt="gemm_op()",
        globals={"gemm_op": gemm_op},
        num_threads=1
    )

    if use_bias:
        workload_per_iter = 2.0 * M * K * N + 2.0 * M * N
    else:
        workload_per_iter = 2.0 * M * K * N

    measured = t.timeit(measure_iters)
    avg_ms = measured.mean * 1000.0  # seconds to milliseconds
    avg_tflops = (workload_per_iter / 1.0e12) / (avg_ms / 1000.0)  # TFLOPs
    return avg_tflops, avg_ms


def run_many(inputs: list[tuple[int, int]], output_csv: str, use_bias: bool, dtype, device: str, warmup_iters: int, measure_iters: int):
    with open(output_csv, "a", newline="") as f:
        writer = csv.writer(f)
        # writer.writerow(["M", "K", "N", "use_bias", "dtype", "avg_tflops", "avg_ms"])

        for i, (M, K) in enumerate(inputs):
            avg_tflops, avg_ms = run_one(
                M=M,
                K=K,
                use_bias=use_bias,
                dtype=dtype,
                device=device,
                warmup_iters=warmup_iters,
                measure_iters=measure_iters,
            )
            N = M
            writer.writerow([M, K, N, use_bias, str(dtype), f"{avg_tflops:.6f}", f"{avg_ms:.6f}"])
            print(f"M=N={M}, K={K}, use_bias={use_bias} -> {avg_tflops:.2f} TFLOPs, {avg_ms:.3f} ms")
            if (i + 1) % 10 == 0:
                f.flush()


def try_recover_last_completed(output_csv: str):
    if not os.path.exists(output_csv):
        return None

    with open(output_csv, "r", newline="") as f:
        reader = csv.reader(f)
        existing_rows = list(reader)
        for row in reversed(existing_rows):
            try:
                last_m = int(row[0])
                last_k = int(row[1])
            except Exception:
                continue

            return last_m, last_k
        
    return None


def partition_by_m(args, world_size: int, rank: int, dtype, device: str, warmup_iters: int, measure_iters: int):
    # partition by m
    m_step = (args.m_end - args.m_start + world_size - 1) // world_size
    m_start = args.m_start + rank * m_step
    m_end = args.m_start + (rank + 1) * m_step

    if m_start == 0:
        m_start = args.m_step  # avoid zero dimension

    if args.k_start == 0:
        args.k_start = args.k_step  # avoid zero dimension

    output_csv = f"gemm_benchmark_k{args.k_start}-{args.k_end}-{args.k_step}_m{m_start}-{m_end}-{args.m_step}_rank{rank}.csv"
    last_completed = try_recover_last_completed(output_csv)
    if last_completed is not None:
        last_m, last_k = last_completed
        print(f"Resuming from M={last_m}, K={last_k}...")

        resume_inputs = list(
            itertools.product(
                [last_m],
                range(last_k + args.k_step, args.k_end + 1, args.k_step),
            )
        )
        m_start = last_m + args.m_step
    else:
        resume_inputs = []


    m_values = range(m_start, m_end, args.m_step)
    if rank == world_size - 1:
        # Last rank includes the end value
        m_values = range(m_start, m_end + 1, args.m_step)
    k_values = range(args.k_start, args.k_end + 1, args.k_step)

    run_many_inputs = resume_inputs + list(itertools.product(m_values, k_values))
    run_many(
        inputs=run_many_inputs,
        output_csv=output_csv,
        use_bias=args.use_bias,
        dtype=dtype,
        device=device,
        warmup_iters=warmup_iters,
        measure_iters=measure_iters,
    )

    print(f"\nWrote results to: {output_csv}")


def recover_completed_set(output_csv: str):
    completed = set()
    if not os.path.exists(output_csv):
        return completed
    
    with open(output_csv, "r", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            try:
                m = int(row[0])
                k = int(row[1])
                completed.add((m, k))
            except Exception:
                continue

    return completed


def partition_unfinished(args, world_size: int, rank: int, dtype, device: str, warmup_iters: int, measure_iters: int):
    last_completed = try_recover_last_completed(args.partition_unfinished)
    if last_completed is not None:
        last_m, last_k = last_completed
        print(f"Resuming from M={last_m}, K={last_k}...")
        resume_inputs = list(
            itertools.product(
                [last_m],
                range(last_k + args.k_step, args.k_end + 1, args.k_step),
            )
        )
        m_start = last_m + args.m_step
        k_start = args.k_start
    else:
        m_start = args.m_start
        k_start = args.k_start
        resume_inputs = []

    if m_start == 0:
        m_start = args.m_step  # avoid zero dimension
    if k_start == 0:
        k_start = args.k_step  # avoid zero dimension

    m_values = range(m_start, args.m_end + 1, args.m_step)
    k_values = range(k_start, args.k_end + 1, args.k_step)
    run_many_inputs = resume_inputs + list(itertools.product(m_values, k_values))

    # Partition the work among workers
    partitioned_inputs = []
    for i, inp in enumerate(run_many_inputs):
        if i % world_size == rank:
            partitioned_inputs.append(inp)
    
    output_csv = f"cond_worldsize{world_size}_rank{rank}_" + os.path.basename(args.partition_unfinished)
    # Recover existing results
    existing_results = recover_completed_set(output_csv)
    partitioned_inputs = [inp for inp in partitioned_inputs if inp not in existing_results]
    print(f"Rank {rank} has {len(partitioned_inputs)} unfinished inputs to process.")

    run_many(
        inputs=partitioned_inputs,
        output_csv=output_csv,
        use_bias=args.use_bias,
        dtype=dtype,
        device=device,
        warmup_iters=warmup_iters,
        measure_iters=measure_iters,
    )



def main():
    parser = argparse.ArgumentParser()
    # Sweep ranges (inclusive end)
    parser.add_argument("--m_start", type=int, required=True)
    parser.add_argument("--m_end", type=int, required=True)
    parser.add_argument("--m_step", type=int, default=32)
    parser.add_argument("--k_start", type=int, required=True)
    parser.add_argument("--k_end", type=int, required=True)
    parser.add_argument("--k_step", type=int, default=32)
    parser.add_argument("--use_bias", action="store_true",
                        help="Include bias add (C = A@B + bias)")

    # Partition previously unfinished work instead of the whole range, pass in a CSV file
    parser.add_argument("--partition_unfinished", type=str, default="",
                        help="Partition only the unfinished work instead of the whole range")

    args = parser.parse_args()

    if args.m_step <= 0 or args.k_step <= 0:
        raise ValueError("m_step and k_step must be > 0")
    if args.m_end < args.m_start or args.k_end < args.k_start:
        raise ValueError("end must be >= start")

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))

    os.environ["NEURON_RT_VISIBLE_CORES"] = str(rank)

    dist.init_process_group(backend="xla")

    dtype = torch.float16
    device = "xla"
    warmup_iters = 2
    measure_iters = 10

    if args.partition_unfinished:
        partition_unfinished(
            args=args,
            world_size=world_size,
            rank=rank,
            dtype=dtype,
            device=device,
            warmup_iters=warmup_iters,
            measure_iters=measure_iters,
        )
    else:
        partition_by_m(
            args=args,
            world_size=world_size,
            rank=rank,
            dtype=dtype,
            device=device,
            warmup_iters=warmup_iters,
            measure_iters=measure_iters,
        )


if __name__ == "__main__":
    main()
