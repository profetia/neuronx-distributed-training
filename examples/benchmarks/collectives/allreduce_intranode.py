import argparse
import os
import tabulate
import torch
import torch.distributed as dist
import torch.utils.benchmark as benchmark
import torch_xla
import torch_xla.core.xla_model as xm
import neuronxcc.nki as nki
import neuronxcc.nki.isa as nisa
import neuronxcc.nki.language as nl
import neuronxcc.nki.typing as nt
import neuronxcc.nki.nccl as nccl

world_size = int(os.environ.get('WORLD_SIZE'))
local_rank = int(os.environ.get('LOCAL_RANK'))

neuron_remap = eval(os.environ.get('REMAP_CORES', "[i for i in range(world_size)]"))
os.environ['NEURON_RT_VISIBLE_CORES'] = str(neuron_remap[local_rank])

dist.init_process_group(backend='xla')
allreduce_group_spmd = eval(os.environ.get('SPMD_GROUP', "[[i for i in range(world_size)]]"))

def sanity_check():
    tensor = torch.tensor([local_rank], device=torch_xla.device())
    xm.all_reduce('sum', [tensor], groups=allreduce_group_spmd)

    def sanity_check_inner():
        for group in allreduce_group_spmd:
            if local_rank not in group:
                continue

            expected = sum(group)
            got = tensor.item()
            assert got == expected, f"Sanity check failed on rank {local_rank}: expected {expected}, got {got}"

        if local_rank == 0:
            print("Sanity check passed")

    xm.add_step_closure(sanity_check_inner)
    torch_xla.sync()

def allreduce_intra_chip_use_xm(tensor: torch.Tensor):
    xm.all_reduce('sum', [tensor], groups=allreduce_group_spmd)
    torch_xla.sync(wait=True)

source_map = {
    'xm': {
        'func': allreduce_intra_chip_use_xm,
        'name': 'xm.all_reduce'
    },
}

def run_benchmark(allreduce_func, size: int, dtype: torch.dtype, iters: int, warmup_iters: int):
    group_world_size = len(allreduce_group_spmd[0])
    # https://forums.developer.nvidia.com/t/what-is-the-busbw-in-nccl-tests/256858
    num_elements = size // torch.tensor([], dtype=dtype).element_size() // group_world_size
    tensor = torch.empty(num_elements, dtype=dtype, device=torch_xla.device())
    for _ in range(warmup_iters):
        allreduce_func(tensor)
    xm.rendezvous('allreduce_warmup_complete')

    timer = benchmark.Timer(
        stmt='allreduce_func(tensor)',
        globals={'allreduce_func': allreduce_func, 'tensor': tensor},
        num_threads=1,
    )
    result = timer.timeit(iters)
    xm.rendezvous('allreduce_benchmark_complete')
    return result

class ReadableSize(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        size_str = values.lower()
        if size_str.endswith('kb'):
            size = int(size_str[:-2]) * 1024
        elif size_str.endswith('mb'):
            size = int(size_str[:-2]) * 1024 * 1024
        elif size_str.endswith('gb'):
            size = int(size_str[:-2]) * 1024 * 1024 * 1024
        else:
            size = int(size_str)
        setattr(namespace, self.dest, size)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--minbytes', action=ReadableSize, default=1024, help='Minimum input size to start from')
    parser.add_argument('--maxbytes', action=ReadableSize, default=1024**2, help='Maximum input size to end at')
    parser.add_argument('--stepfactor', type=int, default=2, help='Multiplication factor between sizes')
    parser.add_argument('--iters', type=int, default=200, help='Number of iterations for benchmarking')
    parser.add_argument('--warmup_iters', type=int, default=20, help='Number of warmup iterations')
    parser.add_argument('--source', choices=source_map.keys(), default='xm',
                        help='Source of allreduce implementation to benchmark')
    return parser.parse_args()

def main(args: argparse.Namespace):
    min_bytes = args.minbytes
    max_bytes = args.maxbytes
    step_factor = args.stepfactor
    iters = args.iters
    warmup_iters = args.warmup_iters
    dtype = torch.bfloat16

    sanity_check()

    volumes = []
    size = min_bytes
    while size <= max_bytes:
        volumes.append(size)
        size *= step_factor

    group_world_size = len(allreduce_group_spmd[0])

    results = []
    for volume in volumes:
        xm.rendezvous(f'start_benchmark_volume_{volume}')

        result = run_benchmark(source_map[args.source]['func'], volume, dtype, iters, warmup_iters)
        time_avg = result.mean * 1e6
        algbw = volume / result.mean / (1024 ** 3)
        busbw = algbw * 2 * (group_world_size - 1) / group_world_size * len(allreduce_group_spmd)

        results.append({
            'size(B)': volume,
            'type': dtype.__str__(),
            'time:avg(us)': time_avg,
            'algbw(GB/s)': algbw,
            'busbw(GB/s)': busbw,
        })

        if local_rank == 0:
            print(results[-1])

    if local_rank == 0:
        print(f"AllReduce Intra-Chip Benchmark using {source_map[args.source]['name']}")
        print(f"Args: minbytes={min_bytes}, maxbytes={max_bytes}, stepfactor={step_factor}, iters={iters}, warmup_iters={warmup_iters}")
        print(f"Parallelism: world_size={world_size}, remap={neuron_remap}, allreduce_groups={allreduce_group_spmd}")
        print(tabulate.tabulate(results, headers="keys", tablefmt="grid"))

if __name__ == '__main__':
    args = parse_args()
    main(args)