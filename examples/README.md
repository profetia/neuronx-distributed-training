# examples/ — Mixtral on Trainium (PPoPP research workspace)

Research experiments for training Mixtral-style MoE models on AWS Trainium,
built on top of [aws-neuron/neuronx-distributed-training](https://github.com/aws-neuron/neuronx-distributed-training).

Covers parallelization mapping (TP/PP splits, "TAP" variants), expert
specialization (`_es`), dropless MoE kernels, and NeuronCore rank placement
solved with Gurobi ILP. Paper figures live here as PDFs and notebooks.

## Layout

| Path | What's inside |
|---|---|
| `conf/` | One YAML per experiment (`hf_mixtral_{small,medium,large,super}`, `_TAP`, `_dense`, `_es`, `_dropless`, weak-scaling variants) |
| `train.sh`, `train_setup.sh` | Launcher scripts (SLURM / EKS-MPI / single-node) |
| `training_orchestrator.py` | Entry point: loads Hydra config, sets env vars, starts training |
| `benchmarks/cc` | Collective (allreduce) microbenchmarks |
| `benchmarks/ops` | GEMM & fused-op microbenchmarks + heatmap notebooks |
| `benchmarks/mem` | Memory (constant zeros) measurements |
| `mapping_solver.ipynb` | Gurobi ILP solver: NeuronCore → (TP, DP, PP) placement |
| `paper_charts.ipynb/.py`, `tensorboard_avg.ipynb` | Paper chart generation |
| `checkpoint_converter_scripts/` | HF ↔ NxDT ↔ NNM checkpoint converters |
| `sft_evaluation/` | SFT evaluation harness |
| `nemo_experiments/`, `df_storage/`, `logs/` | Run artifacts (TB logs, checkpoints, per-node logs) |
| `slurm-*.out`, `old-runs/` | Slurm job outputs |

## Run a training job

Jobs are submitted with `sbatch` on the ParallelCluster (Slurm, up to
8 × trn1 EFA nodes, FSx at `/fsx`). Env vars set on the submit line are
passed through to `train.sh` inside the job.

Compile-only smoke test (1 node — compiles all graphs, then exits):

```bash
TRAIN_ITERS=15  NEURON_CC_FLAGS="--retry_failed_compilation --num-parallel-jobs=32" COMPILE=1 CONF_FILE=hf_mixtral_medium_config_TAP \
    sbatch --exclusive --nodes 1 --cpus-per-task 128 --wrap="srun ./train.sh"
```

Real training run (node count must match the config's expectations):

```bash
TRAIN_ITERS=15 COMPILE=0 CONF_FILE=hf_mixtral_medium_config_TAP \
    sbatch --exclusive --nodes 8 --cpus-per-task 128 --wrap="srun ./train.sh"
```

| Env var | Meaning |
|---|---|
| `CONF_FILE` | Experiment config from `conf/` (name without `.yaml`) |
| `COMPILE=1` | `neuron_parallel_compile` only: builds graphs and exits, no training |
| `TRAIN_ITERS` | Cap `max_steps` for quick real runs (applies when `COMPILE=0`; ignored in compile mode) |
| `NEURON_CC_FLAGS` | Extra neuron compiler flags (appended to existing ones) |
| `NEURON_COMPILE_CACHE_URL` | Shared compile cache on FSx (e.g. `/fsx/linshuy2/neuron_cache`) — skips recompiling unchanged graphs |

Output goes to `slurm-<jobid>.out` (job stdout) and per-node logs under
`logs/<jobid>/<restart>/<nodeid>/`.

## Microbenchmarks

```bash
cd benchmarks/ops && bash run.sh      # GEMM sweeps on 32 NeuronCores
cd benchmarks/cc                      # allreduce intranode
```

## Environment

- Trainium instances (trn1/trn2) with the Neuron SDK — see the repo
  root `README.md` for install steps.
- Python 3.10 venv with all training deps already installed (on FSx, so
  every node uses the same one):
  `source /fsx/linshuy2/aws_neuron_venv_nxd_training/bin/activate`
- Training data: `/fsx/linshuy2/examples_datasets/wikicorpus_llama2_tokenized_4k`
- Checkpoints are saved/loaded with Neuron serialization (`save_xser` /
  `load_xser`); convert with `checkpoint_converter_scripts/` for HF/NNM.
