# CLAUDE.md — examples/ research workspace

This directory is the **PPoPP paper research workspace** for training Mixtral MoE
models on AWS Trainium, on top of the `aws-neuron/neuronx-distributed-training`
(NxDT) library. You are inside the `examples/` subdirectory; the library itself
is `../src/neuronx_distributed_training/` and `../Megatron-LM/` is vendored.

Current branch: `linshuy2/mixtral`. Main author Linshu Yang; Jieyi Zhao
(`jieyi3@illinois.edu`) wrote the NKI router top-k kernel. Recent work (git log):
paper charts, kernel fusion, expert-specialized configs, NeuronCore remap,
mapping solver, GEMM heatmaps.

## How training runs are launched

```
train.sh → source train_setup.sh → torchrun <DISTRIBUTED_ARGS> training_orchestrator.py \
    --config-path=conf --config-name=$CONF_FILE trainer.devices=$PPN trainer.num_nodes=$NTASKS \
    exp_manager.explicit_log_dir=$EXPLICIT_LOGDIR | tee -a $LOG_PATH/log
```

**Canonical submission** is via `sbatch` — env vars on the submit line are
inherited by the job script (sbatch exports the submit environment):

```bash
# Compile-only smoke test (1 node)
TRAIN_ITERS=15 NEURON_CC_FLAGS="--retry_failed_compilation --num-parallel-jobs=32" COMPILE=1 \
CONF_FILE=hf_mixtral_medium_config_TAP \
    sbatch --exclusive --nodes 1 --cpus-per-task 128 --wrap="srun ./train.sh"
# Real run (8 nodes)
TRAIN_ITERS=15 COMPILE=0 CONF_FILE=hf_mixtral_medium_config_TAP \
    sbatch --exclusive --nodes 8 --cpus-per-task 128 --wrap="srun ./train.sh"
```

Job stdout → `slurm-<jobid>.out` in this dir; per-node logs →
`logs/<jobid>/<restart>/<nodeid>/`; Hydra/PTL artifacts → `nemo_experiments/
<hf_mixtral|hf_llama>/<jobid>/` and `outputs/<date>/`.

- `CONF_FILE=<name>` selects `conf/<name>.yaml` (default `megatron_gpt_config`);
  `devices:` is read from the YAML to set `--nproc_per_node`.
- `COMPILE=1` prefixes the command with `neuron_parallel_compile` — compile-only
  pass (trial run capped at `max_steps=10`, tensorboard/checkpoints disabled), then exits.
- `TRAIN_ITERS` overrides `max_steps` for quick real runs, but **only when
  `COMPILE=0`** (code: `elif COMPILE == "0" and TRAIN_ITERS` — in compile mode
  it is silently ignored).
- `train_setup.sh` dispatches on environment:
  - `SLURM_NNODES` set → Slurm/PCluster run: EFA env (`FI_PROVIDER=efa`,
    `FI_EFA_USE_DEVICE_RDMA=1`, `FI_EFA_FORK_SAFE=1`), `sudo sysctl` reserves
    `ip_local_reserved_ports=41000`, `lctl` caps FSx dirty MB to avoid OODs.
  - `OMPI_COMM_WORLD_RANK` set → EKS/MPI run, logs to `/shared/nemo_experiments/<POD_UID>/<rank>/`.
  - Otherwise single-node: `torchrun` on localhost, logs to `nemo_experiments/logs`.
- Fixed env from `train_setup.sh`: `XLA_DISABLE_FUNCTIONALIZATION=0` (needed for
  ZeRO1 convergence parity in PT2.1 — do not remove), `MASTER_PORT=41000`,
  `MALLOC_ARENA_MAX=128`, `HYDRA_FULL_ERROR=1`, `CREATE_TB_LOGGER=True`, `CHECKPOINT_CALLBACK=True`.
- `training_orchestrator.py` `process_config()` maps config → env vars:
  `NEURON_FUSE_SOFTMAX`, `NEURON_EXPERIMENTAL_COMPRESS_RG`, `BUCKET_CAP_MB`,
  `NEURON_COMPILE_CACHE_URL`, `NEURON_CC_FLAGS` (appended, never overwritten),
  `NEURON_RT_EXEC_TIMEOUT`, `NEURON_RT_ASYNC_EXEC_MAX_INFLIGHT_REQUESTS`.

## Config conventions (`conf/`)

Files are the experiment matrix; each YAML is self-contained (no Hydra groups).
Naming grammar: `hf_mixtral_{small,medium,large,super}_config[_TAP][_dense|_es][_dropless][_TAP_weak_<global_bs>]`

- `_TAP` = alternative TP/PP split (e.g. TP=2/PP=4 vs plain TP=1/PP=8 for small;
  TP=8/PP=32 for large) — a key paper ablation.
- `_dense` = dense baseline (MoE swapped for dense FFN), `_es` = expert-specialized models.
- `_dropless` = blockwise NKI kernel (Megablocks-inspired).
- `hf_llama3*_trn2` configs are the trn2 variants; `megatron_*` configs run the
  Megatron-LM path instead of HF-LLaMA/Mixtral modules.
- Parallelism knobs: `distributed_strategy.{tensor_model_parallel_size,
  pipeline_model_parallel_size, virtual_pipeline_model_parallel_size, zero1,
  sequence_parallel, expert_model_parallel_size}`; MoE knobs under `model.moe`
  (`num_experts`, `capacity_factor`, `dropless`, `top_k`, `router_aux_loss_coef`).
- `fusions:` flags — `softmax` (→ `NEURON_FUSE_SOFTMAX`), `flash_attention`
  (NKI flash attention), `nki_router_topk` (fused NKI top-k router kernel).

**Precision modes** — `precision.type` in the YAML maps to env vars (orchestrator
lines ~105-130): `bf16SR` (XLA_USE_BF16=1 + stochastic rounding), `mixed_precision`
(DOWNCAST_BF16=1), `mixed_precision_SR`, `fp32` (`--auto-cast none` appended to
NEURON_CC_FLAGS), `manual`, `autocast`. All non-fp32 modes append
`--enable-mixed-precision-accumulation` to NEURON_CC_FLAGS.

**MoE validation rules** (enforced in orchestrator): `dropless: True` forces
`capacity_factor: 0.0` and requires `hidden_act: silu` in the model JSON;
dropping (dropless False) requires `capacity_factor > 0`.

## Neuron core remapping (the `NEURON_RT_VISIBLE_CORES` mechanism)

`remap_cores()` in `training_orchestrator.py:139` rewrites the rank→core mapping
per run. A `match` statement over `(devices, TP, PP)` returns a hardcoded
`cores_list` permutation (comment: "TP=32 PP=1 or TP=8 PP=4 or TP=8 PP=1 or
TP=2 PP=4") which is assigned rank-wise to `NEURON_RT_VISIBLE_CORES`. These
orderings come from `mapping_solver.ipynb` (Gurobi ILP). Limits: skipped when
`expert_model_parallel_size > 1` (unsupported, TODO) or `TP > devices`.
Benchmarks read the same idea via env directly: `allreduce_intranode.py` takes
`REMAP_CORES` (a Python-list string, eval'd) plus `SPMD_GROUP`.

## Kernels, fusions & toolchain patches

- NKI flash attention: `neuronx_distributed.kernels.flash_attn` (used by
  `modeling_mixtral.py`, `transpose_nki_inputs=False`).
- NKI router top-k: `src/neuronx_distributed_training/kernel/router_topk_kernel.py`
  (183 lines, by Jieyi), enabled via `fusions.nki_router_topk` + `_NKI_KERNEL_IMPORTABLE`
  guard in `models/hf_models/modeling_mixtral.py:501`.
- Dropless blockwise kernel lives in `neuronx_distributed` (not this repo).
- Monkey-patches (three places — all load-bearing):
  1. `training_orchestrator.py` main(): patches `torch_xla`'s
     `ZeroRedundancyOptimizer._register_hook` ("Patch AWS NeuronX toolchain").
  2. `src/.../patches/import_override.py`: dummy `transformers.utils.is_torch_mlu_available`
     for a transformers-version conflict.
  3. `src/.../patches/lightning_neuron_patch.py`: patches Lightning's XLA device
     parser (`_auto_device_count_patched` returns 2 devices).
  4. `../install_setup.sh` sed-patches NeMo **inside the venv**: removes
     `get_megatron_pretrained_bert_models()` (Transformer Engine dependency) and
     the checkpoint filepath-existence check (S3 checkpointing). Reinstalling
     `nemo_toolkit` would undo these — re-run the sed or things break.

## Neuron-specific gotchas

- **Compile cache**: `NEURON_CC_FLAGS="--retry_failed_compilation --num-parallel-jobs=32"`
  is standard for training jobs (4 for benchmark sweeps). Set
  `NEURON_COMPILE_CACHE_URL=/fsx/linshuy2/neuron_cache` to share compiled graphs
  across jobs (most slurm logs use it; older jobs used `/fsx/linshuy2/tmp*`).
  Microbenchmarks `rm -rf /var/tmp/neuron-compile-cache` between runs.
  Compilation can take a long time — a "hung" job is often compiling.
- **Checkpoints** use Neuron serialization (`exp_manager.save_xser: True`,
  `load_xser: True`). They are NOT HF checkpoints.
- `exp_manager.resume_if_exists: True` + `resume_ignore_no_checkpoint: True`
  means re-running a config resumes rather than restarts.
- `exp_manager.log_neuron_top: True` launches `neuron-top` profiling into
  tensorboard (`utils/neuron_top.py` → `neuron_top_logs/<nodename>/`).
- Training runs only work on Trainium instances; don't attempt locally.

## Benchmarks

- `benchmarks/ops/`: GEMM and fused-op (gemm+gelu/layernorm/relu/softmax) sweeps.
  Pattern in `run.sh`: one `torchrun --nproc_per_node=32` per M-range slice writing
  `gemm_benchmark_k..._rankN.csv`, wrapped in a `while true` retry loop with
  `--partition_unfinished` so failed compiles resume where they stopped.
  `draw_heatmap.ipynb` renders the paper heatmaps. CSV schema has two variants
  (with/without `use_bias` column) — `old-runs/remove_headers.py` normalizes them.
- `benchmarks/cc/`: allreduce intranode (uses `REMAP_CORES` + `SPMD_GROUP` env,
  SPMD groups via XLA) + `draw_allreduce.ipynb`.
- `benchmarks/mem/`: `constant_zeros.py` memory measurements.

## Analysis / paper pipeline

Data flows: `nemo_experiments/<model>/<jobid>/` (tensorboard logs) →
`tensorboard_avg.ipynb` (tbparse SummaryReader → pandas) →
`df_storage/<jobid>Dataframe.pkl` (pickled DataFrames, one per job) →
`paper_charts.ipynb` / `paper_charts.py` → the `Evaluation*.pdf`,
`Characterization*.pdf`, `Appendix*.pdf`, `GroupingPerformance.pdf` files in this
dir (outputs, not sources).

- `mapping_solver.ipynb`: Gurobi ILP mapping NeuronCores → (TP, DP, PP). trn1.32xl
  = 4×4 device grid × 2 cores = 32 cores. Params like `TPN, DPN, PPN,
  tp_device_spread`; produces `core_map` + PNG visualizations. Its solutions are
  the `cores_list` tables hardcoded in `remap_cores()`.
- Model size helpers live next to the model JSONs: `dense_size.py` / `moe_size.py`
  in `/fsx/linshuy2/neuronx-distributed/examples/training/mixtral/mixtral_pretrain/configs/`
  (dense-vs-MoE size math for the memory-breakdown figures).

## Checkpoint conversion

`checkpoint_converter_scripts/` — three converters; CLI examples from
`hf_nxdt_mixtral_ckpt_converter.py` docstring:

```bash
# NxDT -> HF
python3 hf_nxdt_mixtral_ckpt_converter.py --model_style megatron --convert_to_full_state \
    --input_dir mixtral_checkpoints --output_dir converted_checkpoints \
    --nxdt_yaml_config conf/mixtral_8x7b_config.yaml --load_xser True \
    --tp_size 32 --pp_size 4 --ep_size 1 --n_layers 32
# HF -> NxDT
python3 hf_nxdt_mixtral_ckpt_converter.py --model_style megatron \
    --input_dir mixtral-8x7B-hf --output_dir converted_checkpoints \
    --config mixtral-8x7B-hf/config.json --save_xser True --convert_from_full_state \
    --tp_size 32 --pp_size 4 --ep_size 1 --n_layers 32
```

(`--nxdt_yaml_config` converts a YAML into the JSON schema when no config.json
exists. The other two scripts handle NNM ↔ NxDT.)

## SFT evaluation

`sft_evaluation/evaluate.py` — evaluates fine-tuned checkpoints with NxD
(`--framework="nxd"`, needs `--nxd_inference_path` and `--traced_model_path`)
or TnX (`--framework="tnx"`); HF-style weights folder, prompt/label templates
(Go templates), metrics like ROUGE, `--sequence_length=4096`.

## Data & models

- Training data: `/fsx/linshuy2/examples_datasets/wikicorpus_llama2_tokenized_4k`
  (`data.train_dir` in each YAML).
- Model JSONs: `/fsx/linshuy2/neuronx-distributed/examples/training/mixtral/mixtral_pretrain/configs/`
  — `8x1b_config.json`, `8x7b_config.json`, `{medium,large,super}.json` plus
  `{medium,large}_{dense,es}.json` variants (pointed to by `model.model_config`).

## Cluster & environment

- ParallelCluster (`us-west-2.yaml`, `us-east-2.yaml`): Slurm `compute1` queue,
  up to 8 × trn1n.32xlarge with EFA + placement group, c5.4xlarge head node,
  Neuron SDK v2.26 via `s3://neuron-s3/.../install_neuron.sh`, FSx Lustre at
  `/fsx` (restored from backup, DeletionPolicy: Retain).
- Python env: shared venv `/fsx/linshuy2/aws_neuron_venv_nxd_training/` (py3.10,
  on FSx so all nodes use it; jobs run with it pre-activated). Activate manually
  with `source /fsx/linshuy2/aws_neuron_venv_nxd_training/bin/activate`. It is
  shared state — don't casually upgrade neuron/torch packages there, and remember
  the NeMo sed patches above live inside it.

## Housekeeping rules

- Do not edit or delete `slurm-*.out` / `logs/` / `old-runs/` / `df_storage/` —
  historical evidence for the paper; ask before cleaning. `old-runs/` holds
  pre-remap outputs.
- `outputs/` is Hydra's date-organized output tree; `nemo_experiments/` is
  organized by model then slurm job id.

## Session history

A `remember` plugin captures session notes into `.remember/` (`now.md`,
`today-*.md`, `recent.md`, `core-memories.md`). If the user asks "how did we run
X before", search `.remember/` first, then `logs/` + `slurm-*.out` + git history.

## Cheat sheet

```bash
# Compile-only smoke test (1 node)
TRAIN_ITERS=15 NEURON_CC_FLAGS="--retry_failed_compilation --num-parallel-jobs=32" COMPILE=1 \
CONF_FILE=hf_mixtral_medium_config_TAP \
    sbatch --exclusive --nodes 1 --cpus-per-task 128 --wrap="srun ./train.sh"
# Real run (8 nodes; TRAIN_ITERS only applies with COMPILE=0)
TRAIN_ITERS=15 COMPILE=0 CONF_FILE=hf_mixtral_medium_config_TAP \
    sbatch --exclusive --nodes 8 --cpus-per-task 128 --wrap="srun ./train.sh"
# GEMM sweep
cd benchmarks/ops && bash run.sh
# Paper charts
python3 paper_charts.py   # or run the notebook
# Manual venv
source /fsx/linshuy2/aws_neuron_venv_nxd_training/bin/activate
```
