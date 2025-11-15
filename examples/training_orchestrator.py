"""Load config, set environment variables, start training."""

import hydra
import os
from transformers.utils import logging
import json
from typing import Union

logger = logging.get_logger(__name__)

def set_env_variable(env_var_name, value, append=False, overwrite=False):
    if append and os.environ.get(env_var_name, None):
        os.environ[env_var_name] = f"{os.environ[env_var_name]} {value}"
        return
    if overwrite:
        os.environ[env_var_name] = value
    elif os.environ.get(env_var_name, None) is None:
        os.environ[env_var_name] = value

def get_dict_from_json(json_file: Union[str, os.PathLike]):
    with open(json_file, "r", encoding="utf-8") as opened_file:
        dict_text = json.loads(opened_file.read())
    return dict_text
    
def process_config(cfg):
    """Map default values and set environment variables from config.

    This function may modify the config variable.
    """
    # map NeMo default to NxDT default
    vpmps = cfg.distributed_strategy.get("virtual_pipeline_model_parallel_size", 1)
    if vpmps is None:
        cfg.distributed_strategy.virtual_pipeline_model_parallel_size = 1
    # set environment variables derived from config
    if cfg.model.fusions.get("softmax", None):
        set_env_variable("NEURON_FUSE_SOFTMAX", "1")
    if cfg.neuron_experimental_compress_rg is True:
        set_env_variable("NEURON_EXPERIMENTAL_COMPRESS_RG", "1")
    else:
        set_env_variable("NEURON_EXPERIMENTAL_COMPRESS_RG", "0")
    set_env_variable("NEURON_RT_ASYNC_EXEC_MAX_INFLIGHT_REQUESTS", str(cfg.aync_exec_max_inflight_requests))
    set_env_variable("BUCKET_CAP_MB", str(cfg.bucket_size_collectives))
    set_env_variable("NEURON_COMPILE_CACHE_URL", cfg.compiler_cache_url)
    set_env_variable("NEURON_CC_FLAGS", cfg.compiler_flags, append=True)
    set_env_variable("NEURON_RT_EXEC_TIMEOUT", str(cfg.neuron_rt_exec_timeout))

    # if doing compile then change the some config vars
    # elif training and have TRAIN_ITERS set (used in tests) set max_steps as TRAIN_ITERS,
    # rest remains as in '.yaml' file
    # else keep everything the same as in '.yaml' file
    # NOTE - Specifying TRAIN_ITERS as env variable is used for test purposes only,
    # recommended approach is to update variable in yaml cfg/pass it to script
    if os.environ.get("COMPILE") == "1":
        cfg.trainer.max_steps = 10
        cfg.exp_manager.create_tensorboard_logger = False
        cfg.exp_manager.create_checkpoint_callback = False
    elif os.environ.get("COMPILE") == "0" and os.environ.get("TRAIN_ITERS") is not None:
        cfg.trainer.max_steps = int(os.environ.get("TRAIN_ITERS"))

	# MOE dropless
    if "moe" in cfg.model:
        dropless = cfg.model.moe.get("dropless", False)
        capacity_factor = cfg.model.moe.capacity_factor
		# Default to True; glu_mlp is not in config file but initialized as True in model setting
        glu_mlp = cfg.model.moe.get("glu_mlp", True)

        if cfg.model_source == "hf":
            model_config = cfg.model.model_config
            try:
                config_dict = get_dict_from_json(model_config)
            except (json.JSONDecodeError, UnicodeDecodeError) as e:
                raise type(e)(f"Error: {str(e)}: Unable to parse the config file at '{model_config}.")

        if dropless:
			# Check for LLaMA-like architecture requirements
            if cfg.model_source == "hf" and config_dict.get("hidden_act", None) != "silu":
                current_activation = config_dict.get("hidden_act", None)
                raise ValueError(
					"Error: Dropless mode is only supported with SiLU activation function. "
					f"Current activation function: {current_activation}. "
					"Please adjust your configuration."
				)
            elif cfg.model_source == "megatron":
                activation = getattr(cfg.model, "activation", None)
                if not (activation == "silu" or activation == "swiglu"):
                    raise ValueError(
						"Error: For Megatron models, dropless mode is only supported with SiLU or SwiGLU activation functions. "
						f"Current activation function: {activation}. "
						"Please adjust your configuration."
					)
            if not glu_mlp:
                raise ValueError("Error: Dropless mode requires GLU_MLP to be True.")
            if capacity_factor > 0.0:
                logger.warning(
					"Dropless mode works with capacity_factor set to 0.0. "
					f"Current value: {capacity_factor}. Setting capacity_factor to 0.0."
				)
                cfg.model.moe.capacity_factor = 0.0
        elif not dropless and capacity_factor <= 0.0:
            raise ValueError(
				"Error: Dropping requires a capacity factor greater than 0.0 Please adjust your configuration."
			)
    # precision setting
    if cfg.precision.get("type") == "bf16SR":
        set_env_variable("XLA_USE_BF16", "1")
        set_env_variable("XLA_DOWNCAST_BF16", "0")
        set_env_variable("NEURON_RT_STOCHASTIC_ROUNDING_EN", "1")
        set_env_variable("NEURON_CC_FLAGS", "--enable-mixed-precision-accumulation", append=True)
    elif cfg.precision.get("type") == "mixed_precision":
        set_env_variable("XLA_USE_BF16", "0")
        set_env_variable("XLA_DOWNCAST_BF16", "1")
        set_env_variable("NEURON_RT_STOCHASTIC_ROUNDING_EN", "0")
        set_env_variable("NEURON_CC_FLAGS", "--enable-mixed-precision-accumulation", append=True)
    elif cfg.precision.get("type") == "mixed_precision_SR":
        set_env_variable("XLA_USE_BF16", "0")
        set_env_variable("XLA_DOWNCAST_BF16", "1")
        set_env_variable("NEURON_RT_STOCHASTIC_ROUNDING_EN", "1")
        set_env_variable("NEURON_CC_FLAGS", "--enable-mixed-precision-accumulation", append=True)
    elif cfg.precision.get("type") == "fp32":
        set_env_variable("XLA_USE_BF16", "0")
        set_env_variable("XLA_DOWNCAST_BF16", "0")
        set_env_variable("NEURON_CC_FLAGS", "--auto-cast none", append=True)
    elif cfg.precision.get("type") == "manual":
        set_env_variable("XLA_USE_BF16", cfg.precision.get("xla_use_bf16"))
        set_env_variable("XLA_DOWNCAST_BF16", cfg.precision.get("xla_downcast_bf16"))
        set_env_variable("NEURON_RT_STOCHASTIC_ROUNDING_EN", cfg.precision.get("neuron_rt_stochastic_rounding_en"))
    elif cfg.precision.get("type") == "autocast":
        set_env_variable("XLA_USE_BF16", "0")
        set_env_variable("XLA_DOWNCAST_BF16", "0")
        set_env_variable("NEURON_RT_STOCHASTIC_ROUNDING_EN", "0")
        set_env_variable("NEURON_CC_FLAGS", "--enable-mixed-precision-accumulation --auto-cast none", append=True)
    else:
        raise ValueError(
            "Invalid option given for precision type. Must be one of "
            + "bf16SR, fp32, autocast, mixed_precision, mixed_precisionSR, or manual."
        )
    return cfg

def remap_cores(cfg):
    tensor_model_parallel_size = cfg.distributed_strategy.tensor_model_parallel_size
    pipeline_model_parallel_size = cfg.distributed_strategy.pipeline_model_parallel_size
    expert_model_parallel_size = cfg.distributed_strategy.get("expert_model_parallel_size", 1)
    if expert_model_parallel_size > 1:
        # TODO: Add support for expert parallelism
        logger.warning("Expert model parallelism is not yet supported for core remapping.")
        return
    
    data_parallel_size = cfg.trainer.devices * cfg.trainer.num_nodes // (tensor_model_parallel_size * pipeline_model_parallel_size * expert_model_parallel_size)
    
    if tensor_model_parallel_size > cfg.trainer.devices:
        logger.warning("Tensor model parallel size greater than number of devices. Skipping core remapping.")
        return
    
    tensor_model_parallel_size_per_node = tensor_model_parallel_size
    pipeline_model_parallel_size_per_node = max(
        min(pipeline_model_parallel_size, cfg.trainer.devices // tensor_model_parallel_size_per_node // data_parallel_size), 1)

    cores_list = None # TP=32 PP=1 or TP=8 PP=4 or TP=8 PP=1 or TP=2 PP=4 
    match (tensor_model_parallel_size_per_node, pipeline_model_parallel_size_per_node):
        case (16, 2) | (16, 1): # TP=16 PP=2 or TP=16 PP=1
            cores_list = [0,1,2,3,4,5,6,7,14,15,12,13,10,11,8,9,24,25,26,27,28,29,30,31,22,23,20,21,18,19,16,17]
        case (8, 2): # TP=8 PP=2
            cores_list = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,24,25,26,27,28,29,30,31,16,17,18,19,20,21,22,23]
        case (4, 8) | (2, 8) | (1, 8): # TP=4 PP=8 or TP=2 PP=8
            cores_list = [0,1,2,3,6,7,4,5,14,15,12,13,8,9,10,11,16,17,18,19,22,23,20,21,30,31,28,29,24,25,26,27]
        case (4, 4): # TP=4 PP=4
            cores_list = [0,1,2,3,6,7,4,5,8,9,10,11,14,15,12,13,16,17,18,19,22,23,20,21,24,25,26,27,30,31,28,29]
        case (4, 2): # TP=4 PP=2
            cores_list = [0,1,2,3,6,7,4,5,8,9,10,11,14,15,12,13,24,25,26,27,30,31,28,29,16,17,18,19,22,23,20,21]
        case (2, 16): # TP=2 PP=16
            cores_list = [0,24,1,25,2,26,3,27,4,28,5,29,6,30,7,31,15,23,14,22,13,21,12,20,11,19,10,18,9,17,8,16]

    if cores_list is not None:
        local_rank = os.environ.get("LOCAL_RANK")
        remapped_core = cores_list[int(local_rank)]
        os.environ["NEURON_RT_VISIBLE_CORES"] = str(remapped_core)
        logger.warning(f"Remapping NEURON_RT_VISIBLE_CORES {cores_list} to " +
                    f"NEURON_RT_VISIBLE_CORES[{local_rank}]={remapped_core}")



@hydra.main(config_path="conf", config_name="megatron_gpt_config", version_base="1.2")
def main(cfg) -> None:
    cfg = process_config(cfg)

    # Patch AWS NeuronX toolchain 
    from torch_xla.distributed.zero_redundancy_optimizer import ZeroRedundancyOptimizer

    __register_hook_original = ZeroRedundancyOptimizer._register_hook
    def __register_hook_patched(self, param, shard):
        if not param.requires_grad:
            return
        return __register_hook_original(self, param, shard)

    ZeroRedundancyOptimizer._register_hook = __register_hook_patched

    remap_cores(cfg)

    from training import train

    train(cfg)


if __name__ == "__main__":
    main()