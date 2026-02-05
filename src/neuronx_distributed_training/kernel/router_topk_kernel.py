
import math
import neuronxcc.nki as nki
import neuronxcc.nki.isa as nisa
import neuronxcc.nki.typing as nt
import neuronxcc.nki.language as nl
from neuronxcc.nki._pre_prod_kernels.topk.topk_core import topk_core

@nki.jit
def rmsnorm_router_top_k_kernel(
    hidden_states: nt.tensor, 
    rmsnorm_weight: nt.tensor,
    router_weight: nt.tensor, 
    top_k: int, 
    router_bias: nt.tensor, 
    act_fn: str = "sigmoid", 
    topk_first: bool = False,
    eps = 1e-6,
    compute_dtype = nl.float32, 
):
    """
    NKI kernel for RMSNorm + Router Top-K computation.
    
    This kernel performs:
    - RMSNorm normalization
    - Router logits computation
    - Top-K expert selection
    - Expert affinities computation (sigmoid/softmax)
    
    Note: This kernel does NOT include mask generation or residual add operations.
    
    NOTE: Residual add is NOT done in this kernel. For Mixtral architecture,
    the residual connection happens AFTER MoE (in Python code), not before RMSNorm.
    This matches PyTorch flow:
        residual = hidden_states
        hidden_states = RMSNorm(hidden_states)
        hidden_states = MoE(hidden_states)
        hidden_states = residual + hidden_states  # ← Post-MoE residual
    
    Errors encountered:
    1. BIR037: Single instruction input tensors <= 196608 bytes
       - tensor_tensor(fp32, fp32): 2 * 128 * TILE_T * 4 <= 196608, where TILE_T <= 192
       - so set TILE_T = 128 to ensure divisibility
    
    2. NKI Scoping Rules:
       - Variables rebound inside loop (x = f(x)) CANNOT be accessed outside
       - PSUM with += operator IS supported for loop accumulation

    """
    T, H = hidden_states.shape
    E, H_R = router_weight.shape
    AFF_DIM = top_k if topk_first else E
    
    assert H == H_R, "Hidden dimension mismatch between input and router weights."
    assert E <= 512, "Router Kernel does not support E > 512."
    
    # Output tensors in HBM
    router_logits = nl.ndarray((T, E), dtype=compute_dtype, buffer=nl.shared_hbm)
    expert_affinities = nl.ndarray((T, AFF_DIM), dtype=hidden_states.dtype, buffer=nl.shared_hbm)
    expert_index = nl.ndarray((T, top_k), dtype=nl.int32, buffer=nl.shared_hbm)
    output_hidden_states = nl.ndarray((T, H), dtype=compute_dtype, buffer=nl.shared_hbm)

    # Sharding
    num_shards = nl.num_programs(0)
    shard_id = nl.program_id(0)
    assert T % num_shards == 0, f"T = {T} dimension is not divisible by number of shards."
    T_PER_SHARD = T // num_shards
    T_OFFSET = T_PER_SHARD * shard_id
    
    # Tile sizes - STABLE
    # TILE_T=128: power of 2 ensures divisibility with common sequence lengths
    # Also satisfies BIR037: 2 * 128 * 128 * 4 = 131072 < 196608 bytes, where TILE_T <= 192
    PSUM_FREE_DIM_LIMIT = 512
    TILE_H = nl.tile_size.pmax
    TILE_T = min(128, T_PER_SHARD, PSUM_FREE_DIM_LIMIT)
    
    H_ITERS = math.ceil(H / TILE_H)
    T_SHARD_ITERS = T_PER_SHARD // TILE_T 

    assert T_PER_SHARD % TILE_T == 0, f"T_PER_SHARD ({T_PER_SHARD}) must be divisible by TILE_T ({TILE_T})"

    # Load constant tensors
    weight_tile = nl.ndarray((nl.par_dim(TILE_H), H_ITERS, E), dtype=compute_dtype, buffer=nl.sbuf)
    for h in nl.affine_range(H_ITERS):
        weight_tile[:, h, :] = nl.load_transpose2d(router_weight[:, h*TILE_H:(h+1)*TILE_H])
    
    rmsnorm_tile = nl.ndarray((nl.par_dim(TILE_H), H_ITERS), dtype=compute_dtype, buffer=nl.sbuf)
    for h in nl.affine_range(H_ITERS):
        rmsnorm_tile[:, h] = nl.load(rmsnorm_weight[h*TILE_H:(h+1)*TILE_H])
    
    if router_bias is not None:
        sbuf_bias = nl.load(router_bias)

    eps_bias = nisa.memset((TILE_H, 1), value=eps, dtype=compute_dtype)
    reduction_vector = nisa.memset((TILE_H, TILE_H), value=1.0, dtype=nl.float32)

    # Go through each shard by TILE_T
    for t in nl.affine_range(T_SHARD_ITERS):
        t_start = t * TILE_T
        t_end = (t + 1) * TILE_T
        
        # 1. Compute sum of squares for RMSNorm
        # Use PSUM for accumulation (required by NKI scoping rules)
        
        # Accumulator in PSUM (fp32) - supports += which is valid for loop accumulation
        # NKI Rule: Variables rebound in loop (x = f(x)) CANNOT be used outside loop
        # NKI Rule: PSUM with += IS valid for accumulation across loop iterations
        hidden_squared_sum = nl.zeros((nl.par_dim(TILE_H), TILE_T), dtype=nl.float32, buffer=nl.psum)
        
        for h in nl.affine_range(H_ITERS):
            # Load tiles - each is (par_dim(128), TILE_T) 
            hidden_tile = nl.ndarray((nl.par_dim(TILE_H), TILE_T), dtype=hidden_states.dtype, buffer=nl.sbuf)
            hidden_tile[:, :] = nl.load_transpose2d(
                hidden_states[T_OFFSET + t_start:T_OFFSET + t_end, h*TILE_H:(h+1)*TILE_H]
            )
            
            # NOTE: Residual add removed. For Mixtral, residual is added AFTER MoE in Python code
            
            # Square and accumulate using += (valid for PSUM)
            hidden_squared = nisa.activation(op=nl.square, data=hidden_tile)
            hidden_squared_sum += hidden_squared
        
        # Copy from PSUM to SBUF (nc_matmul requires both operands in SBUF)
        hidden_squared_sum_sbuf = nl.ndarray((nl.par_dim(TILE_H), TILE_T), dtype=nl.float32, buffer=nl.sbuf)
        hidden_squared_sum_sbuf[:, :] = nl.copy(hidden_squared_sum)
        
        # Reduce across par_dim and compute rms_scale
        # nc_matmul with all-1s reduction_vector replicates column sums to all rows
        hidden_squared_total = nl.ndarray((nl.par_dim(TILE_H), TILE_T), dtype=compute_dtype, buffer=nl.psum)
        hidden_squared_total[:, :] = nisa.nc_matmul(reduction_vector, hidden_squared_sum_sbuf)
        
        rms_scale = nisa.activation(op=nl.rsqrt, data=hidden_squared_total, scale=(1.0/H), bias=eps_bias)
        
        # 2. Apply RMSNorm and compute router logits
        psum_logits = nl.zeros((TILE_T, E), nl.float32, buffer=nl.psum)
        
        for h in nl.affine_range(H_ITERS):
            # Reload hidden_states for RMSNorm application
            hidden_tile = nl.ndarray((nl.par_dim(TILE_H), TILE_T), dtype=hidden_states.dtype, buffer=nl.sbuf)
            hidden_tile[:, :] = nl.load_transpose2d(
                hidden_states[T_OFFSET + t_start:T_OFFSET + t_end, h*TILE_H:(h+1)*TILE_H]
            )
            
            # Apply RMSNorm
            hidden_weighted = nl.ndarray((nl.par_dim(TILE_H), TILE_T), dtype=compute_dtype, buffer=nl.sbuf)
            hidden_weighted[:, :] = nisa.tensor_tensor(hidden_tile, rmsnorm_tile[:, h:h+1], nl.multiply)
            hidden_weighted[:, :] = nisa.tensor_tensor(hidden_weighted, rms_scale, nl.multiply)
            
            # Store normalized output
            nl.store(
                output_hidden_states[T_OFFSET + t_start:T_OFFSET + t_end, h*TILE_H:(h+1)*TILE_H],
                value=nisa.nc_transpose(hidden_weighted)
            )
            
            # Accumulate router logits
            psum_logits += nisa.nc_matmul(hidden_weighted, weight_tile[:, h, :])
        
        # 3. Top-K and output
        sbuf_logits = nl.ndarray((TILE_T, E), dtype=router_logits.dtype, buffer=nl.sbuf)
        if router_bias is not None:
            sbuf_logits[:, :] = nl.add(psum_logits, sbuf_bias)
        else:
            sbuf_logits[:, :] = nisa.activation(op=nl.copy, data=psum_logits, dtype=router_logits.dtype)
        
        nl.store(router_logits[T_OFFSET + t_start:T_OFFSET + t_end, :], value=sbuf_logits)

        sbuf_expert_values, sbuf_expert_index = topk_core(sbuf_logits, top_k)
        nl.store(expert_index[T_OFFSET + t_start:T_OFFSET + t_end, :], value=sbuf_expert_index)

        if topk_first:
            if act_fn == "sigmoid":
                sbuf_affinities = nisa.activation(op=nl.sigmoid, data=sbuf_expert_values)
            else:
                sbuf_affinities = nl.softmax(sbuf_expert_values, axis=1)                
        else:
            if act_fn == "sigmoid":
                sbuf_affinities = nisa.activation(op=nl.sigmoid, data=sbuf_logits)
            else:
                sbuf_affinities = nl.softmax(sbuf_logits, axis=1)

        nl.store(expert_affinities[T_OFFSET + t_start:T_OFFSET + t_end, :], value=sbuf_affinities)

    return output_hidden_states, router_logits, expert_affinities, expert_index
