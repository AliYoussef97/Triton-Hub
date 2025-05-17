import triton

def filter_configs(configs, block_keys=None, include_group_size=False, group_size=1):
    filtered_configs = []
    for cfg in configs:
        new_config = {k: v for k, v in cfg.kwargs.items() if k.split('_')[-1] in block_keys}

        if include_group_size:
            new_config['GROUP_SIZE_M'] = group_size
        
        filtered_configs.append(triton.Config(new_config, num_stages=cfg.num_stages, num_warps=cfg.num_warps))
    return filtered_configs

def get_extra_autotune_config(block_keys=None, include_group_size=False):
    # Extra configs with all M, N, K
    extra_configs = [triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64}, num_stages=3, num_warps=8), 
                     triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
                     triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
                     triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
                     triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
                     triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
                     triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32}, num_stages=5, num_warps=2),
                     triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32}, num_stages=5, num_warps=2),
                     triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=2),]
    filtered_extra_configs = filter_configs(extra_configs, block_keys, include_group_size)
    return filtered_extra_configs

def get_fp8_autotune_config(block_keys=None, include_group_size=False):
    # FP8 configs with all M, N, K
    fp8_configs = [triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128}, num_stages=3, num_warps=8),
                   triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128}, num_stages=3, num_warps=8),
                   triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128}, num_stages=4, num_warps=4),
                   triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128}, num_stages=4, num_warps=4),
                   triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128}, num_stages=4, num_warps=4),
                   triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64}, num_stages=4, num_warps=4),
                   triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64}, num_stages=4, num_warps=4),
                   triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 64}, num_stages=4, num_warps=4)]
    filtered_fp8_configs = filter_configs(fp8_configs, block_keys, include_group_size, group_size=8)
    return filtered_fp8_configs

def get_autotune_config(block_keys, include_group_size=False, include_fp8_configs=False, include_extra_configs=False):
    # Base configs with all M, N, K
    base_configs = [triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64}, num_stages=3, num_warps=8),
                    triton.Config({'BLOCK_SIZE_M': 64,  'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
                    triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
                    triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64,  'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
                    triton.Config({'BLOCK_SIZE_M': 64,  'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
                    triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32,  'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
                    triton.Config({'BLOCK_SIZE_M': 64,  'BLOCK_SIZE_N': 32,  'BLOCK_SIZE_K': 32}, num_stages=5, num_warps=2),
                    triton.Config({'BLOCK_SIZE_M': 32,  'BLOCK_SIZE_N': 64,  'BLOCK_SIZE_K': 32}, num_stages=5, num_warps=2),
                    triton.Config({'BLOCK_SIZE_M': 64,  'BLOCK_SIZE_N': 64,  'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=2)]
    filtered_base_configs = filter_configs(base_configs, block_keys, include_group_size)
    if include_fp8_configs:
        fp8_configs = get_fp8_autotune_config(block_keys, include_group_size)
        filtered_base_configs.extend(fp8_configs)
    if include_extra_configs:
        extra_configs = get_extra_autotune_config(block_keys, include_group_size)
        filtered_base_configs.extend(extra_configs)
    return filtered_base_configs

def get_cuda_autotune_config(block_keys=None, include_group_size=False, include_fp8_configs=False, include_extra_configs=False):
    if block_keys is None:
        config = []
        config.extend(triton.Config({}, num_warps=2**i) for i in range(1, 6))
        return config
    config = get_autotune_config(block_keys, include_group_size, include_fp8_configs, include_extra_configs)
    return config
