import triton
import triton.language as tl
import torch
from TritonHub.autotune import get_cuda_autotune_config

@triton.autotune(
    configs=get_cuda_autotune_config(block_keys=['M', 'K']),
    key=['M', 'K'])
@triton.jit
def _compute_norms_kernel(in_ptr, stride_inb, stride_inm, stride_ink,
                          out_ptr, stride_outb, stride_outr,
                          eps, M: tl.constexpr, K: tl.constexpr, 
                          BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
                          dtype: tl.constexpr):
    pid_m = tl.program_id(axis=0)
    pid_b = tl.program_id(axis=1)

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    in_ptr += (pid_b * stride_inb) + (offs_m[:, None] * stride_inm)
    out = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32) 
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)): 
        in_ptrs = in_ptr + ((k * BLOCK_SIZE_K + offs_k[None, :]) * stride_ink)
        inputs = tl.load(in_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] < (K - k * BLOCK_SIZE_K)), other=0.0).to(tl.float32) 
        inputs = tl.sum(inputs * inputs, axis=1) 
        out += inputs
        in_ptrs += BLOCK_SIZE_K * stride_ink
    out = tl.sqrt(out) 
    out = tl.maximum(out, eps)
    out = out.to(dtype)

    out_ptr += pid_b * stride_outb
    out_ptrs = out_ptr + (offs_m * stride_outr)
    tl.store(out_ptrs, out, mask=offs_m < M)