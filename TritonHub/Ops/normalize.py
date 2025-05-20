import triton
import triton.language as tl
import torch
from TritonHub.Ops.norm import _norm_fwd_kernel
from TritonHub.autotune import get_cuda_autotune_config

@triton.autotune(
    configs=get_cuda_autotune_config(block_keys=['M', 'K']),
    key=['M', 'K'])
@triton.jit
def _normalize_fwd_kernel(x_ptr, stride_xb, stride_xm, stride_xk,
                          norms_x_ptr, stride_normsxb, stride_normsxm,
                          x_norm_ptr, stride_xnormb, stride_xnormm, stride_xnormk,
                          M: tl.constexpr, K: tl.constexpr,
                          BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_K: tl.constexpr, 
                          dtype: tl.constexpr):
    pid_m = tl.program_id(axis=0)
    pid_b = tl.program_id(axis=1)

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    x_ptr += (pid_b * stride_xb) + (offs_m[:, None] * stride_xm)
    norms_x_ptr += (pid_b * stride_normsxb) + (offs_m[:, None] * stride_normsxm)
    x_norm_ptr += (pid_b * stride_xnormb) + (offs_m[:, None] * stride_xnormm)

    norms_x = tl.load(norms_x_ptr, mask=offs_m[:, None] < M, other=0.0).to(dtype)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        x_ptrs = x_ptr + ((k * BLOCK_SIZE_K + offs_k[None, :]) * stride_xk)
        x = tl.load(x_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] < (K - k * BLOCK_SIZE_K)), other=0.0).to(dtype)
        x_n = (x / norms_x).to(dtype)
        x_norm_ptrs = x_norm_ptr + ((k * BLOCK_SIZE_K + offs_k[None, :]) * stride_xnormk)
        tl.store(x_norm_ptrs, x_n, mask=(offs_m[:, None] < M) & (offs_k[None, :] < (K - k * BLOCK_SIZE_K)))

def _normalize_fwd(x, p=2, eps=1e-6):
    if x.stride(-1) != 1:
        x = x.contiguous()
    B, M, K = x.shape
    x_norm = torch.empty((B, M, K), memory_format=torch.contiguous_format, device=x.device, dtype=x.dtype)
    norms_x = torch.empty((B, M), memory_format=torch.contiguous_format, device=x.device, dtype=x.dtype)
    assert x.stride(-1) == 1, "Output tensors must be contiguous"
    dtype = (tl.bfloat16 if x.dtype == torch.bfloat16 else (tl.float16 if x.dtype == torch.float16 else tl.float32))
    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_SIZE_M']),
        B,
    )
    with torch.cuda.device(x.device.index):
        _norm_fwd_kernel[grid](x, x.stride(0), x.stride(1), x.stride(2),
                               norms_x, norms_x.stride(0), norms_x.stride(1),
                               eps, p, M, K, dtype=dtype)
        _normalize_fwd_kernel[grid](x, x.stride(0), x.stride(1), x.stride(2),
                                    norms_x, norms_x.stride(0), norms_x.stride(1), 
                                    x_norm, x_norm.stride(0), x_norm.stride(1), x_norm.stride(2),
                                    M, K, dtype=dtype)
    return x_norm, norms_x

@triton.autotune(
    configs=get_cuda_autotune_config(block_keys=['M', 'K']),
    key=['M', 'K'])
@triton.jit
def _normalize_bwd_kernel(x_ptr, stride_xb, stride_xm, stride_xk,
                          x_norm_ptr, stride_xnormb, stride_xnormm, stride_xnormk,
                          norms_x_ptr, stride_normsxb, stride_normsxm,
                          dout_ptr, stride_dout_b, stride_dout_m, stride_dout_k,
                          dx_ptr, stride_dxb, stride_dxm, stride_dxk,
                          p: tl.constexpr, M: tl.constexpr, K: tl.constexpr,
                          BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_K: tl.constexpr, 
                          dtype: tl.constexpr):
    pid_m = tl.program_id(axis=0)
    pid_b = tl.program_id(axis=1)

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    x_ptr += (pid_b * stride_xb) + (offs_m[:, None] * stride_xm)
    x_norm_ptr += (pid_b * stride_xnormb) + (offs_m[:, None] * stride_xnormm)
    norms_x_ptr += (pid_b * stride_normsxb) + (offs_m[:, None] * stride_normsxm)
    dout_ptr += (pid_b * stride_dout_b) + (offs_m[:, None] * stride_dout_m)
    dx_ptr += (pid_b * stride_dxb) + (offs_m[:, None] * stride_dxm)
    
    norms_x = tl.load(norms_x_ptr, mask=offs_m[:, None] < M, other=0.0).to(tl.float32)
    if p != 1 and p != 2:
        norms_x_p = pow(norms_x, p - 1).to(tl.float32)

    proj = tl.zeros([BLOCK_SIZE_M], dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        dout_ptrs = dout_ptr + ((k * BLOCK_SIZE_K + offs_k[None, :]) * stride_dout_k)
        dout = tl.load(dout_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] < (K - k * BLOCK_SIZE_K)), other=0.0).to(tl.float32)   
        if p == 1:
            x_ptrs = x_ptr + ((k * BLOCK_SIZE_K + offs_k[None, :]) * stride_xk)
            x = tl.load(x_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] < (K - k * BLOCK_SIZE_K)), other=0.0).to(tl.float32)
            proj += tl.sum(x * dout, axis=1)
        else:
            x_norm_ptrs = x_norm_ptr + ((k * BLOCK_SIZE_K + offs_k[None, :]) * stride_xnormk)
            x_norm = tl.load(x_norm_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] < (K - k * BLOCK_SIZE_K)), other=0.0).to(tl.float32)
            proj += tl.sum(x_norm * dout, axis=1)
    
    proj = proj[:, None]
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        dout_ptrs = dout_ptr + ((k * BLOCK_SIZE_K + offs_k[None, :]) * stride_dout_k)
        dout = tl.load(dout_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] < (K - k * BLOCK_SIZE_K)), other=0.0).to(tl.float32)
        if p == 1:
            x_ptrs = x_ptr + ((k * BLOCK_SIZE_K + offs_k[None, :]) * stride_xk)
            x = tl.load(x_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] < (K - k * BLOCK_SIZE_K)), other=0.0).to(tl.float32)
            sign = tl.where(x > 0, 1.0, tl.where(x < 0, -1.0, 0.0))
            dx = (dout / norms_x) - (sign * (proj / (norms_x * norms_x)))
        elif p == 2:
            x_norm_ptrs = x_norm_ptr + ((k * BLOCK_SIZE_K + offs_k[None, :]) * stride_xnormk)
            x_norm = tl.load(x_norm_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] < (K - k * BLOCK_SIZE_K)), other=0.0).to(tl.float32)
            dx = (dout - (x_norm * proj)) / norms_x
        else:
            x_ptrs = x_ptr + ((k * BLOCK_SIZE_K + offs_k[None, :]) * stride_xk)
            x = tl.load(x_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] < (K - k * BLOCK_SIZE_K)), other=0.0).to(tl.float32)
            sign = tl.where(x > 0, 1.0, tl.where(x < 0, -1.0, 0.0))
            dx = (dout / norms_x)  - (sign * pow(tl.abs(x), p - 1) * (proj / (norms_x_p * norms_x)))
        dx_ptrs = dx_ptr + ((k * BLOCK_SIZE_K + offs_k[None, :]) * stride_dxk)
        tl.store(dx_ptrs, dx.to(dtype), mask=(offs_m[:, None] < M) & (offs_k[None, :] < (K - k * BLOCK_SIZE_K)))

def _normalize_bwd(x, x_norm, norms_x, dout, p=2):
    if x_norm.stride(-1) != 1:
        x_norm = x_norm.contiguous()
    if norms_x.stride(-1) != 1:
        norms_x = norms_x.contiguous()
    if dout.stride(-1) != 1:
        dout = dout.contiguous()
    B, M, K = x_norm.shape
    assert dout.shape == (B, M, K), "Output gradient shape mismatch"
    assert x_norm.dtype == norms_x.dtype == dout.dtype, "All tensors must have the same dtype"
    dx = torch.empty((B, M, K), memory_format=torch.contiguous_format, device=x_norm.device, dtype=x_norm.dtype)
    assert dx.stride(-1) == 1, "Output tensors must be contiguous"
    dtype = (tl.bfloat16 if x_norm.dtype == torch.bfloat16 else (tl.float16 if x_norm.dtype == torch.float16 else tl.float32))
    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_SIZE_M']),
        B,
    )
    _normalize_bwd_kernel[grid](x, x.stride(0), x.stride(1), x.stride(2),
                                x_norm, x_norm.stride(0), x_norm.stride(1), x_norm.stride(2),
                                norms_x, norms_x.stride(0), norms_x.stride(1),
                                dout, dout.stride(0), dout.stride(1), dout.stride(2),
                                dx, dx.stride(0), dx.stride(1), dx.stride(2),
                                p, M, K, dtype=dtype)
    return dx

class Normalize(torch.autograd.Function):
    @staticmethod
    @torch.amp.custom_fwd(device_type='cuda')
    def forward(ctx, x, p=2, eps=1e-6):
        assert len(x.shape) == 3, "Expected Tensor to be 3D (B, L, D), add batch dim input is 2D"
        ctx.p = p
        x_norm, norms_x = _normalize_fwd(x, p, eps)
        ctx.save_for_backward(x, x_norm, norms_x)
        return x_norm

    @staticmethod
    @torch.amp.custom_bwd(device_type='cuda')
    def backward(ctx, grad_output):
        p = ctx.p
        x, x_norm, norms_x = ctx.saved_tensors
        grad_x = _normalize_bwd(x, x_norm, norms_x, grad_output, p)
        return grad_x, None, None

class normalize:
    def __init__(self, p=2, eps=1e-6):
        assert p >= 1 and p <= 6, "p must be in [1, 6] for numerical stability"
        assert eps >= 0, "eps must be non-negative"
        self.p = p
        self.eps = eps
        self.normalize = Normalize.apply
    def __call__(self, x):
        return self.normalize(x, self.p, self.eps)