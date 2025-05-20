import triton
import triton.language as tl
import torch
from triton.language.extra.cuda.libdevice import pow
from TritonHub.autotune import get_cuda_autotune_config

@triton.autotune(
    configs=get_cuda_autotune_config(block_keys=['M', 'K'], include_extra_configs=True,include_fp8_configs=True),
    key=['M', 'K'])
@triton.jit
def _norm_fwd_kernel(x_ptr, stride_xb, stride_xm, stride_xk,
                     out_ptr, stride_outb, stride_outr,
                     eps, p: tl.constexpr, 
                     M: tl.constexpr, K: tl.constexpr, 
                     BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
                     dtype: tl.constexpr):
    pid_m = tl.program_id(axis=0)
    pid_b = tl.program_id(axis=1)

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    x_ptr += (pid_b * stride_xb) + (offs_m[:, None] * stride_xm)
    out = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32) 
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)): 
        x_ptrs = x_ptr + ((k * BLOCK_SIZE_K + offs_k[None, :]) * stride_xk)
        x = tl.load(x_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] < (K - k * BLOCK_SIZE_K)), other=0.0).to(tl.float32)
        if p == 1:
            x = tl.sum(tl.abs(x), axis=1)
        elif p == 2:
            x = tl.sum(x * x, axis=1)
        else:
            x = tl.sum(pow(tl.abs(x), p), axis=1)
        out += x
    
    if p == 2:
        out = tl.sqrt(out)
    elif p != 1:
        out = pow(out, 1.0 / p)
    out = tl.maximum(out, eps)
    out = out.to(dtype)

    out_ptr += pid_b * stride_outb
    out_ptrs = out_ptr + (offs_m * stride_outr)
    tl.store(out_ptrs, out, mask=offs_m < M)

def _norm_fwd(x, p=2, eps=1e-6):
    if x.stride(-1) != 1:
        x = x.contiguous()
    B, M, K = x.shape
    norms_x = torch.empty((B, M), memory_format=torch.contiguous_format, device=x.device, dtype=x.dtype)
    assert x.stride(-1) == 1, "Output tensors must be contiguous"
    dtype = (tl.bfloat16 if x.dtype == torch.bfloat16 else (tl.float16 if x.dtype == torch.float16 else tl.float32))
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']),
                         B,)
    with torch.cuda.device(x.device.index):
        _norm_fwd_kernel[grid](x, x.stride(0), x.stride(1), x.stride(2),
                               norms_x, norms_x.stride(0), norms_x.stride(1),
                               eps, p, M, K, dtype=dtype)
    return norms_x

@triton.autotune(
    configs=get_cuda_autotune_config(block_keys=['M', 'K'], include_extra_configs=True,include_fp8_configs=True),
    key=['M', 'K'])
@triton.jit
def _norm_bwd_kernel(x_ptr, stride_xb, stride_xm, stride_xk,
                     norms_x_ptr, stride_normsxb, stride_normsxm,
                     dout_ptr, stride_doutb, stride_doutm,
                     dx_ptr, stride_dxb, stride_dxm,
                     p: tl.constexpr, M: tl.constexpr, K: tl.constexpr,
                     BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
                     dtype: tl.constexpr):
    pid_m = tl.program_id(axis=0)
    pid_b = tl.program_id(axis=1)

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    x_ptr += (pid_b * stride_xb) + (offs_m[:, None] * stride_xm)
    dout_ptr += (pid_b * stride_doutb) + (offs_m[:, None] * stride_doutm)
    dx_ptr += (pid_b * stride_dxb) + (offs_m[:, None] * stride_dxm)

    if p != 1:
        norms_x_ptr += (pid_b * stride_normsxb) + (offs_m[:, None] * stride_normsxm)
        norms_x = tl.load(norms_x_ptr, mask=offs_m[:, None] < M, other=0.0).to(tl.float32)
        if p != 2:
            norms_x = pow(norms_x, p - 1)
    dout = tl.load(dout_ptr, mask=offs_m[:, None] < M, other=0.0).to(tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        x_ptrs = x_ptr + ((k * BLOCK_SIZE_K + offs_k[None, :]) * stride_xk)
        x = tl.load(x_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] < (K - k * BLOCK_SIZE_K)), other=0.0).to(tl.float32)
        if p == 1:
            sign = tl.where(x > 0, 1.0, tl.where(x < 0, -1.0, 0.0))
            grad = (dout * sign)
        elif p == 2:
            grad = (dout * x) / norms_x
        else:
            sign = tl.where(x > 0, 1.0, tl.where(x < 0, -1.0, 0.0))
            grad = (dout * sign * pow(tl.abs(x), p - 1)) / norms_x
        dx_ptrs = dx_ptr + ((k * BLOCK_SIZE_K + offs_k[None, :]) * stride_xk)
        tl.store(dx_ptrs, grad.to(dtype), mask=(offs_m[:, None] < M) & (offs_k[None, :] < (K - k * BLOCK_SIZE_K)))

def _norm_bwd(x, norms_x, dout, p=2):
    if norms_x.stride(-1) != 1:
        norms_x = norms_x.contiguous()
    if dout.stride(-1) != 1:
        dout = dout.contiguous()
    B, M = norms_x.shape
    K = x.shape[-1]
    assert x.shape == (B, M, K), "Input shape mismatch"
    assert norms_x.shape == (B, M), "Input gradient shape mismatch"
    assert dout.shape == (B, M), "Output gradient shape mismatch"
    assert x.dtype == norms_x.dtype == dout.dtype, "All tensors must have the same dtype"
    dx = torch.empty((B, M, K), memory_format=torch.contiguous_format, device=norms_x.device, dtype=norms_x.dtype)
    assert dx.stride(-1) == 1, "Output tensors must be contiguous"
    dtype = (tl.bfloat16 if norms_x.dtype == torch.bfloat16 else (tl.float16 if norms_x.dtype == torch.float16 else tl.float32))
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']),
                         B,)
    with torch.cuda.device(norms_x.device.index):
        _norm_bwd_kernel[grid](x, x.stride(0), x.stride(1), x.stride(2),
                               norms_x, norms_x.stride(0), norms_x.stride(1),
                               dout, dout.stride(0), dout.stride(1),
                               dx, dx.stride(0), dx.stride(1),
                               p, M, K, dtype=dtype)
    return dx

class Norm(torch.autograd.Function):
    @staticmethod
    @torch.amp.custom_fwd(device_type='cuda')
    def forward(ctx, x, p=2, eps=1e-6):
        assert len(x.shape) == 3, "Expected Tensor to be 3D (B, L, D), add batch dim input is 2D"
        ctx.p = p
        norms_x = _norm_fwd(x, p, eps)
        ctx.save_for_backward(x, norms_x)
        return norms_x

    @staticmethod
    @torch.amp.custom_bwd(device_type='cuda')
    def backward(ctx, grad_output):
        p = ctx.p
        x, norms_x = ctx.saved_tensors
        grad_x = _norm_bwd(x, norms_x, grad_output, p)
        return grad_x, None, None

class norm:
    def __init__(self, p=2, eps=1e-6):
        assert p >= 1 and p <= 6, "p must be in [1, 6] for numerical stability"
        assert eps >= 0, "eps must be non-negative"
        self.p = p
        self.eps = eps
        self.norm = Norm.apply
    def __call__(self, x):
        return self.norm(x, self.p, self.eps)