import triton
import triton.language as tl
import torch
from TritonHub.autotune import get_cuda_autotune_config

@triton.autotune(
    configs=get_cuda_autotune_config(block_keys=['M', 'N', 'K']),
    key=['M', 'N', 'K'])
@triton.jit
def _batched_matmul_fwd_kernel(x_ptr, stride_xb, stride_xm, stride_xk,
                               y_ptr, stride_yb, stride_yn, stride_yk,
                               out_ptr, stride_outb, stride_outm, stride_outn,
                               M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                               BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
                               BLOCK_SIZE_K: tl.constexpr, dtype: tl.constexpr):
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    pid_m = tl.program_id(axis=0) % num_pid_m
    pid_n = tl.program_id(axis=0) // num_pid_m
    pid_b = tl.program_id(axis=1)

    x_ptr += pid_b * stride_xb
    y_ptr += pid_b * stride_yb

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    x_ptrs = x_ptr + (offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk)
    y_ptrs = y_ptr + (offs_k[:, None] * stride_yk + offs_n[None, :] * stride_yn)

    out = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        x = tl.load(x_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] < (K - k * BLOCK_SIZE_K)), other=0.0).to(dtype)
        y = tl.load(y_ptrs, mask=(offs_k[:, None] < (K - k * BLOCK_SIZE_K)) & (offs_n[None, :] < N), other=0.0).to(dtype)
        out += tl.dot(x, y)
        x_ptrs += BLOCK_SIZE_K * stride_xk
        y_ptrs += BLOCK_SIZE_K * stride_yk
    
    out = out.to(dtype)

    out_ptr += pid_b * stride_outb
    out_ptrs = out_ptr + (stride_outm * offs_m[:, None] + offs_n[None, :] * stride_outn)
    tl.store(out_ptrs, out, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))

def _batched_matmul_fwd(x, y):
    if x.stride(-1) != 1:
        x = x.contiguous()
    if y.stride(-1) != 1:
        y = y.contiguous()
    B, M, K = x.shape
    _, N, K_ = y.shape
    assert K == K_, "Input dimension mismatch"
    assert x.dtype == y.dtype, "Input tensors must have the same dtype"
    out = torch.empty((B, M, N), memory_format=torch.contiguous_format, device=x.device, dtype=x.dtype)
    assert out.stride(-1) == 1, "Output tensors must be contiguous"
    dtype = (tl.bfloat16 if x.dtype == torch.bfloat16 else (tl.float16 if x.dtype == torch.float16 else tl.float32))
    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_SIZE_M'])*
        triton.cdiv(N, META['BLOCK_SIZE_N']),
        B,
    )
    with torch.cuda.device(x.device.index):
        _batched_matmul_fwd_kernel[grid](x, x.stride(0), x.stride(1), x.stride(2),
                                         y, y.stride(0), y.stride(1), y.stride(2),
                                         out, out.stride(0), out.stride(1), out.stride(2),
                                         M, N, K, dtype=dtype)
    return out

@triton.autotune(
    configs=get_cuda_autotune_config(block_keys=['M', 'N', 'K']),
    key=['M', 'N', 'K'])
@triton.jit
def _batched_matmul_bwd_kernel_x(y_ptr, stride_yb, stride_yn, stride_yk,
                                 dout_ptr, stride_doutb, stride_doutm, stride_doutn,
                                 dx_ptr, stride_dxb, stride_dxm, stride_dxk,
                                 M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                                 BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
                                 BLOCK_SIZE_K: tl.constexpr, dtype: tl.constexpr):
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    pid_m = tl.program_id(axis=0) % num_pid_m
    pid_k = tl.program_id(axis=0) // num_pid_m
    pid_b = tl.program_id(axis=1)

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_k = pid_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
    offs_n = tl.arange(0, BLOCK_SIZE_N)

    y_ptr += pid_b * stride_yb
    dout_ptr += pid_b * stride_doutb
    
    y_ptrs = y_ptr + (offs_n[:, None] * stride_yn + offs_k[None, :] * stride_yk)
    dout_ptrs = dout_ptr + (offs_m[:, None] * stride_doutm + offs_n[None, :] * stride_doutn)
    
    dx = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_K), dtype=tl.float32)
    
    for n in range(0, tl.cdiv(N, BLOCK_SIZE_N)):
        y = tl.load(y_ptrs, mask=(offs_n[:, None] < (N - n * BLOCK_SIZE_N)) & (offs_k[None, :] < K), other=0.0).to(dtype)
        dout = tl.load(dout_ptrs, mask=(offs_m[:, None] < M) & (offs_n[None, :] < (N - n * BLOCK_SIZE_N)), other=0.0).to(dtype)
        dx += tl.dot(dout, y)
        y_ptrs += BLOCK_SIZE_N * stride_yn
        dout_ptrs += BLOCK_SIZE_N * stride_doutn
    
    dx = dx.to(dtype)
    
    dx_ptr += pid_b * stride_dxb
    dx_ptrs = dx_ptr + offs_m[:, None] * stride_dxm + offs_k[None, :] * stride_dxk
    tl.store(dx_ptrs, dx, mask=(offs_m[:, None] < M) & (offs_k[None, :] < K))

@triton.autotune(
    configs=get_cuda_autotune_config(block_keys=['M', 'N', 'K']),
    key=['M', 'N', 'K'])
@triton.jit
def _batched_matmul_bwd_kernel_y(x_ptr, stride_xb, stride_xm, stride_xk,
                                 dout_ptr, stride_doutb, stride_doutm, stride_doutn,
                                 dy_ptr, stride_dyb, stride_dyn, stride_dyk,
                                 M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                                 BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
                                 BLOCK_SIZE_K: tl.constexpr, dtype: tl.constexpr):
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    pid_n = tl.program_id(axis=0) % num_pid_n
    pid_k = tl.program_id(axis=0) // num_pid_n
    pid_b = tl.program_id(axis=1)

    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = pid_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
    offs_m = tl.arange(0, BLOCK_SIZE_M)

    x_ptr += pid_b * stride_xb
    dout_ptr += pid_b * stride_doutb
    
    x_ptrs = x_ptr + (offs_k[:, None] * stride_xk + offs_m[None, :] * stride_xm)
    dout_ptrs = dout_ptr + (offs_m[:, None] * stride_doutm + offs_n[None, :] * stride_doutn)
    
    dy = tl.zeros((BLOCK_SIZE_K, BLOCK_SIZE_N), dtype=tl.float32)
    
    for m in range(0, tl.cdiv(M, BLOCK_SIZE_M)):
        x = tl.load(x_ptrs, mask=(offs_k[:, None] < K) & (offs_m[None, :] < (M - m * BLOCK_SIZE_M)), other=0.0).to(dtype)
        dout = tl.load(dout_ptrs, mask=(offs_m[:, None] < (M - m * BLOCK_SIZE_M)) & (offs_n[None, :] < N), other=0.0).to(dtype)
        dy += tl.dot(x, dout)
        x_ptrs += BLOCK_SIZE_M * stride_xm
        dout_ptrs += BLOCK_SIZE_M * stride_doutm
    
    dy = dy.trans(1, 0).to(dtype)
    
    dy_ptr += pid_b * stride_dyb
    dy_ptrs = dy_ptr + offs_n[:, None] * stride_dyn + offs_k[None, :] * stride_dyk
    tl.store(dy_ptrs, dy, mask=(offs_n[:, None] < N) & (offs_k[None, :] < K))

def _batched_matmul_bwd(x, y, dout):
    if x.stride(-1) != 1:
        x = x.contiguous()
    if y.stride(-1) != 1:
        y = y.contiguous()
    if dout.stride(-1) != 1:
        dout = dout.contiguous()
    B, M, K = x.shape
    _, N, K_ = y.shape
    assert K == K_, "Input dimension mismatch"
    assert dout.shape == (B, M, N), "Output gradient shape mismatch"
    assert x.dtype == y.dtype == dout.dtype, "Input tensors must have the same dtype"
    dx = torch.empty((B, M, K), memory_format=torch.contiguous_format, device=x.device, dtype=x.dtype)
    dy = torch.empty((B, N, K), memory_format=torch.contiguous_format, device=x.device, dtype=x.dtype)
    assert dx.stride(-1) == 1 and dy.stride(-1) == 1, "Output tensors must be contiguous"
    dtype = (tl.bfloat16 if x.dtype == torch.bfloat16 else (tl.float16 if x.dtype == torch.float16 else tl.float32))
    grid_x = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(K, META['BLOCK_SIZE_K']), B)
    grid_y = lambda META: (triton.cdiv(N, META['BLOCK_SIZE_N']) * triton.cdiv(K_, META['BLOCK_SIZE_K']), B)
    _batched_matmul_bwd_kernel_x[grid_x](y, y.stride(0), y.stride(1), y.stride(2),
                                         dout, dout.stride(0), dout.stride(1), dout.stride(2),
                                         dx, dx.stride(0), dx.stride(1), dx.stride(2),
                                         M, N, K, dtype=dtype)
    _batched_matmul_bwd_kernel_y[grid_y](x, x.stride(0), x.stride(1), x.stride(2),
                                         dout, dout.stride(0), dout.stride(1), dout.stride(2),
                                         dy, dy.stride(0), dy.stride(1), dy.stride(2),
                                         M, N, K, dtype=dtype)
    
    return dx, dy

class batched_matmul(torch.autograd.Function):
    @staticmethod
    @torch.amp.custom_fwd(device_type='cuda')
    def forward(ctx, x, y):
        assert len(x.shape) == 3 and len(y.shape) == 3, "Expected 3D (B, M, K) and 3D (B, N, K) tensors"
        output = _batched_matmul_fwd(x, y)
        ctx.save_for_backward(x, y)
        return output

    @staticmethod
    @torch.amp.custom_bwd(device_type='cuda')
    def backward(ctx, grad_output):
        x, y = ctx.saved_tensors
        grad_x, grad_y = _batched_matmul_bwd(x, y, grad_output)
        return grad_x, grad_y

class bmm:
    def __init__(self):
        self.bmm = batched_matmul.apply
    def __call__(self, x, y):
        return self.bmm(x, y)