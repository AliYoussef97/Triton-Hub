# This script is highly influenced by Mamba's implementation of the RMSNorm in Triton which can be found at:
# [https://github.com/state-spaces/mamba/blob/main/mamba_ssm/ops/triton/layer_norm.py]
import torch
import triton
import triton.language as tl
import math

@triton.autotune(
    configs=[
        triton.Config({}, num_warps=2),
        triton.Config({}, num_warps=4),
        triton.Config({}, num_warps=8),
        triton.Config({}, num_warps=16),
        triton.Config({}, num_warps=32)
    ],
    key=['N'],
)
@triton.jit
def _rmsnorm_kernel_fwd(X, 
                        stride_X_row,
                        Y, 
                        stride_Y_row,
                        W,
                        Rstd,
                        eps,
                        N, 
                        BLOCK_SIZE: tl.constexpr,
                        HAS_WEIGHT: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    X = X + row * stride_X_row
    Y = Y + row * stride_Y_row
    x = tl.load(X + cols, mask=cols < N, other=0.0).to(tl.float32)
    xbar = tl.where(cols < N, x, 0.0)
    var = tl.sum(xbar * xbar, axis=0) / N
    rstd = 1 / tl.sqrt(var + eps)
    tl.store(Rstd + row, rstd)
    y = x * rstd
    mask = cols < N
    if HAS_WEIGHT:
        w = tl.load(W + cols, mask=mask).to(tl.float32)
        y = y * w
    tl.store(Y + cols, y, mask=mask)

def _rmsnorm_fwd(x, weight, eps):
    if x.stride(-1) != 1:
        x = x.contiguous()
    batch_shape = x.shape[:-1]
    x = x.reshape(-1, x.shape[-1])
    assert x.stride(-1) == 1, 'expect input to be row-major'
    M, N = x.shape
    if weight is not None:
        weight = weight.contiguous()
        assert weight.shape == (N,), 'expect weight to have shape (N,)'
        assert weight.stride(-1) == 1, 'expect weight to be row-major'
    out = torch.empty_like(x, memory_format=torch.contiguous_format, dtype=x.dtype)
    assert out.stride(-1) == 1, 'expect output to be row-major'
    rstd = torch.empty((M,), dtype=torch.float32, device=x.device)
    HAS_WEIGHT = True if weight is not None else False
    MAX_FUSED_SIZE = 65536 // x.element_size()
    BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))
    if N > BLOCK_SIZE:
        raise RuntimeError("This layer norm doesn't support feature dim >= 64KB.")
    with torch.cuda.device(x.device.index):
        _rmsnorm_kernel_fwd[(M,)](x, 
                                  x.stride(0),
                                  out, 
                                  out.stride(0),
                                  weight,
                                  rstd,
                                  eps,
                                  N, 
                                  BLOCK_SIZE=BLOCK_SIZE,
                                  HAS_WEIGHT=HAS_WEIGHT)
    return out.reshape(*batch_shape, out.shape[-1]), rstd

@triton.autotune(
    configs=[
        triton.Config({}, num_warps=2),
        triton.Config({}, num_warps=4),
        triton.Config({}, num_warps=8),
        triton.Config({}, num_warps=16),
        triton.Config({}, num_warps=32)
    ],
    key=['N'],
)
@triton.jit
def _rmsnorm_kernel_bwd(X, 
                        stride_X_row,
                        DOUT,
                        stride_DOUT_row,
                        DX,
                        stride_DX_row,
                        W,
                        DW,
                        Rstd,
                        eps,
                        M,
                        N,
                        rows_per_program,
                        BLOCK_SIZE: tl.constexpr,
                        HAS_WEIGHT: tl.constexpr):
    rows = tl.program_id(0)
    row_start = rows * rows_per_program
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N
    X = X + row_start * stride_X_row
    DOUT = DOUT + row_start * stride_DOUT_row
    DX = DX + row_start * stride_DX_row
    if HAS_WEIGHT:
        w = tl.load(W + cols, mask=mask).to(tl.float32)
        dw = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    row_end = min((rows + 1) * rows_per_program, M)
    for row in range(row_start, row_end):
        x = tl.load(X + cols, mask=mask, other=0).to(tl.float32)
        dout = tl.load(DOUT + cols, mask=mask, other=0).to(tl.float32)
        rstd = tl.load(Rstd + row)
        xhat = x * rstd
        xhat = tl.where(mask, xhat, 0.0)
        if HAS_WEIGHT:
            wdy = w * dout
            dw = dw + dout * xhat
            c1 = tl.sum(xhat * wdy, axis=0) / N
            dx = (wdy - xhat * c1) * rstd
        else:
            c1 = tl.sum(xhat * dout, axis=0) / N
            dx = (dout - xhat * c1 ) * rstd
        tl.store(DX + cols, dx, mask=mask)
        X = X + stride_X_row
        DOUT = DOUT + stride_DOUT_row
        DX = DX + stride_DX_row
    if HAS_WEIGHT:
        tl.store(DW + rows * N + cols, dw, mask=mask)

def _rmsnorm_bwd(x, dout, weight, rstd, eps):
    batch_shape = x.shape[:-1]
    x = x.reshape(-1, x.shape[-1])
    M, N = x.shape
    if x.stride(-1) != 1:
        x = x.contiguous()
    assert x.stride(-1) == 1, 'expect input to be row-major'
    dout = dout.reshape(-1, dout.shape[-1])
    if dout.stride(-1) != 1:
        dout = dout.contiguous()
    assert dout.stride(-1) == 1, 'expect output to be row-major'
    assert dout.shape == (M, N), 'expect input and output shape to be the same'
    assert x.shape == dout.shape, 'expect input and output shape to be the same'
    if weight is not None:
        weight = weight.contiguous()
        assert weight.shape == (N,), 'expect weight to have shape (N,)'
        assert weight.stride(-1) == 1, 'expect weight to be row-major'
    dx = torch.empty_like(x, memory_format=torch.contiguous_format, dtype=x.dtype)
    assert dx.stride(-1) == 1, 'expect derivative to be row-major'
    MAX_FUSED_SIZE = 65536 // x.element_size()
    BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))
    HAS_WEIGHT = True if weight is not None else False
    sm_count = torch.cuda.get_device_properties(x.device).multi_processor_count
    _dw = torch.empty((sm_count, N), dtype=torch.float32, device=x.device) if HAS_WEIGHT else None
    rows_per_program = math.ceil(M / sm_count)
    grid = (sm_count,)
    with torch.cuda.device(x.device.index):
        _rmsnorm_kernel_bwd[grid](x, 
                                  x.stride(0),
                                  dout, 
                                  dout.stride(0),
                                  dx, 
                                  dx.stride(0),
                                  weight,
                                  _dw,
                                  rstd,
                                  eps,
                                  M,
                                  N,
                                  rows_per_program, 
                                  BLOCK_SIZE=BLOCK_SIZE,
                                  HAS_WEIGHT=HAS_WEIGHT)
    dw = _dw.sum(0).to(weight.dtype) if _dw is not None else None
    return dx.reshape(*batch_shape, dx.shape[-1]), dw

class rmsnorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, eps):
        output, rstd = _rmsnorm_fwd(input, weight, eps)
        ctx.save_for_backward(input, weight, rstd)
        ctx.eps = eps
        return output

    @staticmethod
    def backward(ctx, d_out):
        input, weight, rstd = ctx.saved_tensors
        grad, dw, = _rmsnorm_bwd(input, d_out, weight, rstd, ctx.eps)
        return grad, dw, None

class RMSNorm(torch.nn.Module):
    def __init__(self, dimension, eps=1e-5, elementwise_affine=True, device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.dim = dimension
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = torch.nn.Parameter(torch.empty(self.dim, **factory_kwargs))
        else:
            self.register_parameter('weight', None)
        self.register_parameter('bias', None)
        self.reset_parameters()

        self.rmsnorm_fn = rmsnorm.apply
    
    def reset_parameters(self):
        if self.elementwise_affine:
            torch.nn.init.ones_(self.weight)

    def forward(self, x):
        return self.rmsnorm_fn(x, self.weight, self.eps)