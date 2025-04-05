import torch
import triton
import triton.language as tl

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
def _logsoftmax_kernel_fwd(X, stride_X_row,
                           Y, stride_Y_row,
                           N, BLOCK_SIZE: tl.constexpr,
                           dtype: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.program_id(1) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    X = X + row * stride_X_row
    Y = Y + row * stride_Y_row
    x = tl.load(X + cols, mask=cols < N, other=-float('inf')).to(tl.float32)
    z = x - tl.max(x, axis=0)
    log_denom = tl.log(tl.sum(tl.exp(z), axis=0))
    y = z - log_denom
    y = y.to(dtype)
    tl.store(Y + cols, y, mask=cols < N)

def _logsoftmax_fwd(x):
    if x.stride(-1) != 1:
        x = x.contiguous()
    batch_shape = x.shape[:-1]
    x = x.reshape(-1, x.shape[-1])
    dtype = (tl.bfloat16 if x.dtype == torch.bfloat16 else (tl.float16 if x.dtype == torch.float16 else tl.float32))
    out = torch.empty_like(x, memory_format=torch.contiguous_format, device=x.device, dtype=x.dtype)
    assert out.shape == x.shape, 'expect output shape to be the same as input shape'
    assert out.stride(-1) == 1, 'expect output to be row-major'
    M, N = x.shape
    grid = lambda META: (M, triton.cdiv(N, META['BLOCK_SIZE']))
    MAX_FUSED_SIZE = 65536 // x.element_size()
    BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))
    with torch.cuda.device(x.device.index):
        _logsoftmax_kernel_fwd[grid](x, x.stride(0),
                                     out, out.stride(0),
                                     N, BLOCK_SIZE=BLOCK_SIZE,
                                     dtype=dtype)
    return out.reshape(*batch_shape, out.shape[-1])

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
def _logsoftmax_kernel_bwd(X, stride_X_row,
                           DOUT, stride_DOUT_row,
                           DX, stride_DX_row,
                           N, BLOCK_SIZE: tl.constexpr,
                           dtype: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.program_id(1) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    X = X + row * stride_X_row
    DOUT = DOUT + row * stride_DOUT_row
    DX = DX + row * stride_DX_row
    x = tl.load(X + cols, mask=cols < N, other=0.0).to(tl.float32)
    dout = tl.load(DOUT + cols, mask=cols < N, other=0.0).to(tl.float32)
    softmax = tl.exp(x)
    sum_dout = tl.sum(dout, axis=0)
    dx = dout - (softmax * sum_dout)
    dx = dx.to(dtype)
    tl.store(DX + cols, dx, mask=cols < N)

def _logsoftmax_bwd(x, dout):
    if x.stride(-1) != 1:
        x = x.contiguous()
    if dout.stride(-1) != 1:
        dout = dout.contiguous()
    batch_shape = x.shape[:-1]
    x = x.reshape(-1, x.shape[-1])
    dout = dout.reshape(-1, dout.shape[-1])
    dtype = (tl.bfloat16 if x.dtype == torch.bfloat16 else (tl.float16 if x.dtype == torch.float16 else tl.float32))
    assert x.shape == dout.shape, 'expect input and output shape to be the same'
    dx = torch.empty_like(x, memory_format=torch.contiguous_format, device=x.device, dtype=x.dtype)
    assert dx.stride(-1) == 1, 'expect derivative to be row-major'
    M, N = x.shape
    grid = lambda META: (M, triton.cdiv(N, META['BLOCK_SIZE']))
    MAX_FUSED_SIZE = 65536 // x.element_size()
    BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))
    with torch.cuda.device(x.device.index):
        _logsoftmax_kernel_bwd[grid](x, x.stride(0),
                                     dout, dout.stride(0),
                                     dx, dx.stride(0),
                                     N, BLOCK_SIZE=BLOCK_SIZE,
                                     dtype=dtype)
    return dx.reshape(*batch_shape, dx.shape[-1])

class logsoftmax(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        output = _logsoftmax_fwd(input)
        ctx.save_for_backward(output)
        return output

    @staticmethod
    def backward(ctx, d_out):
        output, = ctx.saved_tensors
        grad = _logsoftmax_bwd(output, d_out)
        return grad

class LogSoftmax:
    def __init__(self):
        self.logsoftmax_fn = logsoftmax.apply
    def __call__(self, x):
        return self.logsoftmax_fn(x)