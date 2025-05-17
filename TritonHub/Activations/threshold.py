import torch
import triton
import triton.language as tl
from TritonHub.autotune import get_cuda_autotune_config

@triton.autotune(
    configs=get_cuda_autotune_config(block_keys=None),
    key=['N'],
)
@triton.jit
def _threshold_kernel_fwd(X, stride_X_row,
                          Y, stride_Y_row,
                          threshold, value,
                          N, BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.program_id(1) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    X = X + row * stride_X_row
    Y = Y + row * stride_Y_row
    x = tl.load(X + cols, mask=cols < N, other=0.0)
    y = tl.where(x > threshold, x, value)
    tl.store(Y + cols, y, mask=cols < N)

def _threshold_fwd(x, threshold, value):
    if x.stride(-1) != 1:
        x = x.contiguous()
    batch_shape = x.shape[:-1]
    x = x.reshape(-1, x.shape[-1])
    out = torch.empty_like(x, memory_format=torch.contiguous_format)
    assert out.shape == x.shape, 'expect output shape to be the same as input shape'
    assert out.stride(-1) == 1, 'expect output to be row-major'
    M, N = x.shape
    grid = lambda META: (M, triton.cdiv(N, META['BLOCK_SIZE']))
    MAX_FUSED_SIZE = 65536 // x.element_size()
    BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))
    with torch.cuda.device(x.device.index):
        _threshold_kernel_fwd[grid](x, x.stride(0),
                                    out, out.stride(0),
                                    threshold, value,
                                    N, BLOCK_SIZE=BLOCK_SIZE)
    return out.reshape(*batch_shape, out.shape[-1])

@triton.autotune(
    configs=get_cuda_autotune_config(block_keys=None),
    key=['N'],
)
@triton.jit
def _threshold_kernel_bwd(X, stride_X_row,
                          DOUT, stride_DOUT_row,
                          DX, stride_DX_row,
                          threshold, value,
                          N, BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.program_id(1) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    X = X + row * stride_X_row
    DOUT = DOUT + row * stride_DOUT_row
    DX = DX + row * stride_DX_row
    x = tl.load(X + cols, mask=cols < N, other=0.0)
    dout = tl.load(DOUT + cols, mask=cols < N, other=0.0)
    dx = tl.where(x > threshold, dout, 0.0)
    tl.store(DX + cols, dx, mask=cols < N)

def _threshold_bwd(x, dout, threshold, value):
    if x.stride(-1) != 1:
        x = x.contiguous()
    if dout.stride(-1) != 1:
        dout = dout.contiguous()
    batch_shape = x.shape[:-1]
    x = x.reshape(-1, x.shape[-1])
    dout = dout.reshape(-1, dout.shape[-1])
    assert x.shape == dout.shape, 'expect input and output shape to be the same'
    dx = torch.empty_like(x, memory_format=torch.contiguous_format)
    assert dx.stride(-1) == 1, 'expect derivative to be row-major'
    M, N = x.shape
    grid = lambda META: (M, triton.cdiv(N, META['BLOCK_SIZE']))
    MAX_FUSED_SIZE = 65536 // x.element_size()
    BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))
    with torch.cuda.device(x.device.index):
        _threshold_kernel_bwd[grid](x, x.stride(0),
                                    dout, dout.stride(0),
                                    dx, dx.stride(0),
                                    threshold, value,
                                    N, BLOCK_SIZE=BLOCK_SIZE)
    return dx.reshape(*batch_shape, dx.shape[-1])

class threshold_fn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, threshold, value):
        output = _threshold_fwd(input, threshold, value)
        ctx.save_for_backward(input)
        ctx.threshold = threshold
        ctx.value = value
        return output

    @staticmethod
    def backward(ctx, d_out):
        input, = ctx.saved_tensors
        threshold = ctx.threshold
        value = ctx.value
        grad = _threshold_bwd(input, d_out, threshold, value)
        return grad, None, None

class Threshold:
    def __init__(self, threshold, value):
        self.threshold_fn = threshold_fn.apply
        self.threshold = threshold
        self.value = value
    def __call__(self, x):
        return self.threshold_fn(x, self.threshold, self.value)