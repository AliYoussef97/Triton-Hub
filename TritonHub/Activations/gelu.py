import torch
import triton
import triton.language as tl
import math
from TritonHub.autotune import get_cuda_autotune_config

@triton.autotune(
    configs=get_cuda_autotune_config(block_keys=None),
    key=['N'],
)
@triton.jit
def _gelu_kernel_fwd(X, stride_X_row,
                     Y, stride_Y_row,
                     N, BLOCK_SIZE: tl.constexpr,
                     approx: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.program_id(1) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    X = X + row * stride_X_row
    Y = Y + row * stride_Y_row
    x = tl.load(X + cols, mask=cols < N, other=0.0).to(tl.float32)
    if approx:
        pi = math.pi
        x_inner_tanh = tl.sqrt(2.0 / pi) * (x + 0.044715 * x * x * x)
        exp_2x = tl.exp(2.0 * x_inner_tanh)
        tanh_x = (exp_2x - 1.0) / (exp_2x + 1.0)
        y = 0.5 * x * (1.0 + tanh_x)
    else:
        y = 0.5 * x * (1.0 + tl.erf(x / tl.sqrt(2.0)))
    tl.store(Y + cols, y, mask=cols < N)

def _gelu_fwd(x, approximate='none'):
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
    approx = (approximate == 'tanh')
    with torch.cuda.device(x.device.index):
        _gelu_kernel_fwd[grid](x, x.stride(0),
                               out, out.stride(0),
                               N, BLOCK_SIZE=BLOCK_SIZE,
                               approx=approx)
    return out.reshape(*batch_shape, out.shape[-1])

@triton.autotune(
    configs=get_cuda_autotune_config(block_keys=None),
    key=['N'],
)
@triton.jit
def _gelu_kernel_bwd(X, stride_X_row,
                     DOUT, stride_DOUT_row,
                     DX, stride_DX_row,
                     N, BLOCK_SIZE: tl.constexpr,
                     approx: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.program_id(1) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    X = X + row * stride_X_row
    DOUT = DOUT + row * stride_DOUT_row
    DX = DX + row * stride_DX_row
    x = tl.load(X + cols, mask=cols < N, other=0.0).to(tl.float32)
    dout = tl.load(DOUT + cols, mask=cols < N, other=0.0).to(tl.float32)
    if approx:
        pi = math.pi
        sqrt_2_pi = tl.sqrt(2.0 / pi)
        g_x = sqrt_2_pi * (x + 0.044715 * x * x * x)        
        exp_2gx = tl.exp(2.0 * g_x)
        tanh_g_x = (exp_2gx - 1.0) / (exp_2gx + 1.0)
        g_prime_x = sqrt_2_pi * (1.0 + 3.0 * 0.044715 * x * x)        
        dy_dx = 0.5 * (1.0 + tanh_g_x + x * (1.0 - tanh_g_x * tanh_g_x) * g_prime_x)
    else:
        cdf = 0.5 * (1.0 + tl.erf(x / tl.sqrt(2.0)))
        pdf = tl.exp(-0.5 * x * x) / tl.sqrt(2.0 * math.pi)
        dy_dx = cdf + x * pdf    
    dx = dout * dy_dx
    tl.store(DX + cols, dx, mask=cols < N)

def _gelu_bwd(x, dout, approximate='none'):
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
    approx = (approximate == 'tanh')
    with torch.cuda.device(x.device.index):
        _gelu_kernel_bwd[grid](x, x.stride(0),
                               dout, dout.stride(0),
                               dx, dx.stride(0),
                               N, BLOCK_SIZE=BLOCK_SIZE,
                               approx=approx)
    return dx.reshape(*batch_shape, dx.shape[-1])

class gelu(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, approximate):
        output = _gelu_fwd(input, approximate)
        ctx.save_for_backward(input)
        ctx.approximate = approximate
        return output

    @staticmethod
    def backward(ctx, d_out):
        input, = ctx.saved_tensors
        grad = _gelu_bwd(input, d_out, ctx.approximate)
        return grad, None

class GeLU:
    def __init__(self, approximate='none'):
        assert approximate in ['none', 'tanh'], 'GeLU approximate must be either none or tanh'
        self.approximate = approximate
        if approximate == 'none':
            assert hasattr(tl, 'erf'), 'GeLU requires tl.erf for non-approximate mode, use approximate="tanh" or upgrade triton.'
        self.gelu_fn = gelu.apply
    def __call__(self, x):
        return self.gelu_fn(x, self.approximate)