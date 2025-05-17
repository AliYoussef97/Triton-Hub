import torch
import triton
import triton.language as tl
import random
from TritonHub.autotune import get_cuda_autotune_config

@triton.autotune(
    configs=get_cuda_autotune_config(block_keys=None),
    key=['M'],
)
@triton.jit
def _dropout_kernel_fwd(X,
                        Y,
                        M, 
                        p, 
                        seed,
                        BLOCK_SIZE: tl.constexpr):
    cols = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    x = tl.load(X + cols, mask=cols < M)
    rand_p = tl.rand(seed, cols)
    y = tl.where(rand_p < p, 0.0, x/(1-p))
    tl.store(Y + cols, y, mask=cols < M)

def _dropout_fwd(x, p, seed, training):
    if x.stride(-1) != 1:
        x = x.contiguous()
    assert x.stride(-1) == 1, 'expect input to be row-major'
    if (p == 0.0) or (not training):
        return x
    elif p == 1.0:
        return torch.zeros_like(x, memory_format=torch.contiguous_format)
    else:
        input_shape = x.shape
        N = x.shape[-1]
        x = x.reshape(-1)
        out = torch.empty_like(x, memory_format=torch.contiguous_format)
        assert out.shape == x.shape, 'expect output shape to be the same as input shape'
        assert out.stride(-1) == 1, 'expect output to be row-major'
        M = x.shape[0]
        grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE']),)
        MAX_FUSED_SIZE = 65536 // x.element_size()
        BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))
        with torch.cuda.device(x.device.index):
            _dropout_kernel_fwd[grid](x,
                                      out,
                                      M, 
                                      p, 
                                      seed,
                                      BLOCK_SIZE=BLOCK_SIZE)
        return out.reshape(*input_shape)

@triton.autotune(
    configs=get_cuda_autotune_config(block_keys=None),
    key=['M'],
)
@triton.jit
def _dropout_kernel_bwd(DOUT,
                        DX,
                        M, 
                        p, 
                        seed,
                        BLOCK_SIZE: tl.constexpr):
    cols = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    dout = tl.load(DOUT + cols, mask=cols < M)
    rand_p = tl.rand(seed, cols)
    dx = tl.where(rand_p < p, 0.0, dout/(1-p))
    tl.store(DX + cols, dx, mask=cols < M)

def _dropout_bwd(dout, p, seed, training):
    if dout.stride(-1) != 1:
        dout = dout.contiguous()
    assert dout.stride(-1) == 1, 'expect input to be row-major'
    if (p == 0.0) or (not training):
        return dout
    elif p == 1.0:
        return torch.zeros_like(dout, memory_format=torch.contiguous_format)
    else:
        input_shape = dout.shape
        N = dout.shape[-1]
        dout = dout.reshape(-1)
        dx = torch.empty_like(dout, memory_format=torch.contiguous_format)
        assert dout.shape == dx.shape, 'expect output shape to be the same as input shape'
        assert dout.stride(-1) == 1, 'expect output to be row-major'
        M = dout.shape[0]
        grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE']),)
        MAX_FUSED_SIZE = 65536 // dout.element_size()
        BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))
        with torch.cuda.device(dout.device.index):
            _dropout_kernel_bwd[grid](dout,
                                      dx,
                                      M, 
                                      p, 
                                      seed,
                                      BLOCK_SIZE=BLOCK_SIZE)
        return dx.reshape(*input_shape)

class dropout(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, p, training):
        seed = random.randint(0, 2**16)
        output = _dropout_fwd(input, p, seed, training)
        ctx.p = p
        ctx.seed = seed
        ctx.training = training
        return output

    @staticmethod
    def backward(ctx, d_out):
        p = ctx.p
        seed = ctx.seed
        training = ctx.training
        grad = _dropout_bwd(d_out, p, seed, training)
        return grad, None, None

class Dropout(torch.nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p
        self.dropout_fn = dropout.apply
    def forward(self, x):
        return self.dropout_fn(x, self.p, self.training)