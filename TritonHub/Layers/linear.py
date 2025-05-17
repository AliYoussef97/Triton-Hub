import torch
import triton
import triton.language as tl
import math
from TritonHub.autotune import get_cuda_autotune_config

@triton.autotune(
    configs=get_cuda_autotune_config(block_keys=['M', 'N', 'K'], include_group_size=True, include_fp8_configs=True),
    key=['M', 'N', 'K'],
)
@triton.jit
def _linear_kernel_fwd(X,
                       W,
                       Y,
                       M, 
                       N,
                       K,
                       stride_am, stride_ak,
                       stride_bk, stride_bn,
                       stride_cm, stride_cn,
                       B,
                       BLOCK_SIZE_M: tl.constexpr,
                       BLOCK_SIZE_N: tl.constexpr,
                       BLOCK_SIZE_K: tl.constexpr,
                       GROUP_SIZE_M: tl.constexpr,
                       HAS_BIAS: tl.constexpr):
        
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    X = X + (offs_am[:, None] * stride_am  + offs_k[None, :] * stride_ak)
    W = W + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    y = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        x = tl.load(X, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        w = tl.load(W, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        y = tl.dot(x, w, y)
        X += BLOCK_SIZE_K * stride_ak
        W += BLOCK_SIZE_K * stride_bk
    if HAS_BIAS:
        y += tl.load(B + offs_bn, mask=offs_bn < N)
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    Y = Y + stride_cm  * offs_cm[:, None] + stride_cn  * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(Y, y, mask=c_mask)
    
def _linear_fwd(x, weight, bias):
    if x.stride(-1) != 1:
        x = x.contiguous()
    batch_shape = x.shape[:-1]
    x = x.reshape(-1, x.shape[-1])
    assert x.stride(-1) == 1, 'expect input to be row-major'
    M, K = x.shape
    N = weight.shape[-1]
    weight = weight.contiguous()
    assert weight.stride(-1) == 1, 'expect weight to be row-major'
    out = torch.empty((M, N), dtype=x.dtype, device=x.device)
    assert out.stride(-1) == 1, 'expect output to be row-major'
    HAS_BIAS = True if bias is not None else False
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), )
    with torch.cuda.device(x.device.index):
        _linear_kernel_fwd[grid](x,
                                 weight,
                                 out,
                                 M,
                                 N,
                                 K,
                                 x.stride(0), x.stride(1),
                                 weight.stride(0), weight.stride(1),
                                 out.stride(0), out.stride(1),
                                 bias,
                                 HAS_BIAS=HAS_BIAS)                  
    return out.reshape(*batch_shape, N)

@triton.autotune(
    configs=get_cuda_autotune_config(block_keys=['M', 'N', 'K'], include_group_size=True, include_fp8_configs=True),
    key=['M', 'N', 'K'],
)
@triton.jit
def _linear_kernel_bwd_dx(DOUT, W, DX,
                          M, N, K,
                          stride_DOUT_m, stride_DOUT_n,
                          stride_W_k, stride_W_n,
                          stride_DX_m, stride_DX_k,
                          BLOCK_SIZE_M: tl.constexpr,
                          BLOCK_SIZE_N: tl.constexpr,
                          BLOCK_SIZE_K: tl.constexpr,
                          GROUP_SIZE_M: tl.constexpr):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_k = tl.cdiv(K, BLOCK_SIZE_K)
    num_pid_in_group = GROUP_SIZE_M * num_pid_k
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_k = (pid % num_pid_in_group) // group_size_m

    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bk = pid_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
    offs_n = tl.arange(0, BLOCK_SIZE_N)
    offs_k = offs_bk
    DOUT_ptr = DOUT + (offs_am[:, None] * stride_DOUT_m + offs_n[None, :] * stride_DOUT_n)
    W_ptr = W + (offs_k[None, :] * stride_W_k + offs_n[:, None] * stride_W_n)
    
    DX_acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_K), dtype=tl.float32)
    
    for n in range(0, tl.cdiv(N, BLOCK_SIZE_N)):
        g = tl.load(DOUT_ptr, mask=offs_n[None, :] < N - n * BLOCK_SIZE_N, other=0.0)
        w = tl.load(W_ptr, mask=offs_n[:, None] < N - n * BLOCK_SIZE_N, other=0.0)
        DX_acc += tl.dot(g, w)
        DOUT_ptr += BLOCK_SIZE_N * stride_DOUT_n
        W_ptr += BLOCK_SIZE_N * stride_W_n

    offs_m = offs_am[:, None]
    offs_k = offs_bk[None, :]
    DX_ptr = DX + (offs_m * stride_DX_m + offs_k * stride_DX_k)
    mask = (offs_m < M) & (offs_k < K)
    tl.store(DX_ptr, DX_acc, mask=mask)

@triton.autotune(
    configs=get_cuda_autotune_config(block_keys=['M', 'N', 'K'], include_group_size=True, include_fp8_configs=True),
    key=['M', 'N', 'K'],
)
@triton.jit
def _linear_kernel_bwd_dw(X, DOUT, DW,
                          M, N, K,
                          stride_X_m, stride_X_k,
                          stride_DOUT_m, stride_DOUT_n,
                          stride_DW_k, stride_DW_n,
                          BLOCK_SIZE_M: tl.constexpr,
                          BLOCK_SIZE_N: tl.constexpr,
                          BLOCK_SIZE_K: tl.constexpr,
                          GROUP_SIZE_M: tl.constexpr):
    pid = tl.program_id(0)
    num_pid_k = tl.cdiv(K, BLOCK_SIZE_K)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_k = group_id * GROUP_SIZE_M
    group_size_k = min(num_pid_k - first_pid_k, GROUP_SIZE_M)
    pid_k = first_pid_k + ((pid % num_pid_in_group) % group_size_k)
    pid_n = (pid % num_pid_in_group) // group_size_k

    offs_k = pid_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_m = tl.arange(0, BLOCK_SIZE_M)
    
    X_ptr = X + (offs_m[None, :] * stride_X_m + offs_k[:, None] * stride_X_k)
    DOUT_ptr = DOUT + (offs_m[:, None] * stride_DOUT_m + offs_n[None, :] * stride_DOUT_n)
    
    DW_acc = tl.zeros((BLOCK_SIZE_K, BLOCK_SIZE_N), dtype=tl.float32)
    
    for m in range(0, tl.cdiv(M, BLOCK_SIZE_M)):
        a = tl.load(X_ptr, mask=offs_m[None, :] < M - m * BLOCK_SIZE_M, other=0.0)
        b = tl.load(DOUT_ptr, mask=offs_m[:, None] < M - m * BLOCK_SIZE_M, other=0.0)
        DW_acc += tl.dot(a, b)
        X_ptr += BLOCK_SIZE_M * stride_X_m
        DOUT_ptr += BLOCK_SIZE_M * stride_DOUT_m
    
    offs_k_out = offs_k[:, None]
    offs_n_out = offs_n[None, :]
    DW_ptr = DW + (offs_k_out * stride_DW_k + offs_n_out * stride_DW_n)
    mask = (offs_k_out < K) & (offs_n_out < N)
    tl.store(DW_ptr, DW_acc, mask=mask)

@triton.autotune(
    configs=get_cuda_autotune_config(block_keys=['M', 'N'], include_group_size=True, include_fp8_configs=True),
    key=['M', 'N'],
)
@triton.jit
def _linear_kernel_bwd_db(DOUT, DB,
                          M, N,
                          stride_DOUT_m, stride_DOUT_n,
                          BLOCK_SIZE_M: tl.constexpr,
                          BLOCK_SIZE_N: tl.constexpr):
    pid = tl.program_id(0)
    offs_n = pid * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_m = tl.arange(0, BLOCK_SIZE_M)
    
    DB_acc = tl.zeros((BLOCK_SIZE_N,), dtype=tl.float32)
    
    for m in range(0, tl.cdiv(M, BLOCK_SIZE_M)):
        offs_am = m * BLOCK_SIZE_M + offs_m
        DOUT_ptr = DOUT + (offs_am[:, None] * stride_DOUT_m + offs_n[None, :] * stride_DOUT_n)
        g = tl.load(DOUT_ptr, mask=(offs_am[:, None] < M) & (offs_n[None, :] < N), other=0.0)
        DB_acc += tl.sum(g, axis=0)
    tl.store(DB + offs_n, DB_acc, mask=offs_n < N)

def _linear_bwd(x, dout, weight, bias):
    batch_shape = x.shape[:-1]
    x = x.reshape(-1, x.shape[-1])
    M, K = x.shape
    assert x.stride(-1) == 1, 'expect input to be row-major'
    dout = dout.reshape(-1, dout.shape[-1])
    if dout.stride(-1) != 1:
        dout = dout.contiguous()
    assert dout.stride(-1) == 1, 'expect output to be row-major'
    N = weight.shape[-1]
    dx = torch.empty_like(x, memory_format=torch.contiguous_format, dtype=x.dtype)
    dw = torch.empty_like(weight, memory_format=torch.contiguous_format, dtype=weight.dtype)
    
    grid_dx = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(K, META['BLOCK_SIZE_K']),)
    grid_dw = lambda META: (triton.cdiv(K, META['BLOCK_SIZE_K']) * triton.cdiv(N, META['BLOCK_SIZE_N']),)

    # Compute grad_input
    with torch.cuda.device(x.device.index):
        _linear_kernel_bwd_dx[grid_dx](dout, weight, dx,
                                       M, N, K,
                                       dout.stride(0), dout.stride(1),
                                       weight.stride(0), weight.stride(1),
                                       dx.stride(0), dx.stride(1))

        # Compute grad_weight
        _linear_kernel_bwd_dw[grid_dw](x, dout, dw,
                                       M, N, K,
                                       x.stride(0), x.stride(1),
                                       dout.stride(0), dout.stride(1),
                                       dw.stride(0), dw.stride(1))

        # Compute grad_bias if bias is not None
        if bias is not None:
            db = torch.empty_like(bias, memory_format=torch.contiguous_format, dtype=bias.dtype)
            grid_db = lambda META: (triton.cdiv(N, META['BLOCK_SIZE_N']),)
            _linear_kernel_bwd_db[grid_db](dout, db,
                                           M, N,
                                           dout.stride(0), dout.stride(1),)
        else:
            db = None
    return dx.reshape(*batch_shape, K), dw, db
 
class linear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias):
        output = _linear_fwd(input, weight, bias)
        ctx.save_for_backward(input, weight, bias)
        return output

    @staticmethod
    def backward(ctx, d_out):
        input, weight, bias = ctx.saved_tensors
        grad, dw, db = _linear_bwd(input, d_out, weight, bias)
        return grad, dw, db

class Linear(torch.nn.Module):
    def __init__(self, in_features, out_features, bias=True, device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.Parameter(torch.empty(in_features, out_features, **factory_kwargs))
        if bias:
            self.bias = torch.nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        self.linear_fn = linear.apply
    
    def reset_parameters(self) -> None:
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            torch.nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        return self.linear_fn(x, self.weight, self.bias)