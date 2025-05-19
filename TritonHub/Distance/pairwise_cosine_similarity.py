import triton
import triton.language as tl
import torch
from TritonHub.Ops.norm import _norm_fwd_kernel
from TritonHub.Ops.normalize import _normalize_fwd_kernel
from TritonHub.Ops.batched_matmul import _batched_matmul_fwd_kernel
from TritonHub.autotune import get_cuda_autotune_config

def _cos_sim_fwd(x, y, eps=1e-6):
    if x.stride(-1) != 1:
        x = x.contiguous()
    if y.stride(-1) != 1:
        y = y.contiguous()
    B, M, K = x.shape
    _, N, K_ = y.shape
    assert K == K_, "Input dimension mismatch"
    assert x.dtype == y.dtype, "Input tensors must have the same dtype"
    x_norm = torch.empty((B, M, K), memory_format=torch.contiguous_format, device=x.device, dtype=x.dtype)
    y_norm = torch.empty((B, N, K), memory_format=torch.contiguous_format, device=x.device, dtype=x.dtype)
    norms_x = torch.empty((B, M), memory_format=torch.contiguous_format, device=x.device, dtype=x.dtype)
    norms_y = torch.empty((B, N), memory_format=torch.contiguous_format, device=x.device, dtype=x.dtype)
    out = torch.empty((B, M, N), memory_format=torch.contiguous_format, device=x.device, dtype=x.dtype)
    assert out.stride(-1) == 1 and x.stride(-1) == 1 and y.stride(-1) == 1, "Output tensors must be contiguous"
    dtype = (tl.bfloat16 if x.dtype == torch.bfloat16 else (tl.float16 if x.dtype == torch.float16 else tl.float32))
    grid_norm_x = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']), B)
    grid_norm_y = lambda META: (triton.cdiv(N, META['BLOCK_SIZE_M']), B)
    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_SIZE_M'])*
        triton.cdiv(N, META['BLOCK_SIZE_N']),
        B,
    )
    with torch.cuda.device(x.device.index):
        _norm_fwd_kernel[grid_norm_x](x, x.stride(0), x.stride(1), x.stride(2),
                                      norms_x, norms_x.stride(0), norms_x.stride(1),
                                      eps, 2, M, K, dtype=dtype)
        _norm_fwd_kernel[grid_norm_y](y, y.stride(0), y.stride(1), y.stride(2),
                                      norms_y, norms_y.stride(0), norms_y.stride(1),
                                      eps, 2, N, K, dtype=dtype)
        _normalize_fwd_kernel[grid_norm_x](x, x.stride(0), x.stride(1), x.stride(2),
                                           norms_x, norms_x.stride(0), norms_x.stride(1),
                                           x_norm, x_norm.stride(0), x_norm.stride(1), x_norm.stride(2),
                                           M, K, dtype=dtype)
        _normalize_fwd_kernel[grid_norm_y](y, y.stride(0), y.stride(1), y.stride(2),
                                           norms_y, norms_y.stride(0), norms_y.stride(1),
                                           y_norm, y_norm.stride(0), y_norm.stride(1), y_norm.stride(2),
                                           N, K, dtype=dtype)
        _batched_matmul_fwd_kernel[grid](x_norm, x_norm.stride(0), x_norm.stride(1), x_norm.stride(2),
                                         y_norm, y_norm.stride(0), y_norm.stride(1), y_norm.stride(2),
                                         out, out.stride(0), out.stride(1), out.stride(2),
                                         M, N, K, dtype=dtype)
    return out, x_norm, y_norm, norms_x, norms_y

@triton.autotune(
    configs=get_cuda_autotune_config(block_keys=['M', 'N', 'K'], include_extra_configs=True),
    key=['M', 'N', 'K'])
@triton.jit
def _cos_sim_bwd_kernel(out_ptr, stride_outb, stride_outm, stride_outn, 
                        x_norm_ptr, stride_xnormb, stride_xnormm, stride_xnormk,
                        y_norm_ptr, stride_ynormb, stride_ynormn, stride_ynormk,
                        norms_ptr, stride_normsb, stride_normsxr,
                        dout_ptr, stride_doutb, stride_doutm, stride_doutn,
                        din_ptr, stride_dinb, stride_dinr, stride_dink,
                        M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
                        BLOCK_SIZE_K: tl.constexpr, dtype: tl.constexpr, compute_dx: tl.constexpr):
        pid_b = tl.program_id(1)
        pid = tl.program_id(0)
        
        if compute_dx:
            main_dim = M
            aux_dim = N
            MAIN_BLOCK:tl.constexpr = BLOCK_SIZE_M
            AUX_BLOCK:tl.constexpr = BLOCK_SIZE_N
        else:
            main_dim = N
            aux_dim = M
            MAIN_BLOCK:tl.constexpr = BLOCK_SIZE_N
            AUX_BLOCK:tl.constexpr = BLOCK_SIZE_M
        
        num_pid_main = tl.cdiv(main_dim, MAIN_BLOCK)
        pid_main = pid % num_pid_main
        pid_k = pid // num_pid_main
    
        offs_main = pid_main * MAIN_BLOCK + tl.arange(0, MAIN_BLOCK)
        offs_k = pid_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)

        x_norm_ptr += pid_b * stride_xnormb
        y_norm_ptr += pid_b * stride_ynormb
        out_ptr += pid_b * stride_outb
        dout_ptr += pid_b * stride_doutb
        din_ptr += pid_b * stride_dinb

        norms_ptr += pid_b * stride_normsb + offs_main * stride_normsxr
        norms = tl.load(norms_ptr, mask=offs_main < main_dim, other=0.0)
    
        term1 = tl.zeros((MAIN_BLOCK, BLOCK_SIZE_K), dtype=tl.float32)
        term2 = tl.zeros((MAIN_BLOCK,), dtype=tl.float32)

        for aux in range(0, tl.cdiv(aux_dim, AUX_BLOCK)):
            offs_aux = aux * AUX_BLOCK + tl.arange(0, AUX_BLOCK)

            if compute_dx:
                other_norm_ptr = y_norm_ptr + offs_aux[:, None] * stride_ynormn + offs_k[None, :] * stride_ynormk
                out_ptrs = out_ptr + offs_main[:, None] * stride_outm + offs_aux[None, :] * stride_outn
                mask_out = (offs_main[:, None] < main_dim) & (offs_aux[None, :] < aux_dim)
                dout_ptrs = dout_ptr + offs_main[:, None] * stride_doutm + offs_aux[None, :] * stride_doutn
                mask_dout = (offs_main[:, None] < main_dim) & (offs_aux[None, :] < aux_dim)
            else:
                other_norm_ptr = x_norm_ptr + offs_aux[:, None] * stride_xnormm + offs_k[None, :] * stride_xnormk
                out_ptrs = out_ptr + offs_aux[:, None] * stride_outm + offs_main[None, :] * stride_outn
                mask_out = (offs_aux[:, None] < aux_dim) & (offs_main[None, :] < main_dim)
                dout_ptrs = dout_ptr + offs_aux[:, None] * stride_doutm + offs_main[None, :] * stride_doutn
                mask_dout = (offs_aux[:, None] < aux_dim) & (offs_main[None, :] < main_dim)
            
            other_norm = tl.load(other_norm_ptr, mask=(offs_aux[:, None] < aux_dim) & (offs_k[None, :] < K), other=0.0).to(dtype)
            out = tl.load(out_ptrs, mask=mask_out, other=0.0).to(dtype)
            dout = tl.load(dout_ptrs, mask=mask_dout, other=0.0).to(dtype)

            term1 += tl.dot(dout if compute_dx else dout.trans(1, 0), other_norm, allow_tf32=True)
            term2 += tl.sum(dout * out, axis=1 if compute_dx else 0)
        
        if compute_dx:
            main_norm_ptrs = x_norm_ptr + offs_main[:, None] * stride_xnormm + offs_k[None, :] * stride_xnormk
        else:
            main_norm_ptrs = y_norm_ptr + offs_main[:, None] * stride_ynormn + offs_k[None, :] * stride_ynormk
        main_norm = tl.load(main_norm_ptrs, mask=(offs_main[:, None] < main_dim) & (offs_k[None, :] < K), other=0.0).to(dtype)

        din = (term1 - main_norm * term2[:, None]) / norms[:, None]
        din = din.to(dtype)

        din_ptrs = din_ptr + offs_main[:, None] * stride_dinr + offs_k[None, :] * stride_dink
        tl.store(din_ptrs, din, mask=(offs_main[:, None] < main_dim) & (offs_k[None, :] < K))

def _cos_sim_bwd(out, x_norm, y_norm, norms_x, norms_y, dout):
    if out.stride(-1) != 1:
        out = out.contiguous()
    if x_norm.stride(-1) != 1:
        x_norm = x_norm.contiguous()
    if y_norm.stride(-1) != 1:
        y_norm = y_norm.contiguous()
    if norms_x.stride(-1) != 1:
        norms_x = norms_x.contiguous()
    if norms_y.stride(-1) != 1:
        norms_y = norms_y.contiguous()
    if dout.stride(-1) != 1:
        dout = dout.contiguous()
    B, M, K = x_norm.shape
    _, N, K_ = y_norm.shape
    assert K == K_, "Input dimension mismatch"
    assert out.shape == (B, M, N), "Output shape mismatch"
    assert dout.shape == (B, M, N), "Output gradient shape mismatch"
    assert out.dtype == x_norm.dtype == y_norm.dtype == norms_x.dtype == norms_y.dtype == dout.dtype, "All tensors must have the same dtype"
    dx = torch.empty((B, M, K), memory_format=torch.contiguous_format, device=x_norm.device, dtype=x_norm.dtype)
    dy = torch.empty((B, N, K), memory_format=torch.contiguous_format, device=y_norm.device, dtype=y_norm.dtype)
    assert dx.stride(-1) == 1 and dy.stride(-1) == 1, "Output tensors must be contiguous"
    dtype = (tl.bfloat16 if x_norm.dtype == torch.bfloat16 else (tl.float16 if x_norm.dtype == torch.float16 else tl.float32))
    grid_x = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(K, META['BLOCK_SIZE_K']), B)
    grid_y = lambda META: (triton.cdiv(N, META['BLOCK_SIZE_N']) * triton.cdiv(K, META['BLOCK_SIZE_K']), B)
    _cos_sim_bwd_kernel[grid_x](out, out.stride(0), out.stride(1), out.stride(2),
                                x_norm, x_norm.stride(0), x_norm.stride(1), x_norm.stride(2),
                                y_norm, y_norm.stride(0), y_norm.stride(1), y_norm.stride(2),
                                norms_x, norms_x.stride(0), norms_x.stride(1),
                                dout, dout.stride(0), dout.stride(1), dout.stride(2),
                                dx, dx.stride(0), dx.stride(1), dx.stride(2),
                                M, N, K, dtype=dtype, compute_dx=True)
    _cos_sim_bwd_kernel[grid_y](out, out.stride(0), out.stride(1), out.stride(2),
                                x_norm, x_norm.stride(0), x_norm.stride(1), x_norm.stride(2),
                                y_norm, y_norm.stride(0), y_norm.stride(1), y_norm.stride(2),
                                norms_y, norms_y.stride(0), norms_y.stride(1),
                                dout, dout.stride(0), dout.stride(1), dout.stride(2),
                                dy, dy.stride(0), dy.stride(1), dy.stride(2),
                                M, N, K, dtype=dtype, compute_dx=False)
    return dx, dy

class CosineSimilarity(torch.autograd.Function):
    @staticmethod
    @torch.amp.custom_fwd(device_type='cuda')
    def forward(ctx, x, y, eps=1e-6):
        assert len(x.shape) == 3 and len(y.shape) == 3, "Expected 3D (B, M, D) and 3D (B, N, D) tensors, add batch dim inputs are 2D"
        output, x_norm, y_norm, norms_x, norms_y = _cos_sim_fwd(x, y, eps)
        ctx.save_for_backward(output, x_norm, y_norm, norms_x, norms_y)
        return output

    @staticmethod
    @torch.amp.custom_bwd(device_type='cuda')
    def backward(ctx, grad_output):
        output, x_norm, y_norm, norms_x, norms_y = ctx.saved_tensors
        grad_x, grad_y = _cos_sim_bwd(output, x_norm, y_norm, norms_x, norms_y, grad_output)
        return grad_x, grad_y, None

class cosine_similarity:
    def __init__(self, eps=1e-6):
        self.eps = eps
        self.cos_sim = CosineSimilarity.apply
    def __call__(self, x, y):
        return self.cos_sim(x, y, self.eps)