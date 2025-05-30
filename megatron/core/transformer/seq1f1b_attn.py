import torch
from dataclasses import dataclass
from typing import Dict, Any, Callable

@dataclass
class SpanInfo:
    span_idx: int
    num_span: int
    kv_cache: Dict
    seq_dim: int
    def __str__(self):
        return f"SpanInfo: span_idx={self.span_idx}, num_span={self.num_span}, seq_dim={self.seq_dim}"

def _slice(tensor: torch.Tensor, start: int, end: int, dim: int) -> torch.Tensor:
    """
    Returns a slice of the input tensor along the specified dimension.
    
    Args:
        tensor (torch.Tensor): Input tensor to be sliced.
        start (int): Start index of the slice (inclusive).
        end (int): End index of the slice (exclusive).
        dim (int): Dimension along which to perform the slice.

    Returns:
        torch.Tensor: A contiguous tensor slice of the input along the specified dimension.
    """
    slices = [slice(None)] * tensor.dim()
    slices[dim] = slice(start, end)
    return tensor[tuple(slices)]

class Seq1F1BAttn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, module: Callable, span_info: SpanInfo, query:torch.Tensor, key:torch.Tensor, value:torch.Tensor, *args):
        ctx.module = module  # Save the self reference for backward
        ctx.span_info = span_info
        ctx.args = args
        seq_dim = span_info.seq_dim
        kv_cache_pool = span_info.kv_cache
        span_idx = ctx.span_info.span_idx
        if span_idx != 0:
            k_cache, v_cache = kv_cache_pool["k_cache"], kv_cache_pool["v_cache"]
            kv_cache_pool["tensor_ref"][span_idx - 1] = (k_cache, v_cache)
            offset = k_cache.shape[seq_dim]
            key = torch.cat([k_cache, key], dim=seq_dim).detach().requires_grad_()
            value = torch.cat([v_cache, value], dim=seq_dim).detach().requires_grad_()
            k_cache.data = torch.tensor(
                [], device=key.device, dtype=key.dtype
            )
            v_cache.data = torch.tensor(
                [], device=key.device, dtype=key.dtype
            )
        else:
            kv_cache_pool["tensor_ref"] = {}
            key = key.detach().requires_grad_()
            value = value.detach().requires_grad_()
            offset = 0
        seqlen_k = key.shape[seq_dim]
        seqlen_q = query.shape[seq_dim]
        ctx._seqlen_k = seqlen_k
        ctx._seqlen_q = seqlen_q
        ctx._offset = offset
        # for 4 spans
        # [0, 128, 256, 384, 512]
        # seqlen_k: [128, 256, 384, 512], which represent the actual key/value length in compute
        # seqlen_q: [128, 128, 128, 128], which represent the actual query/output/grad_output length in compute
        # offset: [0, 128, 256, 384], which represent the position of current span in sequence
        ctx.kv_cache_pool = kv_cache_pool
        with torch.enable_grad():
            core_attn_out = module(
                query,
                key,
                value,
                *args,
            )
        kv_cache_pool["k_cache"] = key
        kv_cache_pool["v_cache"] = value
        ctx.save_for_backward(query, core_attn_out)
        return core_attn_out.clone()

    @staticmethod
    def backward(ctx, grad_output):
        query, out = ctx.saved_tensors
        span_idx = ctx.span_info.span_idx
        span_num = ctx.span_info.num_span
        seq_dim = ctx.span_info.seq_dim
        last_idx = span_idx == span_num - 1

        pk = ctx.kv_cache_pool["k_cache"].contiguous()
        pv = ctx.kv_cache_pool["v_cache"].contiguous()
        if span_idx != 0:
            ctx.kv_cache_pool["k_cache"], ctx.kv_cache_pool["v_cache"] = (
                # pop up the KV cache for the current span_idx since it is no longer needed for precede span
                _slice(pk, None, -ctx._seqlen_q, seq_dim), # pk[:, : -ctx._seqlen_q], when seq_dim == 1
                _slice(pv, None, -ctx._seqlen_q, seq_dim), # pv[:, : -ctx._seqlen_q], when seq_dim == 1
            )
        else:
            del ctx.kv_cache_pool["k_cache"], ctx.kv_cache_pool["v_cache"]

        if not last_idx:
            key, value = ctx.kv_cache_pool["tensor_ref"][span_idx]
            key.data = pk
            value.data = pv
            del ctx.kv_cache_pool["tensor_ref"][span_idx]
        else:
            key = pk
            value = pv
        dq, dk, dv = torch.autograd.grad(
            outputs=out,
            inputs=(query, key, value),
            grad_outputs=grad_output,
        )
        if not last_idx:
            k_grad_p, v_grad_p = (
                ctx.kv_cache_pool["k_grad"], 
                ctx.kv_cache_pool["v_grad"],
            )
            dk += k_grad_p
            dv += v_grad_p
        ctx.kv_cache_pool["k_grad"], ctx.kv_cache_pool["v_grad"] = (
            # rearrange the kv grad tensor 
            _slice(dk, None, ctx._offset, seq_dim).contiguous(), # dk[:, : ctx._offset], index start to current span position
            _slice(dv, None, ctx._offset, seq_dim).contiguous(), # dv[:, : ctx._offset], index start to current span position
        )
        if span_idx == 0:
            del ctx.kv_cache_pool["k_grad"]
            del ctx.kv_cache_pool["v_grad"]
        dk = _slice(dk, ctx._offset, ctx._seqlen_k, seq_dim)
        dv = _slice(dv, ctx._offset, ctx._seqlen_k, seq_dim)
        return None, None, dq, dk, dv, *[None for i in range(len(ctx.args))]
