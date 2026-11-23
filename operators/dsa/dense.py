"""Correctness-first DSA dense-cache forward with bounded NPU intermediates."""

import math

import torch


def dsa_dense_fwd(
    q, kv, indices, *, dim_tail, stride_kv=1, q_start_index_s, sm_scale=None, query_chunk=8
):
    """Causal BSHD sparse attention using PyTorch operators on the input device.

    Indices name cache rows. The cache length is the padding sentinel.
    Repeated indices count once, as in the TileOPs reference.
    """
    if q.ndim != 4 or kv.ndim != 4 or indices.ndim != 4:
        raise ValueError("q, kv and indices must be rank four")
    b, s, h, width = q.shape
    bk, sk, g, kw = kv.shape
    if min(b, s, h, width, sk, g, indices.shape[-1]) < 1:
        raise ValueError("empty input dimensions are unsupported")
    if (bk, kw) != (b, width) or h % g or indices.shape[:3] != (b, s, g):
        raise ValueError("incompatible BSHD and index shapes")
    if q.dtype not in (torch.float16, torch.bfloat16) or kv.dtype != q.dtype:
        raise ValueError("q and kv must have the same FP16 or BF16 dtype")
    if q.device != kv.device or q.device != indices.device or indices.dtype != torch.int32:
        raise ValueError("inputs must share a device and indices must be int32")
    if not 0 <= dim_tail < width or stride_kv < 1 or query_chunk < 1:
        raise ValueError("invalid tail, stride or chunk")
    if not (q_start_index_s > stride_kv or (q_start_index_s == 0 and stride_kv == 1)):
        raise ValueError("query start outside the TileOPs manifest")
    scale = width**-0.5 if sm_scale is None else sm_scale
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("sm_scale must be finite and positive")

    d = width - dim_tail
    cache = kv.transpose(1, 2).contiguous().view(b * g * sk, width)
    offsets = torch.arange(b * g, device=q.device, dtype=torch.int64).view(b, 1, g, 1) * sk
    output = q.new_empty((b, s, h, d))
    for start in range(0, s, query_chunk):
        stop = min(start + query_chunk, s)
        index = indices[:, start:stop].sort(dim=-1).values
        positions = torch.arange(start, stop, device=q.device) + q_start_index_s
        valid = (index >= 0) & (index < sk)
        valid &= index < ((positions + 1) // stride_kv).view(1, -1, 1, 1)
        valid[..., 1:] &= index[..., 1:] != index[..., :-1]
        # The TileOPs reference adds row zero for these initial local queries.
        prefix = positions - q_start_index_s < stride_kv - 1
        valid &= ~(prefix.view(1, -1, 1, 1) & (index == 0))
        zero = torch.zeros_like(index[..., :1])
        index = torch.cat((index.masked_fill(~valid, 0), zero), dim=-1)
        valid = torch.cat((valid, prefix.view(1, -1, 1, 1).expand_as(zero)), dim=-1)
        rows = cache.index_select(0, (index.long() + offsets).flatten())
        rows = rows.view(b, stop - start, g, -1, width).float()
        query = q[:, start:stop].reshape(b, stop - start, g, h // g, width).float()
        score = (query @ rows.transpose(-1, -2)) * scale
        score = score.masked_fill(~valid.unsqueeze(-2), float("-inf"))
        score = score.masked_fill(~valid.any(-1)[..., None, None], 0.0)
        prob = score.softmax(-1).masked_fill(~valid.unsqueeze(-2), 0.0)
        result = prob @ rows[..., :d]
        output[:, start:stop] = result.reshape(b, stop - start, h, d).to(q.dtype)
    return output
