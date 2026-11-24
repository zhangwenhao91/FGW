"""Rolling shared-KV diagnostic for prevalidated, causally visible indices."""

import torch

from operators.dsa.dense_tilelang import build_kernel
from operators.dsa.gather_once import build_gather


def build_rolling(q, kv, indices, *, chunk):
    """Build matching serial and overlapped calls for the discrete benchmark.

    The benchmark generates only visible indices. Local chunk kernels may
    admit a wider causal prefix, so these calls must not take raw indices.
    The caller stream owns the output and waits on every required gather.
    """
    sq, topk = indices.shape
    sk, width = kv.shape
    if sq % chunk or sq < 2 * chunk:
        raise ValueError("expected at least two whole query chunks")
    gather = build_gather(chunk, sk, topk)
    attention = build_kernel(
        chunk,
        sk,
        topk,
        reuse_kv=True,
        dma_gather=True,
        pipeline=True,
        head_tile=64,
        pv_width=512,
        softmax_nz=True,
        staged=True,
        simd_softmax=True,
        simd_pv=True,
        batch_pv=True,
        packed_nd=True,
    )
    inputs = tuple(zip(q.split(chunk), indices.split(chunk), strict=True))
    buffers = [kv.new_empty((chunk * topk, width)) for _ in range(2)]
    gather_stream = torch.npu.Stream(device=q.device)

    def serial():
        outputs = []
        for step, (query, ids) in enumerate(inputs):
            stage = buffers[step % 2]
            gather(kv, ids, stage)
            outputs.append(attention(query, stage, ids))
        return torch.cat(outputs).unsqueeze(0)

    def overlap():
        caller = torch.npu.current_stream(q.device)
        gather_stream.wait_stream(caller)
        released = [None, None]
        outputs = []
        for step, (query, ids) in enumerate(inputs):
            slot = step % 2
            stage = buffers[slot]
            with torch.npu.stream(gather_stream):
                if released[slot] is not None:
                    gather_stream.wait_event(released[slot])
                gather(kv, ids, stage)
                ready = gather_stream.record_event()
            caller.wait_event(ready)
            outputs.append(attention(query, stage, ids))
            released[slot] = caller.record_event()
        return torch.cat(outputs).unsqueeze(0)

    return serial, overlap, sum(buf.numel() * buf.element_size() for buf in buffers)
