"""Single-launch DSA/SFA driver for the Ascend instruction simulator."""

import argparse
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "implementation", choices=("sfa", "tilelang", "tilelang-packed")
    )
    parser.add_argument("--source", type=Path)
    parser.add_argument("--kv-tile", type=int, choices=(32, 128), default=32)
    parser.add_argument("--long-topk", action="store_true")
    args = parser.parse_args()
    if args.long_topk and args.kv_tile != 128:
        parser.error("--long-topk requires --kv-tile 128")
    packed_prob = args.implementation == "tilelang-packed"
    if packed_prob and args.kv_tile != 128:
        parser.error("tilelang-packed requires --kv-tile 128")
    import torch_npu

    sq, sk, topk, h, d, tail = 16, 256, 128, 128, 512, 64
    if args.kv_tile == 128:
        sk, topk = (4096, 1024) if args.long_topk else (512, 256)
    width = d + tail
    g = torch.Generator().manual_seed(7)
    q = torch.randn((1, sq, h, width), generator=g).to(torch.bfloat16)
    kv = torch.randn((1, sk, 1, width), generator=g).to(torch.bfloat16)
    ids = torch.stack(
        [
            torch.randperm(sk - sq + row + 1, generator=g)[:topk].sort().values
            for row in range(sq)
        ]
    ).to(torch.int32)
    if args.implementation == "sfa":
        operands = [
            x.contiguous().npu()
            for x in (
                q[..., :d],
                kv[..., :d],
                q[..., d:],
                kv[..., d:],
                ids.reshape(1, sq, 1, topk),
            )
        ]
        q_main, key, qr, kr, sparse = operands
        torch.npu.synchronize()
        out = torch_npu.npu_sparse_flash_attention(
            q_main,
            key,
            key,
            sparse,
            width**-0.5,
            query_rope=qr,
            key_rope=kr,
            sparse_block_size=1,
            layout_query="BSND",
            layout_kv="BSND",
            sparse_mode=3,
            attention_mode=2,
            return_softmax_lse=False,
        )[0]
    else:
        from operators.dsa.dense_tilelang import build_kernel

        kernel = build_kernel(
            sq,
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
            persistent=True,
            simd_output=True,
            lineinfo=True,
            kv_tile=args.kv_tile,
            packed_prob=packed_prob,
        )
        if args.source:
            args.source.write_text(kernel.get_kernel_source())
        q, kv, ids = q[0].npu(), kv.view(sk, width).npu(), ids.npu()
        torch.npu.synchronize()
        out = kernel(q, kv, ids).unsqueeze(0)
    torch.npu.synchronize()
    print(
        f"launch_completed={args.implementation} output_shape={tuple(out.shape)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
