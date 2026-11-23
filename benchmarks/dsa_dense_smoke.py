"""Run with python -m benchmarks.dsa_dense_smoke --device npu:0."""
import argparse

import torch

from operators.dsa.dense import dsa_dense_fwd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.device.startswith("npu"):
        import torch_npu  # noqa: F401
    torch.manual_seed(7)
    for dtype in (torch.float16, torch.bfloat16):
        for stride, start, tail in ((1, 0, 0), (1, 3, 64), (3, 5, 64)):
            q = torch.randn(2, 5, 8, 128 + tail).to(dtype)
            kv = torch.randn(2, 9, 2, 128 + tail).to(dtype)
            ids = torch.randint(0, 10, (2, 5, 2, 7), dtype=torch.int32)
            ids[:, 0] = 9
            ids[..., 1] = ids[..., 0]
            ids[..., -1] = 9
            selected = torch.zeros(2, 5, 2, 10, dtype=torch.bool)
            selected.scatter_(-1, ids.long(), True)
            mask = selected[..., :9] & (
                torch.arange(9) * stride + stride - 1
                <= (torch.arange(5) + start).view(1, 5, 1, 1)
            )
            mask[:, : stride - 1, :, 0] = True
            scores = torch.einsum(
                "bsghd,bngd->bsghn", q.float().reshape(2, 5, 2, 4, -1), kv.float()
            ) * q.shape[-1] ** -0.5
            probs = scores.masked_fill(~mask.unsqueeze(-2), -torch.inf).softmax(-1).nan_to_num()
            expected = torch.einsum("bsghn,bngd->bsghd", probs, kv[..., :128].float())
            got = dsa_dense_fwd(
                q.to(args.device), kv.to(args.device), ids.to(args.device),
                dim_tail=tail, stride_kv=stride, q_start_index_s=start, query_chunk=2,
            )
            torch.testing.assert_close(got.cpu(), expected.reshape_as(got).to(dtype), atol=0.012, rtol=0.012)
            print(f"passed {dtype} stride={stride} tail={tail}", flush=True)


if __name__ == "__main__":
    main()
