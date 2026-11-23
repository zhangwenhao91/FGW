"""Compare the DSA correctness baseline with native AscendC SFA on Ascend 950."""

import argparse
import json
import statistics
from functools import partial

import torch

from operators.dsa.dense import dsa_dense_fwd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--output", default="dsa_dense_sfa.json")
    args = parser.parse_args()
    import torch_npu

    torch.npu.set_device(args.device)
    print(f"torch={torch.__version__} torch_npu={torch_npu.__version__}", flush=True)
    native = getattr(torch_npu, "npu_sparse_flash_attention", None)
    if native is None:
        parser.exit(
            1, "Installed torch_npu does not expose npu_sparse_flash_attention\n"
        )

    sq, sk, h, d, tail = 1024, 2048, 128, 512, 64
    scale = (d + tail) ** -0.5
    generator = torch.Generator().manual_seed(7)
    q = torch.randn(1, sq, h, d + tail, generator=generator).to(torch.bfloat16)
    kv = torch.randn(1, sk, 1, d + tail, generator=generator).to(torch.bfloat16)
    device = args.device
    qn, kvn = q.to(device), kv.to(device)
    key_ids = torch.arange(sk, dtype=torch.int32)
    # Supply the same ascending full candidate list. Right-aligned causality
    # reduces the list to [0, sk-sq+row] on both paths.
    indices = key_ids.view(1, 1, 1, sk).expand(1, sq, 1, sk).contiguous().to(device)
    query, key = qn[..., :d].contiguous(), kvn[..., :d].contiguous()
    query_rope, key_rope = qn[..., d:].contiguous(), kvn[..., d:].contiguous()

    def ascendc():
        return native(
            query,
            key,
            key,
            indices,
            scale,
            query_rope=query_rope,
            key_rope=key_rope,
            sparse_block_size=1,
            layout_query="BSND",
            layout_kv="BSND",
            sparse_mode=3,
            attention_mode=2,
            return_softmax_lse=False,
        )[0]

    baseline = partial(
        dsa_dense_fwd, qn, kvn, indices, dim_tail=tail, q_start_index_s=sk - sq
    )
    print("Checking native SFA output", flush=True)
    native_out = ascendc().cpu()
    print("Checking PyTorch NPU baseline output", flush=True)
    baseline_out = baseline().cpu()
    report = {
        "shape": [1, sq, sk, h, 1, sk, d, tail],
        "dtype": "bfloat16",
        "torch": torch.__version__,
        "torch_npu": torch_npu.__version__,
        "accuracy": {},
    }
    key_fp32 = kv[0, :, 0].float()
    for name, out in (
        ("native_sfa", native_out),
        ("torch_npu_composite", baseline_out),
    ):
        assert out.shape == (1, sq, h, d) and out.dtype == q.dtype
        errors = []
        for start in range(0, sq, 16):
            end = min(start + 16, sq)
            scores = (q[0, start:end].float() @ key_fp32.T) * scale
            allowed = key_ids <= (torch.arange(start, end) + sk - sq)[:, None, None]
            ref = (
                scores.masked_fill(~allowed, -torch.inf).softmax(-1) @ key_fp32[:, :d]
            ).to(q.dtype)
            actual = out[0, start:end]
            torch.testing.assert_close(actual, ref, atol=0.012, rtol=0.012)
            errors.append((actual.float() - ref.float()).abs().max().item())
        report["accuracy"][name] = {"passed": True, "max_abs_error": max(errors)}

    def measure(call):
        call()
        torch.npu.synchronize()
        begin = torch.npu.Event(enable_timing=True)
        end = torch.npu.Event(enable_timing=True)
        begin.record()
        call()
        end.record()
        end.synchronize()
        return begin.elapsed_time(end) * 1000

    print("Accuracy passed. Starting alternating Event measurements", flush=True)
    calls = {"native_sfa": ascendc, "torch_npu_composite": baseline}
    samples = {name: [] for name in calls}
    for repeat in range(6):
        order = list(calls)[:: (-1 if repeat % 2 else 1)]
        for name in order:
            samples[name].append(measure(calls[name]))
        print(f"timing {repeat + 1}/6", flush=True)
    medians = {name: statistics.median(values) for name, values in samples.items()}
    report.update(
        samples_us=samples,
        median_us=medians,
        composite_latency_over_sfa=medians["torch_npu_composite"]
        / medians["native_sfa"],
    )
    result = json.dumps(report, indent=2)
    with open(args.output, "w", encoding="utf-8") as file:
        file.write(result + "\n")
    print(result)


if __name__ == "__main__":
    main()
