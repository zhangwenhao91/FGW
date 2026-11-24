"""Compare TileLang DSA with native AscendC SFA on Ascend 950."""

import argparse
import json
import statistics

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument(
        "--case",
        choices=(
            "full",
            "discrete",
            "gather-diagnostic",
            "pv-diagnostic",
            "reuse-kv",
            "dma-gather",
            "dma-pipeline",
            "head-tile",
            "head64-pv",
            "kv-tile",
            "softmax-nz",
            "staged-nz",
        ),
        default="full",
    )
    parser.add_argument("--output")
    args = parser.parse_args()
    import torch_npu

    from operators.dsa.dense_tilelang import build_kernel

    torch.npu.set_device(args.device)
    print(f"torch={torch.__version__} torch_npu={torch_npu.__version__}", flush=True)
    native = getattr(torch_npu, "npu_sparse_flash_attention", None)
    if native is None:
        parser.exit(
            1, "Installed torch_npu does not expose npu_sparse_flash_attention\n"
        )

    sq, sk, h, d, tail = 1024, 2048, 128, 512, 64
    if args.case != "full":
        sk = 4096
    diagnostic = args.case == "gather-diagnostic"
    pv_diagnostic = args.case == "pv-diagnostic"
    reuse_diagnostic = args.case == "reuse-kv"
    head_diagnostic = args.case == "head-tile"
    pipeline_diagnostic = args.case in ("dma-pipeline", "head-tile")
    dma_diagnostic = args.case in ("dma-gather", "dma-pipeline", "head-tile")
    head_pv_diagnostic = args.case == "head64-pv"
    kv_diagnostic = args.case == "kv-tile"
    nz_diagnostic = args.case == "softmax-nz"
    staged_diagnostic = args.case == "staged-nz"
    if (
        diagnostic
        or pv_diagnostic
        or reuse_diagnostic
        or dma_diagnostic
        or head_pv_diagnostic
        or kv_diagnostic
        or nz_diagnostic
        or staged_diagnostic
    ):
        sq = 64
    topk = sk if args.case == "full" else 1024
    scale = (d + tail) ** -0.5
    generator = torch.Generator().manual_seed(7)
    q = torch.randn(1, sq, h, d + tail, generator=generator).to(torch.bfloat16)
    kv = torch.randn(1, sk, 1, d + tail, generator=generator).to(torch.bfloat16)
    device = args.device
    qn, kvn = q.to(device), kv.to(device)
    key_ids = torch.arange(sk, dtype=torch.int32)
    if args.case == "full":
        selected = key_ids.expand(sq, sk).contiguous()
    else:
        selected = torch.stack(
            [
                torch.randperm(sk - sq + row + 1, generator=generator)[:topk]
                .sort()
                .values
                for row in range(sq)
            ]
        ).to(torch.int32)
    indices = selected.view(1, sq, 1, topk).to(device)
    selection_mask = torch.zeros((sq, sk), dtype=torch.bool)
    selection_mask.scatter_(1, selected.long(), True)
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

    print(f"Checking native SFA output ({args.case})", flush=True)
    native_out = ascendc().cpu()
    calls = {"native_sfa": ascendc}
    outputs = {"native_sfa": native_out}
    if not (head_pv_diagnostic or kv_diagnostic or nz_diagnostic or staged_diagnostic):
        print("Compiling TileLang DSA", flush=True)
        kernel = build_kernel(sq, sk, topk)

        def tilelang_dsa():
            return kernel(
                qn[0], kvn.view(sk, d + tail), indices.view(sq, topk)
            ).unsqueeze(0)

        print("Checking TileLang DSA output", flush=True)
        tilelang_out = tilelang_dsa().cpu()
        calls["tilelang_dsa"] = tilelang_dsa
        outputs["tilelang_dsa"] = tilelang_out
    if diagnostic:
        print("Preparing contiguous KV in candidate order (untimed)", flush=True)
        packed = kv[0, :, 0][selected.long()].reshape(sq * topk, d + tail).to(device)
        packed_kernel = build_kernel(sq, sk, topk, prepacked=True)

        def tilelang_prepacked():
            return packed_kernel(qn[0], packed, indices.view(sq, topk)).unsqueeze(0)

        outputs["tilelang_prepacked"] = tilelang_prepacked().cpu()
        calls["tilelang_prepacked"] = tilelang_prepacked
    if pv_diagnostic:
        print("Compiling wide PV (256 columns per Cube call)", flush=True)
        wide_kernel = build_kernel(sq, sk, topk, pv_width=256)

        def tilelang_wide_pv():
            return wide_kernel(
                qn[0], kvn.view(sk, d + tail), indices.view(sq, topk)
            ).unsqueeze(0)

        outputs["tilelang_wide_pv"] = tilelang_wide_pv().cpu()
        calls["tilelang_wide_pv"] = tilelang_wide_pv
    if reuse_diagnostic or dma_diagnostic:
        print("Compiling shared L1 KV tile for QK and PV", flush=True)
        shared_kernel = build_kernel(sq, sk, topk, reuse_kv=True)

        def tilelang_reuse_kv():
            return shared_kernel(
                qn[0], kvn.view(sk, d + tail), indices.view(sq, topk)
            ).unsqueeze(0)

        outputs["tilelang_reuse_kv"] = tilelang_reuse_kv().cpu()
        calls["tilelang_reuse_kv"] = tilelang_reuse_kv
    if dma_diagnostic:
        print("Compiling row DMA gather with adjacent-pair coalescing", flush=True)
        dma_kernel = build_kernel(sq, sk, topk, reuse_kv=True, dma_gather=True)

        def tilelang_dma_gather():
            return dma_kernel(
                qn[0], kvn.view(sk, d + tail), indices.view(sq, topk)
            ).unsqueeze(0)

        outputs["tilelang_dma_gather"] = tilelang_dma_gather().cpu()
        calls["tilelang_dma_gather"] = tilelang_dma_gather
    if pipeline_diagnostic:
        print("Compiling double-buffered DMA pipeline", flush=True)
        pipeline_kernel = build_kernel(
            sq, sk, topk, reuse_kv=True, dma_gather=True, pipeline=True
        )

        def tilelang_dma_pipeline():
            return pipeline_kernel(
                qn[0], kvn.view(sk, d + tail), indices.view(sq, topk)
            ).unsqueeze(0)

        outputs["tilelang_dma_pipeline"] = tilelang_dma_pipeline().cpu()
        calls["tilelang_dma_pipeline"] = tilelang_dma_pipeline
    if head_diagnostic or head_pv_diagnostic:
        print("Compiling 64-head DMA prefetch candidate", flush=True)
        wide_head_kernel = build_kernel(
            sq, sk, topk, reuse_kv=True, dma_gather=True, pipeline=True, head_tile=64
        )

        def tilelang_head64():
            return wide_head_kernel(
                qn[0], kvn.view(sk, d + tail), indices.view(sq, topk)
            ).unsqueeze(0)

        outputs["tilelang_head64"] = tilelang_head64().cpu()
        calls["tilelang_head64"] = tilelang_head64
    if head_pv_diagnostic or kv_diagnostic or nz_diagnostic:
        print("Compiling 64-head candidate with 256-column PV", flush=True)
        wide_pv_kernel = build_kernel(
            sq,
            sk,
            topk,
            reuse_kv=True,
            dma_gather=True,
            pipeline=True,
            head_tile=64,
            pv_width=256,
        )

        def tilelang_head64_pv256():
            return wide_pv_kernel(
                qn[0], kvn.view(sk, d + tail), indices.view(sq, topk)
            ).unsqueeze(0)

        outputs["tilelang_head64_pv256"] = tilelang_head64_pv256().cpu()
        calls["tilelang_head64_pv256"] = tilelang_head64_pv256
    if kv_diagnostic:
        print("Compiling 64-row KV tile", flush=True)
        wide_kv_kernel = build_kernel(
            sq,
            sk,
            topk,
            reuse_kv=True,
            dma_gather=True,
            pipeline=True,
            head_tile=64,
            pv_width=256,
            kv_tile=64,
        )

        def tilelang_kv64():
            return wide_kv_kernel(
                qn[0], kvn.view(sk, d + tail), indices.view(sq, topk)
            ).unsqueeze(0)

        outputs["tilelang_kv64"] = tilelang_kv64().cpu()
        calls["tilelang_kv64"] = tilelang_kv64
    if nz_diagnostic or staged_diagnostic:
        print("Compiling direct NZ softmax writeback", flush=True)
        nz_kernel = build_kernel(
            sq,
            sk,
            topk,
            reuse_kv=True,
            dma_gather=True,
            pipeline=True,
            head_tile=64,
            pv_width=256,
            softmax_nz=True,
        )

        def tilelang_softmax_nz():
            return nz_kernel(
                qn[0], kvn.view(sk, d + tail), indices.view(sq, topk)
            ).unsqueeze(0)

        outputs["tilelang_softmax_nz"] = tilelang_softmax_nz().cpu()
        calls["tilelang_softmax_nz"] = tilelang_softmax_nz
    if staged_diagnostic:
        print("Compiling three-stage NZ pipeline", flush=True)
        staged_kernel = build_kernel(
            sq,
            sk,
            topk,
            reuse_kv=True,
            dma_gather=True,
            pipeline=True,
            head_tile=64,
            pv_width=256,
            softmax_nz=True,
            staged=True,
        )

        def tilelang_staged():
            return staged_kernel(
                qn[0], kvn.view(sk, d + tail), indices.view(sq, topk)
            ).unsqueeze(0)

        outputs["tilelang_staged"] = tilelang_staged().cpu()
        calls["tilelang_staged"] = tilelang_staged
    report = {
        "case": args.case,
        "shape": [1, sq, sk, h, 1, topk, d, tail],
        "dtype": "bfloat16",
        "torch": torch.__version__,
        "torch_npu": torch_npu.__version__,
        "accuracy": {},
    }
    key_fp32 = kv[0, :, 0].float()
    for name, out in outputs.items():
        assert out.shape == (1, sq, h, d) and out.dtype == q.dtype
        errors = []
        for start in range(0, sq, 16):
            end = min(start + 16, sq)
            scores = (q[0, start:end].float() @ key_fp32.T) * scale
            allowed = key_ids <= (torch.arange(start, end) + sk - sq)[:, None, None]
            allowed = allowed & selection_mask[start:end, None, :]
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
    )
    if "tilelang_dsa" in medians:
        report["speedup_vs_sfa"] = medians["native_sfa"] / medians["tilelang_dsa"]
    if head_pv_diagnostic:
        wide = medians["tilelang_head64_pv256"]
        report["pv256_speedup_vs_pv128"] = medians["tilelang_head64"] / wide
        report["pv256_speedup_vs_sfa"] = medians["native_sfa"] / wide
    if diagnostic:
        gather = medians["tilelang_dsa"]
        contiguous = medians["tilelang_prepacked"]
        report["prepacked_bytes"] = packed.numel() * packed.element_size()
        report["indexed_over_prepacked"] = gather / contiguous
        report["indexed_minus_prepacked_us"] = gather - contiguous
    if pv_diagnostic:
        wide = medians["tilelang_wide_pv"]
        report["wide_pv_speedup_vs_baseline"] = medians["tilelang_dsa"] / wide
        report["wide_pv_speedup_vs_sfa"] = medians["native_sfa"] / wide
    if reuse_diagnostic:
        reused = medians["tilelang_reuse_kv"]
        report["reuse_kv_speedup_vs_baseline"] = medians["tilelang_dsa"] / reused
        report["reuse_kv_speedup_vs_sfa"] = medians["native_sfa"] / reused
    if dma_diagnostic:
        dma = medians["tilelang_dma_gather"]
        report["dma_speedup_vs_simt_reuse"] = medians["tilelang_reuse_kv"] / dma
        report["dma_speedup_vs_sfa"] = medians["native_sfa"] / dma
    if pipeline_diagnostic:
        pipelined = medians["tilelang_dma_pipeline"]
        report["pipeline_speedup_vs_serial_dma"] = (
            medians["tilelang_dma_gather"] / pipelined
        )
        report["pipeline_speedup_vs_sfa"] = medians["native_sfa"] / pipelined
    if head_diagnostic:
        report["head64_speedup_vs_head32"] = (
            medians["tilelang_dma_pipeline"] / medians["tilelang_head64"]
        )
        report["head64_speedup_vs_sfa"] = (
            medians["native_sfa"] / medians["tilelang_head64"]
        )
    if kv_diagnostic:
        wide = medians["tilelang_kv64"]
        report["kv64_speedup_vs_kv32"] = medians["tilelang_head64_pv256"] / wide
        report["kv64_speedup_vs_sfa"] = medians["native_sfa"] / wide
    if nz_diagnostic:
        nz = medians["tilelang_softmax_nz"]
        report["softmax_nz_speedup_vs_baseline"] = medians["tilelang_head64_pv256"] / nz
        report["softmax_nz_speedup_vs_sfa"] = medians["native_sfa"] / nz
    if staged_diagnostic:
        staged = medians["tilelang_staged"]
        report["staged_speedup_vs_prefetch"] = medians["tilelang_softmax_nz"] / staged
        report["staged_speedup_vs_sfa"] = medians["native_sfa"] / staged
    result = json.dumps(report, indent=2)
    with open(
        args.output or f"dsa_dense_sfa_{args.case}.json", "w", encoding="utf-8"
    ) as file:
        file.write(result + "\n")
    print(result)


if __name__ == "__main__":
    main()
