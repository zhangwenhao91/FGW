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
            "selective-zero",
            "simd-softmax",
            "simd-pv",
            "simd-pv-long-full",
            "simd-pv-long-discrete",
            "batched-pv-long",
            "pv512-long",
            "persistent-long",
            "native-sfa-long",
            "simd-output-long",
            "kv64-simd-long",
            "kv128-simd-long",
            "pv-fma-long",
            "kv-nz-diagnostic",
            "direct-kv-nz",
            "gather-once",
            "rolling-gather",
        ),
        default="full",
    )
    parser.add_argument("--output")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument(
        "--profile-dir",
        help="Capture each selected implementation in a separate NPU trace",
    )
    args = parser.parse_args()
    profile_cases = ("staged-nz", "simd-softmax", "simd-pv", "simd-pv-long-discrete")
    profile_cases += (
        "batched-pv-long",
        "pv512-long",
        "kv-nz-diagnostic",
        "direct-kv-nz",
    )
    profile_cases += (
        "persistent-long",
        "simd-output-long",
        "native-sfa-long",
        "kv64-simd-long",
    )
    profile_cases += ("kv128-simd-long", "pv-fma-long")
    if args.profile_dir and args.case not in profile_cases:
        parser.error("--profile-dir requires one of: " + ", ".join(profile_cases))
    import torch_npu

    native_only = args.case == "native-sfa-long"
    if not native_only:
        from operators.dsa.dense_tilelang import build_kernel

    torch.npu.set_device(args.device)
    print(f"torch={torch.__version__} torch_npu={torch_npu.__version__}", flush=True)
    native = getattr(torch_npu, "npu_sparse_flash_attention", None)
    if native is None:
        parser.exit(
            1, "Installed torch_npu does not expose npu_sparse_flash_attention\n"
        )

    sq, sk, h, d, tail = 1024, 2048, 128, 512, 64
    kv_nz_diagnostic = args.case == "kv-nz-diagnostic"
    direct_nz = args.case == "direct-kv-nz"
    rolling = args.case == "rolling-gather"
    gather_once = args.case in ("gather-once", "rolling-gather")
    short_pv512 = kv_nz_diagnostic or direct_nz or gather_once
    pv_fma = args.case == "pv-fma-long"
    kv128_simd = args.case == "kv128-simd-long" or pv_fma
    kv64_simd = args.case == "kv64-simd-long" or kv128_simd
    simd_output = args.case == "simd-output-long" or kv64_simd
    persistent = args.case == "persistent-long" or simd_output
    wide_pv_long = args.case == "pv512-long" or persistent or short_pv512
    batched_pv = args.case == "batched-pv-long" or wide_pv_long
    long_pv = batched_pv or args.case in ("simd-pv-long-full", "simd-pv-long-discrete")
    full_indices = args.case in ("full", "simd-pv-long-full")
    if not full_indices:
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
    zero_diagnostic = args.case == "selective-zero"
    simd_pv_diagnostic = args.case == "simd-pv" or long_pv
    simd_diagnostic = simd_pv_diagnostic or args.case == "simd-softmax"
    if (
        diagnostic
        or pv_diagnostic
        or reuse_diagnostic
        or dma_diagnostic
        or head_pv_diagnostic
        or kv_diagnostic
        or nz_diagnostic
        or staged_diagnostic
        or zero_diagnostic
        or simd_diagnostic
    ):
        sq = 1024 if long_pv and not short_pv512 else 64
    topk = sk if full_indices else 1024
    scale = (d + tail) ** -0.5
    generator = torch.Generator().manual_seed(7)
    q = torch.randn(1, sq, h, d + tail, generator=generator).to(torch.bfloat16)
    kv = torch.randn(1, sk, 1, d + tail, generator=generator).to(torch.bfloat16)
    device = args.device
    qn, kvn = q.to(device), kv.to(device)
    key_ids = torch.arange(sk, dtype=torch.int32)
    if full_indices:
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
    if not (
        native_only
        or head_pv_diagnostic
        or kv_diagnostic
        or nz_diagnostic
        or staged_diagnostic
        or zero_diagnostic
        or simd_diagnostic
    ):
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
    if nz_diagnostic or (staged_diagnostic and not args.profile_dir):
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
    if staged_diagnostic or zero_diagnostic or (simd_diagnostic and not long_pv):
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
    if zero_diagnostic:
        print("Compiling selective KV zero fill", flush=True)
        zero_kernel = build_kernel(
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
            selective_zero=True,
        )

        def tilelang_selective_zero():
            return zero_kernel(
                qn[0], kvn.view(sk, d + tail), indices.view(sq, topk)
            ).unsqueeze(0)

        outputs["tilelang_selective_zero"] = tilelang_selective_zero().cpu()
        calls["tilelang_selective_zero"] = tilelang_selective_zero

        print("Checking mixed padding and fully masked query", flush=True)
        padded = indices.view(sq, topk).clone()
        padded[:, -17:] = sk
        padded[0, :] = sk
        padded_ref = staged_kernel(qn[0], kvn.view(sk, d + tail), padded).cpu()
        padded_out = zero_kernel(qn[0], kvn.view(sk, d + tail), padded).cpu()
        torch.testing.assert_close(padded_out, padded_ref, atol=0.002, rtol=0.002)
        torch.testing.assert_close(
            padded_out[0], torch.zeros_like(padded_out[0]), atol=0, rtol=0
        )
    if simd_diagnostic and not (batched_pv or (long_pv and args.profile_dir)):
        print("Compiling SIMD softmax with direct NZ stores", flush=True)
        simd_kernel = build_kernel(
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
            simd_softmax=True,
        )

        def tilelang_simd_softmax():
            return simd_kernel(
                qn[0], kvn.view(sk, d + tail), indices.view(sq, topk)
            ).unsqueeze(0)

        outputs["tilelang_simd_softmax"] = tilelang_simd_softmax().cpu()
        calls["tilelang_simd_softmax"] = tilelang_simd_softmax

        if not long_pv:
            padded = indices.view(sq, topk).clone()
            padded[:, -17:] = sk
            padded[0, :] = sk
            padded_ref = staged_kernel(qn[0], kvn.view(sk, d + tail), padded).cpu()
            padded_out = simd_kernel(qn[0], kvn.view(sk, d + tail), padded).cpu()
            torch.testing.assert_close(padded_out, padded_ref, atol=0.002, rtol=0.002)
            torch.testing.assert_close(
                padded_out[0], torch.zeros_like(padded_out[0]), atol=0, rtol=0
            )
    if simd_pv_diagnostic and not wide_pv_long:
        print("Compiling SIMD PV accumulation", flush=True)
        pv_kernel = build_kernel(
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
            simd_softmax=True,
            simd_pv=True,
        )

        def tilelang_simd_pv():
            return pv_kernel(
                qn[0], kvn.view(sk, d + tail), indices.view(sq, topk)
            ).unsqueeze(0)

        outputs["tilelang_simd_pv"] = tilelang_simd_pv().cpu()
        calls["tilelang_simd_pv"] = tilelang_simd_pv
        if not long_pv:
            padded_pv = pv_kernel(qn[0], kvn.view(sk, d + tail), padded).cpu()
            torch.testing.assert_close(padded_pv, padded_ref, atol=0.002, rtol=0.002)
            torch.testing.assert_close(
                padded_pv[0], torch.zeros_like(padded_pv[0]), atol=0, rtol=0
            )
    if batched_pv and not (short_pv512 or persistent):
        print("Compiling one SIMD accumulation per KV tile", flush=True)
        batched_kernel = build_kernel(
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
            simd_softmax=True,
            simd_pv=True,
            batch_pv=True,
        )

        def tilelang_batched_pv():
            return batched_kernel(
                qn[0], kvn.view(sk, d + tail), indices.view(sq, topk)
            ).unsqueeze(0)

        outputs["tilelang_batched_pv"] = tilelang_batched_pv().cpu()
        calls["tilelang_batched_pv"] = tilelang_batched_pv
    if wide_pv_long and not kv64_simd:
        print("Compiling one 512-column PV and one FIXPIPE per KV tile", flush=True)
        wide_kernel = build_kernel(
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
        )

        def tilelang_pv512():
            return wide_kernel(
                qn[0], kvn.view(sk, d + tail), indices.view(sq, topk)
            ).unsqueeze(0)

        outputs["tilelang_pv512"] = tilelang_pv512().cpu()
        calls["tilelang_pv512"] = tilelang_pv512
    if kv_nz_diagnostic:
        print("Packing KV into per-query NZ slabs outside timing", flush=True)
        # [query, tile, AIV, row, C0-group, C0] -> compact NZ with one pad row.
        gathered = kv[0, :, 0][selected.long()].reshape(sq, topk // 32, 2, 16, 36, 16)
        packed_nd = gathered.view(sq * topk, d + tail).to(device)
        packed_nd_kernel = build_kernel(
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
            packed_nd=True,
        )

        def tilelang_packed_kv_nd():
            return packed_nd_kernel(qn[0], packed_nd, indices.view(sq, topk)).unsqueeze(
                0
            )

        outputs["tilelang_packed_kv_nd"] = tilelang_packed_kv_nd().cpu()
        calls["tilelang_packed_kv_nd"] = tilelang_packed_kv_nd
        slabs = torch.zeros((sq, topk // 32, 2, 36, 17, 16), dtype=kv.dtype)
        slabs[..., :16, :] = gathered.transpose(3, 4)
        packed_nz = slabs.view(-1, 36 * 17 * 16).to(device)
        packed_nz_kernel = build_kernel(
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
            packed_nz=True,
        )

        def tilelang_packed_kv_nz():
            return packed_nz_kernel(qn[0], packed_nz, indices.view(sq, topk)).unsqueeze(
                0
            )

        outputs["tilelang_packed_kv_nz"] = tilelang_packed_kv_nz().cpu()
        calls["tilelang_packed_kv_nz"] = tilelang_packed_kv_nz
    if direct_nz:
        print("Compiling indexed strided DMA directly into NZ", flush=True)
        direct_kernel = build_kernel(
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
            direct_nz=True,
        )
        flat_kv = kvn.view(sk, d + tail)

        def tilelang_direct_kv_nz():
            return direct_kernel(qn[0], flat_kv, indices.view(sq, topk)).unsqueeze(0)

        outputs["tilelang_direct_kv_nz"] = tilelang_direct_kv_nz().cpu()
        calls["tilelang_direct_kv_nz"] = tilelang_direct_kv_nz
        padded = indices.view(sq, topk).clone()
        padded[:, -17:] = sk
        padded[0, :] = sk
        padded_ref = wide_kernel(qn[0], flat_kv, padded).cpu()
        padded_out = direct_kernel(qn[0], flat_kv, padded).cpu()
        torch.testing.assert_close(padded_out, padded_ref, atol=0.002, rtol=0.002)
        torch.testing.assert_close(
            padded_out[0], torch.zeros_like(padded_out[0]), atol=0, rtol=0
        )
    if gather_once:
        from operators.dsa.gather_once import build_gather

        print("Compiling shared KV gather and packed ND attention", flush=True)
        gather_kernel = build_gather(sq, sk, topk)
        packed_kernel = build_kernel(
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
            packed_nd=True,
        )
        workspace = torch.empty((sq * topk, d + tail), dtype=kv.dtype, device=device)
        flat_kv = kvn.view(sk, d + tail)
        flat_indices = indices.view(sq, topk)

        def collect():
            gather_kernel(flat_kv, flat_indices, workspace)

        def attend():
            return packed_kernel(qn[0], workspace, flat_indices).unsqueeze(0)

        def shared_kv():
            collect()
            return attend()

        outputs["tilelang_gather_once"] = shared_kv().cpu()
        expected_workspace = kv[0, :, 0][selected.long()].reshape_as(workspace)
        torch.testing.assert_close(workspace.cpu(), expected_workspace, rtol=0, atol=0)
        calls["tilelang_gather_once"] = shared_kv
        calls["gather_only"] = collect
        calls["packed_attention_only"] = attend
    if rolling:
        from benchmarks.dsa_rolling import build_rolling

        print("Compiling 16-query double-buffered KV gather", flush=True)
        serial_chunks, overlapped_chunks, rolling_bytes = build_rolling(
            qn[0], flat_kv, flat_indices, chunk=16
        )
        outputs["tilelang_rolling_overlap"] = overlapped_chunks().cpu()
        outputs["tilelang_rolling_serial"] = serial_chunks().cpu()
        calls["tilelang_rolling_serial"] = serial_chunks
        calls["tilelang_rolling_overlap"] = overlapped_chunks
    if persistent and not kv64_simd:
        print("Compiling fixed-28-core indexed PV512", flush=True)
        persistent_kernel = build_kernel(
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
        )

        def tilelang_persistent():
            return persistent_kernel(
                qn[0], kvn.view(sk, d + tail), indices.view(sq, topk)
            ).unsqueeze(0)

        outputs["tilelang_persistent"] = tilelang_persistent().cpu()
        calls["tilelang_persistent"] = tilelang_persistent
    if simd_output and not kv128_simd:
        print("Compiling SIMD normalization and DMA output", flush=True)
        output_kernel = build_kernel(
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
        )

        def tilelang_simd_output():
            return output_kernel(
                qn[0], kvn.view(sk, d + tail), indices.view(sq, topk)
            ).unsqueeze(0)

        outputs["tilelang_simd_output"] = tilelang_simd_output().cpu()
        calls["tilelang_simd_output"] = tilelang_simd_output
    if kv64_simd and not pv_fma:
        print("Compiling SIMD KV64 with two PV K slices and one FIXPIPE", flush=True)
        wide_kv = build_kernel(
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
            kv_tile=64,
        )

        def tilelang_kv64_simd():
            return wide_kv(
                qn[0], kvn.view(sk, d + tail), indices.view(sq, topk)
            ).unsqueeze(0)

        outputs["tilelang_kv64_simd"] = tilelang_kv64_simd().cpu()
        calls["tilelang_kv64_simd"] = tilelang_kv64_simd
    if kv128_simd:
        print("Compiling KV128: two-stage pipeline and four PV K slices", flush=True)
        kv128_kernel = build_kernel(
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
            kv_tile=128,
        )

        def tilelang_kv128_simd():
            return kv128_kernel(
                qn[0], kvn.view(sk, d + tail), indices.view(sq, topk)
            ).unsqueeze(0)

        outputs["tilelang_kv128_simd"] = tilelang_kv128_simd().cpu()
        calls["tilelang_kv128_simd"] = tilelang_kv128_simd
    if pv_fma:
        print("Compiling KV128 with fused SIMD PV multiply-add", flush=True)
        fma_kernel = build_kernel(
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
            kv_tile=128,
            pv_fma=True,
        )

        def tilelang_pv_fma():
            return fma_kernel(
                qn[0], kvn.view(sk, d + tail), indices.view(sq, topk)
            ).unsqueeze(0)

        outputs["tilelang_pv_fma"] = tilelang_pv_fma().cpu()
        calls["tilelang_pv_fma"] = tilelang_pv_fma
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

    if args.verify_only:
        print(json.dumps(report, indent=2))
        return

    if args.profile_dir and not native_only:
        from benchmarks.dsa_profile import capture

        if long_pv and not batched_pv:
            calls = {name: calls[name] for name in ("native_sfa", "tilelang_simd_pv")}
        capture(
            calls,
            args.profile_dir,
            case=args.case,
            shape=report["shape"],
            dtype=report["dtype"],
        )
        return

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
    if zero_diagnostic:
        selective = medians["tilelang_selective_zero"]
        report["padding_checks_passed"] = True
        report["selective_zero_speedup_vs_staged"] = (
            medians["tilelang_staged"] / selective
        )
        report["selective_zero_speedup_vs_sfa"] = medians["native_sfa"] / selective
    if simd_diagnostic and not batched_pv:
        simd = medians["tilelang_simd_softmax"]
        if not long_pv:
            report["padding_checks_passed"] = True
            report["simd_speedup_vs_staged"] = medians["tilelang_staged"] / simd
        report["simd_speedup_vs_sfa"] = medians["native_sfa"] / simd
    if simd_pv_diagnostic and not wide_pv_long:
        pv = medians["tilelang_simd_pv"]
        if not batched_pv:
            report["simd_pv_speedup_vs_simd_softmax"] = (
                medians["tilelang_simd_softmax"] / pv
            )
        report["simd_pv_speedup_vs_sfa"] = medians["native_sfa"] / pv
    if batched_pv and not (short_pv512 or persistent):
        batched = medians["tilelang_batched_pv"]
        if not wide_pv_long:
            report["batched_pv_speedup_vs_simd_pv"] = (
                medians["tilelang_simd_pv"] / batched
            )
        report["batched_pv_speedup_vs_sfa"] = medians["native_sfa"] / batched
    if wide_pv_long and not (short_pv512 or persistent):
        wide = medians["tilelang_pv512"]
        report["pv512_speedup_vs_batched_pv256"] = medians["tilelang_batched_pv"] / wide
        report["pv512_speedup_vs_sfa"] = medians["native_sfa"] / wide
    if kv_nz_diagnostic:
        nz = medians["tilelang_packed_kv_nz"]
        nd = medians["tilelang_packed_kv_nd"]
        report["packed_nd_bytes"] = packed_nd.numel() * packed_nd.element_size()
        report["packed_nd_speedup_vs_indexed_pv512"] = medians["tilelang_pv512"] / nd
        report["packed_nz_speedup_vs_packed_nd"] = nd / nz
        report["packed_nz_bytes"] = packed_nz.numel() * packed_nz.element_size()
        report["packed_nz_speedup_vs_indexed_pv512"] = medians["tilelang_pv512"] / nz
        report["packed_nz_speedup_vs_sfa"] = medians["native_sfa"] / nz
    if direct_nz:
        nz = medians["tilelang_direct_kv_nz"]
        report["padding_checks_passed"] = True
        report["direct_nz_speedup_vs_indexed_pv512"] = medians["tilelang_pv512"] / nz
        report["direct_nz_speedup_vs_sfa"] = medians["native_sfa"] / nz
    if gather_once:
        shared = medians["tilelang_gather_once"]
        report["workspace_bytes"] = workspace.numel() * workspace.element_size()
        report["workspace_exact"] = True
        report["gather_once_speedup_vs_indexed_pv512"] = (
            medians["tilelang_pv512"] / shared
        )
        report["gather_once_speedup_vs_sfa"] = medians["native_sfa"] / shared
    if rolling:
        overlap = medians["tilelang_rolling_overlap"]
        report["rolling_workspace_bytes"] = rolling_bytes
        report["rolling_overlap_speedup_vs_serial_chunks"] = (
            medians["tilelang_rolling_serial"] / overlap
        )
        report["rolling_overlap_speedup_vs_full_gather"] = (
            medians["tilelang_gather_once"] / overlap
        )
        report["rolling_overlap_speedup_vs_indexed"] = (
            medians["tilelang_pv512"] / overlap
        )
    if persistent and not kv64_simd:
        fixed = medians["tilelang_persistent"]
        report["persistent_speedup_vs_indexed_pv512"] = (
            medians["tilelang_pv512"] / fixed
        )
        report["persistent_speedup_vs_sfa"] = medians["native_sfa"] / fixed
    if simd_output and not kv64_simd:
        new = medians["tilelang_simd_output"]
        report["simd_output_speedup_vs_persistent"] = (
            medians["tilelang_persistent"] / new
        )
        report["simd_output_speedup_vs_sfa"] = medians["native_sfa"] / new
    if kv64_simd and not kv128_simd:
        wide = medians["tilelang_kv64_simd"]
        report["kv64_speedup_vs_kv32"] = medians["tilelang_simd_output"] / wide
        report["kv64_speedup_vs_sfa"] = medians["native_sfa"] / wide
    if kv128_simd and not pv_fma:
        wide = medians["tilelang_kv128_simd"]
        report["kv128_speedup_vs_kv64"] = medians["tilelang_kv64_simd"] / wide
        report["kv128_speedup_vs_sfa"] = medians["native_sfa"] / wide
    if pv_fma:
        fused = medians["tilelang_pv_fma"]
        report["pv_fma_speedup_vs_kv128"] = medians["tilelang_kv128_simd"] / fused
        report["pv_fma_speedup_vs_sfa"] = medians["native_sfa"] / fused
    result = json.dumps(report, indent=2)
    with open(
        args.output or f"dsa_dense_sfa_{args.case}.json", "w", encoding="utf-8"
    ) as file:
        file.write(result + "\n")
    print(result)
    if args.profile_dir and native_only:
        from benchmarks.dsa_profile import capture

        capture(
            calls,
            args.profile_dir,
            case=args.case,
            shape=report["shape"],
            dtype=report["dtype"],
        )


if __name__ == "__main__":
    main()
