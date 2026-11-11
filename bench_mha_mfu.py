"""Measure current paged-MHA prototype using warmed NPU event samples.

MFU.md policy: count useful QK+PV FLOPs only; normalize to input-dtype
dense Cube peak. The current kernel executes FP32 SIMT, NOT Cube MMA.
"""
import argparse
import datetime
import json
import math
import importlib
import pathlib
import statistics
import traceback


# name, batch, queries, heads, dimension, page size, logical KV length
CASES = [
    ("decode_b1_k512", 1, 1, 16, 128, 128, 512),
    ("decode_b2_k2048", 2, 1, 8, 64, 128, 2048),
    ("spec_b2_q4_k2048", 2, 4, 8, 128, 128, 2048),
    ("decode_b8_k2048", 8, 1, 8, 64, 128, 2048),
    ("decode_b1_k8192", 1, 1, 8, 64, 256, 8192),
]


def useful_flops(lengths, sq, heads, dim, causal):
    pairs = sum(max(length - sq + q + 1, 0) if causal else length
                for length in lengths for q in range(sq))
    return 4 * heads * dim * pairs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", nargs="+", choices=[c[0] for c in CASES])
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=31)
    parser.add_argument("--cube-peak-tflops", type=float, default=378.0,
                        help="MFU.md 28-AIC FP16/BF16 dense Cube peak; override for another configuration")
    parser.add_argument("--cache", choices=["warm", "flush"], default="warm")
    parser.add_argument("--flush-mb", type=int, default=256)
    parser.add_argument("--impl", choices=["v0", "v1", "v2", "v3"], default="v0")
    parser.add_argument("--baseline-report", help="Optional v0/v1 JSON report for matched-case speedup")
    parser.add_argument("--output", default=None)
    parser.add_argument("--list-only", action="store_true")
    args = parser.parse_args()
    if args.output is None:
        args.output = "mha_mfu_report.json" if args.impl == "v0" else f"mha_mfu_{args.impl}_report.json"
    if args.repeat < 3 or args.warmup < 1 or not math.isfinite(args.cube_peak_tflops) or args.cube_peak_tflops <= 0 or args.flush_mb <= 0:
        parser.error("repeat >= 3, warmup >= 1, and finite positive peak/flush size required")
    cases = [c for c in CASES if not args.cases or c[0] in args.cases]
    if args.list_only:
        print(json.dumps(cases, indent=2))
        return 0
    baseline = {}
    if args.baseline_report:
        if pathlib.Path(args.baseline_report).resolve() == pathlib.Path(args.output).resolve():
            parser.error("Output must not overwrite baseline report")
        base = json.loads(pathlib.Path(args.baseline_report).read_text(encoding="utf-8"))
        if base.get("status") != "passed":
            parser.error("Baseline report must have passed")
        settings = base["settings"]
        baseline_impl = settings.get("impl", "v0")
        if baseline_impl not in ("v0", "v1", "v2"):
            parser.error("Baseline implementation must be v0, v1 or v2")
        for key in ("cache", "flush_mb", "cube_peak_tflops", "repeat", "warmup", "device"):
            if settings.get(key) != getattr(args, key):
                parser.error(f"Baseline setting mismatch: {key}")
        baseline = {(r["case"], r["dtype"], r["causal"]): r for r in base["results"]}
    report = {"status": "running", "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "settings": vars(args), "results": [],
              "flops_policy": "4*H*D*sum_bq(visible_keys); QK and PV only; excludes softmax/rescaling",
              "peak_source": "MFU.md: 28 AIC, dense FP16/BF16 Cube 378 TFLOPS (user-supplied specification)",
              "execution": "FP32 SIMT Vector; Cube-normalized MFU is not actual Vector hardware utilization",
              "timing": "Median of individual NPU event intervals; compile, H2D, reference, warmup, cache flush excluded. Direct compiled-kernel callable, including stream-visible launch gaps/output allocation effects; not host wall time."}
    path = pathlib.Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)

    def save():
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        tmp.replace(path)

    save()
    try:
        import torch
        import torch_npu
        import tilelang
        from mha_paged_decode_ascend_v0 import reference
        impl_module = importlib.import_module(f"mha_paged_decode_ascend_{args.impl}")
        build_kernel = impl_module.build_kernel
        if args.impl == "v3":
            report["execution"] = "BF16 D128 page%128=0: Cube QK/PV + SIMD softmax (query duplicated to 128 rows); other cases: v2 SIMT fallback. Inspect execution_path per result. Effective FLOPs exclude duplicated query work."
        torch.npu.set_device(args.device)
        device = f"npu:{args.device}"
        report["versions"] = {"torch": torch.__version__, "torch_npu": torch_npu.__version__, "tilelang": tilelang.__version__}
        report["device_name"] = torch.npu.get_device_name(args.device)
        props = torch.npu.get_device_properties(args.device)
        report["device_properties"] = str(props)
        core_count = getattr(props, "cube_core_num", None)
        if core_count is not None and core_count != 28 and args.cube_peak_tflops == 378.0:
            raise ValueError("Default peak requires 28 Cube cores; supply the correct --cube-peak-tflops")
        flush = torch.empty(args.flush_mb * 1024 * 1024, dtype=torch.uint8, device=device) if args.cache == "flush" else None
        for name, b, sq, h, d, p, length in cases:
            width = (length + p - 1) // p
            # Separate, randomly ordered physical pages per request.
            generator = torch.Generator().manual_seed(7)
            table = torch.randperm(b * width, generator=generator).reshape(b, width).to(torch.int32)
            lengths = torch.full((b,), length, dtype=torch.int32)
            for dtype in (torch.float16, torch.bfloat16):
                q = torch.randn(b, sq, h, d, generator=generator).to(dtype)
                k = torch.randn(b * width * p, h, d, generator=generator).to(dtype)
                v = torch.randn(b * width * p, h, d, generator=generator).to(dtype)
                inputs = [x.to(device) for x in (q, k, v, table, lengths)]
                for causal in (False, True):
                    row = {"case": name, "B": b, "Sq": sq, "H": h, "D": d,
                           "page_size": p, "lengths": lengths.tolist(), "dtype": str(dtype),
                           "causal": causal, "status": "running"}
                    report["results"].append(row)
                    row["execution_path"] = impl_module.route(dtype, d, p) if args.impl == "v3" else args.impl
                    save()
                    extra = {"device_index": args.device} if args.impl != "v0" else {}
                    kernel = build_kernel(b, sq, h, d, k.shape[0], width, p, causal,
                                          "float16" if dtype == torch.float16 else "bfloat16", **extra)
                    got = kernel(*inputs).cpu()
                    expected = reference(q, k, v, table, lengths, p, causal)
                    rtol, atol = (1e-2, 4e-3) if dtype == torch.bfloat16 else (2e-3, 2e-3)
                    torch.testing.assert_close(got, expected, rtol=rtol, atol=atol)
                    for _ in range(args.warmup):
                        kernel(*inputs)
                    torch.npu.synchronize()
                    # Create events before timing; one invocation per sample.
                    samples = []
                    start = torch.npu.Event(enable_timing=True)
                    end = torch.npu.Event(enable_timing=True)
                    for _ in range(args.repeat):
                        if flush is not None:
                            flush.zero_()
                            torch.npu.synchronize()
                        start.record()
                        output = kernel(*inputs)
                        end.record()
                        end.synchronize()
                        ms = start.elapsed_time(end)
                        if not math.isfinite(ms) or ms <= 0:
                            raise ValueError(f"Invalid NPU event time: {ms}")
                        samples.append(ms)
                    latency = statistics.median(samples)
                    flops = useful_flops(lengths.tolist(), sq, h, d, causal)
                    achieved = flops / (latency * 1e-3) / 1e12
                    row.update(status="passed", correctness="passed", samples_ms=samples,
                               median_ms=latency, min_ms=min(samples), max_ms=max(samples),
                               effective_flops=flops, effective_tflops=achieved,
                               cube_peak_tflops=args.cube_peak_tflops,
                               cube_normalized_mfu_percent=100 * achieved / args.cube_peak_tflops)
                    previous = baseline.get((name, str(dtype), causal))
                    if previous is not None:
                        if previous["effective_flops"] != flops:
                            raise ValueError("Baseline effective FLOPs mismatch")
                        row.update(baseline_median_ms=previous["median_ms"], baseline_implementation=baseline_impl,
                                   speedup_vs_baseline=previous["median_ms"] / latency)
                        row[f"speedup_vs_{baseline_impl}"] = previous["median_ms"] / latency
                    save()
                    print(f"{name} {dtype} causal={causal}: {latency:.6f} ms, "
                          f"{achieved:.8f} TFLOPS, MFU={row['cube_normalized_mfu_percent']:.8f}% "
                          f"speedup_vs_baseline={row.get('speedup_vs_baseline', 'n/a')}", flush=True)
        report["status"] = "passed"
    except Exception:
        report.update(status="error", error=traceback.format_exc())
        if report["results"] and report["results"][-1]["status"] == "running":
            report["results"][-1].update(status="error", error=report["error"])
        traceback.print_exc()
    finally:
        save()
    print(f"{report['status'].upper()}; report={path}", flush=True)
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
