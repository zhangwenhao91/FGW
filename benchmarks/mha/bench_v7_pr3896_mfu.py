"""Warm-cache MFU: v7 production shapes and matched single-head PR3896 cases."""
import argparse
import datetime
import functools
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import statistics
import sys
import traceback

Q_LENGTHS = (1, 4, 16, 32, 64, 127, 128, 129, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768)
ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--q-lengths", nargs="+", type=int, default=list(Q_LENGTHS))
    parser.add_argument("--kv", type=int, default=2048)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--repeat", type=int, default=51)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--peak-tflops", type=float, default=378.0)
    parser.add_argument("--list-only", action="store_true")
    parser.add_argument("--output", default="mha_v7_pr3896_mfu.json")
    args = parser.parse_args()
    if (min(args.q_lengths) < 1 or args.kv < 128 or args.kv % 128 or args.rounds < 1
            or args.repeat < 3 or args.warmup < 1 or not math.isfinite(args.peak_tflops) or args.peak_tflops <= 0):
        parser.error("positive Q, 128-aligned KV, rounds>=1, repeat>=3, warmup>=1, positive finite peak required")
    cases = [(track, b, h, q) for track, b, h in (("v7_production", 2, 8), ("matched_single_head", 1, 1))
             for q in args.q_lengths]
    if args.list_only:
        print(json.dumps(cases, indent=2))
        return 0
    report = {"status": "running", "settings": vars(args), "results": [],
        "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "upstream": json.loads((ROOT / "third_party/tilelang_pr3896/SOURCE.json").read_text()),
        "flops_policy": "4*B*H*Q*KV*128, noncausal useful QK+PV only; no padding/softmax FLOPs",
        "timing": "Full native callable, warm-cache NPU Event median, one call per sample; compilation, H2D, CPU reference and upstream dense KV preparation excluded. v7 includes Q conversion and merge. PR uses prepared contiguous 2D Q/K/V, no paging/batch/head adapter in timed call.",
        "comparison": "Only matched_single_head has identical B/H/Q/KV and values. Production B2/H8 is v7 only; PR native API cannot batch independent heads. PR has dynamic Q/KV, default auto schedule/stages3; original example FP32 output plus core-supported BF16 output. v7 is static, manual/stages2, BF16 output. All input BF16/D128/page128, noncausal, seed7."}
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    summary_path = path.with_name(path.stem + "_summary.json")

    def save():
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        tmp.replace(path)
        compact = {k: v for k, v in report.items() if k != "results"}
        compact["results"] = [{k: v for k, v in row.items() if k != "samples_ms"} for row in report["results"]]
        summary_path.write_text(json.dumps(compact, indent=2) + "\n", encoding="utf-8")

    save()
    try:
        import torch
        import torch_npu
        import tilelang
        import tilelang.ascend.language as T
        from operators.mha.kernels import mha_paged_decode_ascend_v7 as v7
        from benchmarks.mha.bench_mha_mfu import chunked_cpu_reference
        torch.npu.set_device(args.device)
        device = torch.device("npu", args.device)
        props = torch.npu.get_device_properties(args.device)
        if getattr(props, "cube_core_num", 28) != 28 and args.peak_tflops == 378.0:
            raise ValueError("378 TFLOPS assumes 28 Cube cores; override --peak-tflops")
        report["device"] = str(props)
        report["versions"] = {"torch": torch.__version__, "torch_npu": torch_npu.__version__, "tilelang": tilelang.__version__}
        core_path = ROOT / "third_party/tilelang_pr3896/core.py"
        for name, source in report["upstream"]["files"].items():
            assert hashlib.sha256((core_path.parent/name).read_bytes()).hexdigest() == source["sha256"], name
        report["v7_sha256"] = hashlib.sha256(Path(v7.__file__).read_bytes()).hexdigest()
        spec = importlib.util.spec_from_file_location("fgw_pr3896_core", core_path)
        core = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = core
        spec.loader.exec_module(core)
        upstream_cache = {}

        def upstream(dtype):
            if dtype not in upstream_cache:
                try:
                    program = core.flash_attention_fwd(q_len=T.dynamic("q_len"), kv_len=T.dynamic("kv_len"),
                        head_dim=128, out_dtype=dtype)
                    upstream_cache[dtype] = tilelang.compile(program, out_idx=-1,
                        pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True})
                except Exception:
                    upstream_cache[dtype] = traceback.format_exc()
            value = upstream_cache[dtype]
            if isinstance(value, str):
                raise RuntimeError(value)
            return value

        for track, b, h, sq in cases:
            report["active_case"] = {"track": track, "B": b, "H": h, "Q": sq, "KV": args.kv}
            save()
            print(f"Preparing {track} B={b} H={h} Q={sq} KV={args.kv}", flush=True)
            gen = torch.Generator().manual_seed(7)
            width = args.kv // 128
            table = torch.randperm(b * width, generator=gen).reshape(b, width).int()
            lengths = torch.full((b,), args.kv, dtype=torch.int32)
            q = torch.randn(b, sq, h, 128, generator=gen).bfloat16()
            k = torch.randn(b * args.kv, h, 128, generator=gen).bfloat16()
            v = torch.randn(b * args.kv, h, 128, generator=gen).bfloat16()
            expected = chunked_cpu_reference(q, k, v, table, lengths, 128, False, 128)
            inputs = [x.to(device) for x in (q, k, v, table, lengths)]
            programs = {}
            correctness = {}
            variants = ("v7_manual_bf16",) if track == "v7_production" else ("v7_manual_bf16", "pr3896_fp32", "pr3896_bf16")
            if track == "matched_single_head":
                rows = (table[0, :, None].long() * 128 + torch.arange(128)[None, :]).reshape(-1)
                dense = [q[0, :, 0].contiguous().to(device), k[rows, 0].contiguous().to(device), v[rows, 0].contiguous().to(device)]
            for variant in variants:
                try:
                    if variant == "v7_manual_bf16":
                        kernel = v7.build_kernel(b, sq, h, 128, k.shape[0], width, 128, False,
                            "bfloat16", args.device, schedule_mode="manual")
                        call = functools.partial(kernel, *inputs)
                        got = call().cpu()
                        kernel_mode = kernel.execution_mode
                    else:
                        dtype = "float32" if variant.endswith("fp32") else "bfloat16"
                        call = functools.partial(upstream(dtype), *dense)
                        got = call().cpu().reshape(1, sq, 1, 128)
                        kernel_mode = "original_dynamic_single_head"
                    torch.testing.assert_close(got.float(), expected.float(), rtol=1e-2, atol=4e-3)
                    correctness[variant] = {"max_abs_error": (got.float()-expected.float()).abs().max().item(),
                        "rtol": 1e-2, "atol": 4e-3}
                    programs[variant] = (call, kernel_mode)
                except Exception:
                    report["results"].append({"track": track, "variant": variant, "B": b, "H": h,
                        "Q": sq, "KV": args.kv, "status": "error", "error": traceback.format_exc()})
                    traceback.print_exc()
                    save()
            for round_id in range(args.rounds):
                order = list(programs) if round_id % 2 == 0 else list(reversed(programs))
                for _ in range(args.warmup):
                    for label in order:
                        programs[label][0]()
                torch.npu.synchronize()
                samples = {label: [] for label in order}
                start, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
                for sample in range(args.repeat):
                    offset = sample % max(len(order), 1)
                    rotated = order[offset:] + order[:offset]
                    if sample % 2:
                        rotated.reverse()
                    for label in rotated:
                        start.record()
                        output = programs[label][0]()
                        end.record()
                        end.synchronize()
                        ms = start.elapsed_time(end)
                        if not math.isfinite(ms) or ms <= 0:
                            raise ValueError(f"Invalid Event time {ms}")
                        samples[label].append(ms)
                for label in order:
                    median = statistics.median(samples[label])
                    flops = 4 * b * h * sq * args.kv * 128
                    tflops = flops / (median * 1e-3) / 1e12
                    row = {"track": track, "variant": label, "B": b, "H": h, "Q": sq, "KV": args.kv,
                        "D": 128, "round": round_id+1, "status": "passed", "correctness": "passed",
                        "accuracy": correctness[label],
                        "input_dtype": "bfloat16", "output_dtype": "float32" if label.endswith("fp32") else "bfloat16",
                        "kernel_mode": programs[label][1], "median_us": median*1000,
                        "speedup_vs_pr_bf16": (statistics.median(samples["pr3896_bf16"])/median
                            if "pr3896_bf16" in samples and label == "v7_manual_bf16" else None),
                        "effective_flops": flops, "effective_tflops": tflops,
                        "mfu_percent": 100*tflops/args.peak_tflops, "samples_ms": samples[label]}
                    report["results"].append(row)
                    print(f"{track} Q={sq} {label} round={round_id+1}: {median*1000:.3f} us MFU={row['mfu_percent']:.4f}%", flush=True)
                save()
            del programs, inputs
        report.pop("active_case", None)
        report["status"] = "passed" if all(r["status"] == "passed" for r in report["results"]) else "partial_error"
    except Exception:
        report.update(status="error", error=traceback.format_exc())
        traceback.print_exc()
    finally:
        save()
    print(f"{report['status']}: {path}; compact report: {summary_path}", flush=True)
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
