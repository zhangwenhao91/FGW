"""Compare automatic split-KV with forced split=1 in current v7 on identical inputs."""
import argparse
import datetime
import functools
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import traceback

Q_LENGTHS = (1, 4, 16, 32, 64, 127, 128, 129, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--q-lengths", nargs="+", type=int, default=list(Q_LENGTHS))
    parser.add_argument("--kv", type=int, default=2048)
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--repeat", type=int, default=51)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--peak-tflops", type=float, default=378.0)
    parser.add_argument("--list-only", action="store_true")
    parser.add_argument("--output", default="mha_v7_split1_paired.json")
    args = parser.parse_args()
    if (min(args.q_lengths) < 1 or args.kv < 128 or args.kv % 128 or args.rounds < 1
            or args.repeat < 3 or args.warmup < 1 or not math.isfinite(args.peak_tflops) or args.peak_tflops <= 0):
        parser.error("positive Q, 128-aligned KV, rounds>=1, repeat>=3, warmup>=1, positive finite peak required")
    cases = [(track, b, h, q) for track, b, h in (("batch2_head8", 2, 8), ("batch1_head1", 1, 1))
             for q in args.q_lengths]
    if args.list_only:
        print(json.dumps(cases, indent=2))
        return 0
    report = {"status": "running", "settings": vars(args), "results": [],
        "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "baseline": "Current v7 automatic split selection; candidate forces num_splits=1, all other tuning identical",
        "speedup_policy": "speedup_split1_over_auto = automatic median latency / forced split1 median latency; >1 means forced split1 faster",
        "flops_policy": "4*B*H*Q*KV*128, noncausal useful QK+PV only; no padding/softmax FLOPs",
        "timing": "Full native v7 callable, warm-cache NPU Event median; one invocation per sample. Includes Q conversion and output allocation plus direct kernel or partial+merge. Excludes compile, H2D, CPU reference and metadata preparation. Not CPU wall time.",
        "comparison": "Same B/H/Q/KV, BF16 Q/K/V/O, D128, page128, random page table, per-request lengths, noncausal, scale=128**-0.5, seed7. Both manual/stages2/Q resident. Forced split1 switches to existing direct normalization/cast/store path; this measures the total strategy change, not isolated merge latency. Default remains automatic."}
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
        from operators.mha.kernels import mha_paged_decode_ascend_v7 as v7
        from benchmarks.mha.bench_mha_mfu import chunked_cpu_reference
        torch.npu.set_device(args.device)
        device = torch.device("npu", args.device)
        props = torch.npu.get_device_properties(args.device)
        if getattr(props, "cube_core_num", 28) != 28 and args.peak_tflops == 378.0:
            raise ValueError("378 TFLOPS assumes 28 Cube cores; override --peak-tflops")
        report["device"] = str(props)
        report["versions"] = {"torch": torch.__version__, "torch_npu": torch_npu.__version__, "tilelang": tilelang.__version__}
        report["v7_sha256"] = hashlib.sha256(Path(v7.__file__).read_bytes()).hexdigest()
        report["cann_environment"] = {key: os.environ.get(key) for key in
            ("ASCEND_HOME_PATH", "ASCEND_TOOLKIT_HOME", "ASCEND_OPP_PATH", "ASCEND_RT_VISIBLE_DEVICES")}
        save()

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
            variants = ("v7_auto_split", "v7_split1")
            geometry = {}
            for variant in variants:
                try:
                    kernel = v7.build_kernel(b, sq, h, 128, k.shape[0], width, 128, False,
                        "bfloat16", args.device, schedule_mode="manual",
                        num_splits=1 if variant == "v7_split1" else None)
                    call = functools.partial(kernel, *inputs)
                    got = call().cpu()
                    kernel_mode = kernel.execution_mode
                    geometry[variant] = {"num_splits": kernel.num_splits,
                        "query_tile_tasks": kernel.query_tile_tasks,
                        "compute_grid": kernel.query_tile_tasks * kernel.num_splits}
                    if got.dtype != torch.bfloat16 or tuple(got.shape) != tuple(expected.shape):
                        raise ValueError(f"Unexpected output dtype/shape: {got.dtype} {got.shape}")
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
                    if (sample // max(len(order), 1)) % 2:
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
                        "input_dtype": "bfloat16", "output_dtype": "bfloat16",
                        "kernel_mode": programs[label][1], "geometry": geometry[label], "median_us": median*1000,
                        "effective_flops": flops, "effective_tflops": tflops,
                        "mfu_percent": 100*tflops/args.peak_tflops, "samples_ms": samples[label]}
                    report["results"].append(row)
                    print(f"{track} Q={sq} {label} round={round_id+1}: {median*1000:.3f} us MFU={row['mfu_percent']:.4f}%", flush=True)
                if all(label in samples for label in variants):
                    auto_ms = statistics.median(samples["v7_auto_split"])
                    direct_ms = statistics.median(samples["v7_split1"])
                    comparison = {"track": track, "B": b, "H": h, "Q": sq, "KV": args.kv,
                        "round": round_id+1, "auto_num_splits": geometry["v7_auto_split"]["num_splits"],
                        "auto_mode": programs["v7_auto_split"][1],
                        "auto_median_us": auto_ms*1000, "split1_median_us": direct_ms*1000,
                        "latency_saved_us": (auto_ms-direct_ms)*1000,
                        "speedup_split1_over_auto": auto_ms/direct_ms,
                        "split1_latency_change_percent": 100*(direct_ms/auto_ms-1)}
                    report.setdefault("comparisons", []).append(comparison)
                    print(f"COMPARE {track} Q={sq} round={round_id+1}: auto_splits={comparison['auto_num_splits']} split1 speedup={auto_ms/direct_ms:.4f}x saved={comparison['latency_saved_us']:.3f} us", flush=True)
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
