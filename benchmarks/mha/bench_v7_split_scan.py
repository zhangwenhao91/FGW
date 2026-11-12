"""Scan v7 split=1/2/4/8/16 with paired timing and unchanged automatic baseline."""
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

Q_LENGTHS = (1, 16, 128, 129, 512, 1024, 4096, 8192)
SPLITS = (1, 2, 4, 8, 16)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--q-lengths", nargs="+", type=int, default=list(Q_LENGTHS))
    parser.add_argument("--bh", nargs="+", default=["2x8", "1x1"], help="Batch x heads, e.g. 1x8 4x4 2x16")
    parser.add_argument("--evaluate-cost-model", action="store_true")
    parser.add_argument("--kv", type=int, default=2048)
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--repeat", type=int, default=51)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--peak-tflops", type=float, default=378.0)
    parser.add_argument("--list-only", action="store_true")
    parser.add_argument("--output", default="mha_v7_split_scan.json")
    args = parser.parse_args()
    if (min(args.q_lengths) < 1 or args.kv < 128 or args.kv % 128 or args.rounds < 1
            or args.repeat < 3 or args.warmup < 1 or not math.isfinite(args.peak_tflops) or args.peak_tflops <= 0):
        parser.error("positive Q, 128-aligned KV, rounds>=1, repeat>=3, warmup>=1, positive finite peak required")
    try:
        bh = [tuple(map(int, item.split("x"))) for item in args.bh]
        if any(len(pair) != 2 or min(pair) <= 0 for pair in bh):
            raise ValueError()
    except ValueError:
        parser.error("--bh requires positive BxH pairs")
    cases = [(f"batch{b}_head{h}", b, h, q) for b, h in bh for q in args.q_lengths]
    if max(b * h * math.ceil(q/128) * 16 for _, b, h, q in cases) > 32768:
        parser.error("Scan compute grid exceeds validated 32768 cap; reduce B/H/Q")
    if args.list_only:
        print(json.dumps(cases, indent=2))
        return 0
    report = {"status": "running", "settings": vars(args), "results": [], "skipped": [], "best_by_case": [],
        "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "baseline": "Current v7 automatic selection compared with fixed split=1/2/4/8/16; auto candidate reuses the automatic callable",
        "speedup_policy": "speedup_vs_auto = automatic split median latency / candidate median latency; >1 means candidate faster",
        "flops_policy": "4*B*H*Q*KV*128, noncausal useful QK+PV only; no padding/softmax FLOPs",
        "timing": "Full native v7 callable, warm-cache NPU Event median; one invocation per sample. Includes Q conversion and output allocation plus direct kernel or partial+merge. Excludes compile, H2D, CPU reference and metadata preparation. Not CPU wall time.",
        "comparison": "Same B/H/Q/KV, BF16 Q/K/V/O, D128, page128, random page table, per-request lengths, noncausal, scale=128**-0.5, seed7. All manual/stages2/Q resident. split1 uses direct normalization/cast/store; other candidates use partial+merge. Whole strategy cost, not isolated merge time. Default remains automatic. Candidates exceeding KV tiles are explicitly skipped."}
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
        if args.evaluate_cost_model:
            from operators.mha.split_cost_model import MODEL_PATH
            report["cost_model"] = json.loads(MODEL_PATH.read_text(encoding="utf-8"))
            report["cost_model_sha256"] = hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest()
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
            auto_kernel = v7.build_kernel(b, sq, h, 128, k.shape[0], width, 128, False,
                "bfloat16", args.device, schedule_mode="manual", split_policy="legacy")
            model_kernel = None
            if args.evaluate_cost_model:
                model_kernel = v7.build_kernel(b, sq, h, 128, k.shape[0], width, 128, False,
                    "bfloat16", args.device, schedule_mode="manual", split_policy="cost_model")
            auto_splits = auto_kernel.num_splits
            auto_label = f"split{auto_splits}"
            variants = tuple(f"split{n}" for n in SPLITS if n <= width)
            for n in SPLITS:
                if n > width:
                    report["skipped"].append({"track": track, "Q": sq, "KV": args.kv,
                        "num_splits": n, "reason": "More splits than KV128 tiles; would create empty splits"})
            geometry = {}
            case_start = len(report["results"])
            for variant in variants:
                try:
                    n = int(variant.removeprefix("split"))
                    kernel = (auto_kernel if n == auto_splits else model_kernel
                        if model_kernel is not None and n == model_kernel.num_splits else None)
                    kernel = kernel if kernel is not None else v7.build_kernel(
                        b, sq, h, 128, k.shape[0], width, 128, False, "bfloat16", args.device,
                        schedule_mode="manual", num_splits=n)
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
                if auto_label in samples:
                    auto_ms = statistics.median(samples[auto_label])
                    for label in samples:
                        latency = statistics.median(samples[label])
                        report.setdefault("comparisons", []).append({"track": track, "B": b, "H": h,
                            "Q": sq, "KV": args.kv, "round": round_id+1,
                            "auto_num_splits": auto_splits, "num_splits": geometry[label]["num_splits"],
                            "auto_median_us": auto_ms*1000, "candidate_median_us": latency*1000,
                            "speedup_vs_auto": auto_ms/latency, "latency_saved_us": (auto_ms-latency)*1000})
                save()
            rows = report["results"][case_start:]
            stats = []
            for label in variants:
                measured = [row for row in rows if row["variant"] == label and row["status"] == "passed"]
                if len(measured) == args.rounds:
                    stats.append({"num_splits": geometry[label]["num_splits"],
                        "median_round_us": statistics.median(row["median_us"] for row in measured),
                        "min_round_us": min(row["median_us"] for row in measured),
                        "max_round_us": max(row["median_us"] for row in measured),
                        "median_round_mfu_percent": statistics.median(row["mfu_percent"] for row in measured),
                        "rounds": len(measured)})
            stats.sort(key=lambda item: item["median_round_us"])
            if stats:
                winners = []
                for rid in range(1, args.rounds+1):
                    candidates = [row for row in rows if row["status"] == "passed" and row["round"] == rid]
                    if candidates:
                        winners.append(min(candidates, key=lambda row: row["median_us"])["geometry"]["num_splits"])
                best = stats[0]
                auto = next((item for item in stats if item["num_splits"] == auto_splits), None)
                report["best_by_case"].append({"track": track, "B": b, "H": h, "Q": sq, "KV": args.kv,
                    "query_tile_tasks": auto_kernel.query_tile_tasks, "kv_tiles": width,
                    "auto_num_splits": auto_splits, "best_num_splits": best["num_splits"],
                    "best_median_us": best["median_round_us"], "winner_per_round": winners,
                    "within_2pct_of_best": [item["num_splits"] for item in stats if item["median_round_us"] <= best["median_round_us"]*1.02],
                    "best_speedup_vs_auto": auto["median_round_us"]/best["median_round_us"] if auto else None,
                    "all_candidates_passed": all(row["status"] == "passed" for row in rows), "ranking": stats})
                if model_kernel is not None:
                    chosen = next((item for item in stats if item["num_splits"] == model_kernel.num_splits), None)
                    report["best_by_case"][-1]["cost_model"] = {
                        "selected_num_splits": model_kernel.num_splits,
                        "median_us": chosen["median_round_us"] if chosen else None,
                        "relative_to_best": chosen["median_round_us"]/best["median_round_us"] if chosen else None,
                        "speedup_vs_legacy": auto["median_round_us"]/chosen["median_round_us"] if auto and chosen else None,
                        "selection_passed": chosen is not None}
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
