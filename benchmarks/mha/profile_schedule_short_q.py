"""Compare auto/auto-offset/manual schedules with interleaved short-Q timings."""
import argparse
import datetime
import hashlib
import json
import pathlib
import statistics
import traceback


MODES = ("auto", "auto-offset", "manual")
CASES = [
    ("q16", 2, 16, 8, 2048, False),
    ("q32", 2, 32, 8, 2048, False),
    ("b1_decode_noncausal", 1, 1, 16, 512, False),
    ("b1_decode_causal", 1, 1, 16, 512, True),
    ("q1_long_kv", 1, 1, 8, 8192, False),
    ("q127", 2, 127, 8, 2048, False),
    ("q129", 2, 129, 8, 2048, False),
    ("direct_q1", 2, 1, 64, 2048, False),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rounds", type=int, default=8)
    parser.add_argument("--pairs", type=int, default=61)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--output", default="mha_schedule_short_q_paired.json")
    args = parser.parse_args()
    if args.rounds < 2 or args.pairs < 3 or args.warmup < 1:
        parser.error("rounds>=2, pairs>=3, warmup>=1 required")
    report = {"status": "running", "settings": vars(args), "results": [], "cases": [],
              "time_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "policy": "Same process/input/current stream; all three variants compiled and warmed before timing; rotating/reversing three-mode order per sample group; warm cache; no automatic default change.",
              "timing": "Full-call NPU Event microseconds including stream-visible launch gaps and allocation effects; no baseline overhead subtraction."}
    path = pathlib.Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)

    def save():
        path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    save()
    try:
        import torch
        import torch_npu
        import tilelang
        import operators.mha.kernels.mha_paged_decode_ascend_v7 as implementation
        from operators.mha.kernels.mha_paged_decode_ascend_v0 import reference
        torch.npu.set_device(args.device)
        device = torch.device("npu", args.device)
        report["device"] = str(torch.npu.get_device_properties(args.device))
        report["versions"] = {"torch": torch.__version__, "torch_npu": torch_npu.__version__, "tilelang": tilelang.__version__}
        report["source_sha256"] = hashlib.sha256(pathlib.Path(implementation.__file__).read_bytes()).hexdigest()
        bundles = []
        # Finish all compilation and correctness work before any measured round.
        for name, b, sq, h, length, causal in CASES:
            report["active_case"] = name
            print(f"Preparing {name}: B={b} Q={sq} H={h} KV={length} causal={causal}", flush=True)
            save()
            gen = torch.Generator().manual_seed(7)
            q = torch.randn(b, sq, h, 128, generator=gen).bfloat16()
            k = torch.randn(b * length, h, 128, generator=gen).bfloat16()
            v = torch.randn(b * length, h, 128, generator=gen).bfloat16()
            width = length // 128
            table = torch.randperm(b * width, generator=gen).reshape(b, width).int()
            lengths = torch.full((b,), length, dtype=torch.int32)
            inputs = [x.to(device) for x in (q, k, v, table, lengths)]
            programs = {mode: implementation.build_kernel(b, sq, h, 128, k.shape[0], width,
                128, causal, "bfloat16", args.device, pipeline_stages=2, buffer_versions=2,
                q_resident=True, kv_versions=None, schedule_mode=mode, persistent_programs=None)
                for mode in MODES}
            tasks = b * ((sq + 127) // 128) * h
            splits = 1
            while splits < 16 and splits * 2 <= length // 128 and tasks * splits < 128:
                splits *= 2
            report["cases"].append({"case": name, "B": b, "Sq": sq, "H": h,
                "D": 128, "KV": length, "causal": causal, "dtype": "bfloat16",
                "page_size": 128, "seed": 7, "rtol": 1e-2, "atol": 4e-3,
                "query_tile_tasks": tasks, "splits": splits,
                "execution_modes": {mode: p.execution_mode for mode, p in programs.items()}})
            expected = reference(q, k, v, table, lengths, 128, causal)
            for mode, program in programs.items():
                report["active_schedule_mode"] = mode
                save()
                torch.testing.assert_close(program(*inputs).cpu(), expected, rtol=1e-2, atol=4e-3)
            bundles.append((name, inputs, programs))
        report.pop("active_schedule_mode", None)
        report["correctness"] = "passed"
        start, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)

        def measure(program, inputs):
            start.record()
            output = program(*inputs)
            end.record()
            end.synchronize()
            return start.elapsed_time(end) * 1000

        for round_id in range(args.rounds):
            order_cases = bundles if round_id % 2 == 0 else list(reversed(bundles))
            for name, inputs, programs in order_cases:
                report["active_case"] = name
                for _ in range(args.warmup):
                    for mode in MODES:
                        programs[mode](*inputs)
                torch.npu.synchronize()
                samples = {mode: [] for mode in MODES}
                row = {"case": name, "round": round_id+1, "status": "running"}
                report["results"].append(row)
                save()
                for pair in range(args.pairs):
                    offset = (pair + round_id) % len(MODES)
                    order = MODES[offset:] + MODES[:offset]
                    if (pair + round_id) % 2:
                        order = tuple(reversed(order))
                    for label in order:
                        samples[label].append(measure(programs[label], inputs))
                medians = {mode: statistics.median(values) for mode, values in samples.items()}
                speedups = {mode: medians["auto"] / medians[mode] for mode in MODES}
                row.update(status="passed", samples_us=samples, median_us=medians,
                           speedup_vs_auto=speedups)
                save()
                print(f"round={round_id+1} {name}: medians_us={medians} speedup_vs_auto={speedups}", flush=True)
        report["summary"] = []
        for name, *_ in CASES:
            rows = [r for r in report["results"] if r["case"] == name]
            for mode in MODES:
                report["summary"].append({"case": name, "schedule_mode": mode,
                    "median_round_us": statistics.median(r["median_us"][mode] for r in rows),
                    "median_round_speedup_vs_auto": statistics.median(r["speedup_vs_auto"][mode] for r in rows),
                    "min_round_speedup_vs_auto": min(r["speedup_vs_auto"][mode] for r in rows),
                    "max_round_speedup_vs_auto": max(r["speedup_vs_auto"][mode] for r in rows),
                    "faster_than_auto_rounds": sum(r["speedup_vs_auto"][mode] > 1 for r in rows),
                    "rounds": len(rows)})
        report.pop("active_case", None)
        report["status"] = "passed"
    except Exception:
        report.update(status="error", error=traceback.format_exc())
        traceback.print_exc()
    finally:
        save()
    print(f"{report['status'].upper()}: {path}", flush=True)
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
