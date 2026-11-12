"""Interleave base/resident executions on identical inputs before default selection."""
import argparse
import datetime
import hashlib
import json
import pathlib
import statistics
import traceback


CASES = [
    ("q16", 2, 16, 8, 2048, False),
    ("q32", 2, 32, 8, 2048, False),
    ("b1_decode_noncausal", 1, 1, 16, 512, False),
    ("b1_decode_causal", 1, 1, 16, 512, True),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rounds", type=int, default=8)
    parser.add_argument("--pairs", type=int, default=61)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--output", default="mha_q_resident_paired.json")
    args = parser.parse_args()
    if args.rounds < 2 or args.pairs < 3 or args.warmup < 1:
        parser.error("rounds>=2, pairs>=3, warmup>=1 required")
    report = {"status": "running", "settings": vars(args), "results": [],
              "time_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "schedule_mode": "auto",
              "policy": "Same process/input/current stream; both variants compiled and warmed before timing; per-pair AB/BA alternation; warm cache; no automatic default change.",
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
            gen = torch.Generator().manual_seed(7)
            q = torch.randn(b, sq, h, 128, generator=gen).bfloat16()
            k = torch.randn(b * length, h, 128, generator=gen).bfloat16()
            v = torch.randn(b * length, h, 128, generator=gen).bfloat16()
            width = length // 128
            table = torch.randperm(b * width, generator=gen).reshape(b, width).int()
            lengths = torch.full((b,), length, dtype=torch.int32)
            inputs = [x.to(device) for x in (q, k, v, table, lengths)]
            programs = {label: implementation.build_kernel(b, sq, h, 128, k.shape[0], width,
                128, causal, "bfloat16", args.device, pipeline_stages=2, buffer_versions=2,
                q_resident=resident, kv_versions=2, schedule_mode="auto", split_policy="legacy", persistent_programs=None)
                for label, resident in (("base", False), ("resident", True))}
            expected = reference(q, k, v, table, lengths, 128, causal)
            for p in programs.values():
                torch.testing.assert_close(p(*inputs).cpu(), expected, rtol=1e-2, atol=4e-3)
            bundles.append((name, inputs, programs))
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
                for _ in range(args.warmup):
                    programs["base"](*inputs)
                    programs["resident"](*inputs)
                torch.npu.synchronize()
                samples = {"base": [], "resident": []}
                row = {"case": name, "round": round_id+1, "status": "running"}
                report["results"].append(row)
                save()
                for pair in range(args.pairs):
                    order = ("base", "resident") if (pair + round_id) % 2 == 0 else ("resident", "base")
                    for label in order:
                        samples[label].append(measure(programs[label], inputs))
                base = statistics.median(samples["base"])
                resident = statistics.median(samples["resident"])
                row.update(status="passed", samples_us=samples, base_median_us=base,
                           resident_median_us=resident, speedup=base/resident,
                           resident_latency_change_percent=(resident/base-1)*100)
                save()
                print(f"round={round_id+1} {name}: base={base:.3f} us resident={resident:.3f} us speedup={base/resident:.4f}", flush=True)
        report["summary"] = []
        for name, *_ in CASES:
            rows = [r for r in report["results"] if r["case"] == name]
            report["summary"].append({"case": name,
                "median_round_base_us": statistics.median(r["base_median_us"] for r in rows),
                "median_round_resident_us": statistics.median(r["resident_median_us"] for r in rows),
                "median_round_speedup": statistics.median(r["speedup"] for r in rows),
                "min_round_speedup": min(r["speedup"] for r in rows),
                "max_round_speedup": max(r["speedup"] for r in rows),
                "resident_faster_rounds": sum(r["speedup"] > 1 for r in rows), "rounds": len(rows)})
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
