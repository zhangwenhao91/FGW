"""Collect upstream msprof_detail durations and pipe counters before tuning.

Upstream reports aggregated per-invocation profiler records, not an Event
median. Keep this measurement separate from previous Event timing results.
"""
import argparse
import dataclasses
import datetime
import hashlib
import inspect
import importlib
import json
import pathlib
import statistics
import traceback


CASES = [
    ("decode_noncausal", 1, 1, 16, 512, False),
    ("decode_causal", 1, 1, 16, 512, True),
    ("query4", 2, 4, 8, 2048, False),
    ("query128", 1, 128, 2, 2048, False),
]
DEFAULT_CASE_NAMES = {case[0] for case in CASES}
CASES += [
    ("query4096", 2, 4096, 8, 2048, False),
    ("query32768", 2, 32768, 8, 2048, False),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", nargs="+", choices=[c[0] for c in CASES])
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--repeat", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--impl", choices=["v4", "v5", "v6", "v7"], default="v4")
    parser.add_argument("--pipeline-stages", type=int, choices=[1, 2, 3], default=2)
    parser.add_argument("--buffer-versions", type=int, choices=[1, 2, 3], default=2)
    parser.add_argument("--kv-versions", type=int, choices=[1, 2, 3], default=None)
    parser.add_argument("--q-resident", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--reference-chunk-size", type=int, default=128)
    parser.add_argument("--output", default="mha_msprof_report.json")
    parser.add_argument("--schedule-mode", choices=["auto", "auto-offset", "manual"], default="auto")
    args = parser.parse_args()
    if args.schedule_mode != "auto" and args.impl != "v7":
        parser.error("Schedule experiments require --impl v7")
    if args.q_resident is None:
        args.q_resident = args.impl == "v7"
    if (args.q_resident or args.kv_versions is not None) and args.impl != "v7":
        parser.error("Overlap candidates currently require --impl v7")
    if min(args.rounds, args.repeat, args.warmup) < 1:
        parser.error("rounds, repeat, warmup must be positive")
    if args.reference_chunk_size < 1:
        parser.error("reference-chunk-size must be positive")
    report = {"status": "running", "settings": vars(args), "schedule_mode": args.schedule_mode, "results": [],
              "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "baseline_config": {"q_tile": 128, "kv_tile": 128, "pipeline_stages": args.pipeline_stages,
                                  "streamed_buffer_versions": args.buffer_versions,
                                  "kv_l1_versions": args.kv_versions if args.kv_versions is not None else args.buffer_versions,
                                  "q_resident_l0a": args.q_resident},
              "notes": ["Upstream msprof_detail flushes cache before each invocation (256 MiB buffer).",
                        "dur_ns is upstream profiler record aggregation per invocation, not Event elapsed median.",
                        "Pipe fields are ratios against their own AIC/AIV total cycles; they can overlap and must not be summed as a time breakdown.",
                        "Full includes Q layout conversion if materialized, partial and merge; isolated stages run in separate profiler sessions.",
                        "Do not compare this cache-flushed measurement directly with earlier warm-cache Event results.",
                        "Candidate parameters are recorded; defaults preserve the measured baseline."]}
    path = pathlib.Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)

    def save():
        path.write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")

    save()
    try:
        import torch
        import torch_npu
        import tilelang
        import tilelang.profiler.msprof as msprof_module
        from tilelang.profiler import do_bench
        implementation = importlib.import_module(f"operators.mha.kernels.mha_paged_decode_ascend_{args.impl}")
        build_kernel = implementation.build_kernel
        report["kernel_source_sha256"] = hashlib.sha256(pathlib.Path(implementation.__file__).read_bytes()).hexdigest()
        from benchmarks.mha.bench_mha_mfu import chunked_cpu_reference
        from operators.mha.kernels.mha_paged_decode_ascend_v0 import reference
        torch.npu.set_device(args.device)
        device = torch.device("npu", args.device)
        report["device"] = str(torch.npu.get_device_properties(args.device))
        report["implementation"] = args.impl
        if args.impl == "v5":
            report["notes"].append("v5 merge includes head-major to contiguous B,Sq,H,D output conversion")
        if args.impl == "v6":
            report["notes"].append("v6 merge writes final output directly and precomputes row weights")
        report["versions"] = {"torch": torch.__version__, "torch_npu": torch_npu.__version__, "tilelang": tilelang.__version__}
        # Preserve parser provenance: upstream parsing can change between builds.
        source = inspect.getsource(msprof_module)
        report["msprof_source_sha256"] = hashlib.sha256(source.encode()).hexdigest()
        source_path = path.with_name(path.stem + "_msprof_source.py")
        source_path.write_text(source, encoding="utf-8")
        report["msprof_source_file"] = source_path.name
        report["do_bench_signature"] = str(inspect.signature(do_bench))
        for name, batch, sq, heads, length, causal in CASES:
            selected = set(args.cases) if args.cases else DEFAULT_CASE_NAMES
            if name not in selected:
                continue
            d, page, width = 128, 128, length // 128
            gen = torch.Generator().manual_seed(7)
            q = torch.randn(batch, sq, heads, d, generator=gen).bfloat16()
            k = torch.randn(batch * length, heads, d, generator=gen).bfloat16()
            v = torch.randn(batch * length, heads, d, generator=gen).bfloat16()
            table = torch.randperm(batch * width, generator=gen).reshape(batch, width).int()
            lengths = torch.full((batch,), length, dtype=torch.int32)
            inputs = [x.to(device) for x in (q, k, v, table, lengths)]
            extra = {"q_resident": args.q_resident, "kv_versions": args.kv_versions, "schedule_mode": args.schedule_mode} if args.impl == "v7" else {}
            program = build_kernel(batch, sq, heads, d, k.shape[0], width, page, causal, "bfloat16", args.device,
                                   pipeline_stages=args.pipeline_stages, buffer_versions=args.buffer_versions, **extra)
            got = program(*inputs).cpu()
            if sq > 512:
                expected = chunked_cpu_reference(q, k, v, table, lengths, page, causal, args.reference_chunk_size)
            else:
                expected = reference(q, k, v, table, lengths, page, causal)
            torch.testing.assert_close(got, expected, rtol=1e-2, atol=4e-3)
            prepared = program.prepare_q(inputs[0])
            direct_mode = getattr(program, "execution_mode", "split_and_merge") == "direct_single_split"
            if direct_mode:
                functions = {"full": lambda: program(*inputs),
                             "direct": lambda: program.direct_kernel(prepared, *inputs[1:])}
            else:
                parts, lse = program.partial_kernel(prepared, *inputs[1:])
                torch.npu.synchronize()
                functions = {"full": lambda: program(*inputs),
                             "partial": lambda: program.partial_kernel(prepared, *inputs[1:]),
                             "merge": lambda: program.merge_kernel(parts, lse)}
            for round_id in range(args.rounds):
                stages = list(functions)
                offset = round_id % len(stages)
                stages = stages[offset:] + stages[:offset]
                for stage in stages:
                    row = {"case": name, "stage": stage, "round": round_id+1,
                           "shape": {"B": batch, "Sq": sq, "H": heads, "D": d, "KV": length},
                           "causal": causal, "correctness": "passed", "status": "running"}
                    row["kernel_mode"] = "direct_single_split" if direct_mode else "split_and_merge"
                    tasks = batch * ((sq + 127) // 128) * heads
                    splits = 1
                    while splits < 16 and splits * 2 <= length // 128 and tasks * splits < 128:
                        splits *= 2
                    row["launch_geometry"] = {"query_tile_tasks": tasks, "splits": splits,
                        "partial_grid": tasks * splits,
                        "merge_work": batch * heads * ((sq + 7) // 8),
                        "merge_grid": 0 if direct_mode else min(batch * heads * ((sq + 7) // 8), 32768) if args.impl in ("v6", "v7") else batch * heads * ((sq + 7) // 8)}
                    report["results"].append(row)
                    save()
                    profile = do_bench(functions[stage], backend="msprof_detail", device=device,
                                       _n_warmup=args.warmup, _n_repeat=args.repeat, cache_size=256)
                    if not dataclasses.is_dataclass(profile):
                        raise TypeError(f"Expected KernelProfile dataclass, got {type(profile)}")
                    row.update(status="passed", profile=dataclasses.asdict(profile),
                               profiler_duration_us=float(profile.dur_ns) / 1000)
                    save()
                    print(f"{name} round={round_id+1} {stage}: {profile}", flush=True)
        report["summary"] = []
        for name, *_ in CASES:
            for stage in ("full", "partial", "merge", "direct"):
                rows = [r for r in report["results"] if r["case"] == name and r["stage"] == stage]
                if rows:
                    report["summary"].append({"case": name, "stage": stage,
                        "median_round_profiler_duration_us": statistics.median(r["profiler_duration_us"] for r in rows),
                        "round_profiler_durations_us": [r["profiler_duration_us"] for r in rows]})
        report["status"] = "passed"
    except Exception:
        report.update(status="error", error=traceback.format_exc())
        if report["results"] and report["results"][-1]["status"] == "running":
            report["results"][-1].update(status="error", error=report["error"])
        traceback.print_exc()
    finally:
        save()
    print(f"{report['status'].upper()}: {path}", flush=True)
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
