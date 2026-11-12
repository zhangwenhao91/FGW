"""Paired warm-cache Event timings: general Cube versus the existing v2 Vector callable."""
import argparse
import datetime
import hashlib
import itertools
import json
import math
from pathlib import Path
import statistics
import traceback


# name, B, Q, H, D, page_size, table_width, runtime lengths, causal
CASES = [(f"dim{d}_q16", 2, 16, 2, d, 128, 4, (512, 512), False)
         for d in (16, 48, 64, 80, 96, 128, 192, 256)]
CASES += [(f"d{d}_q{q}_kv2048", 2, q, 2, d, 128, 16, (2048, 2048), False)
          for d in (64, 128) for q in (1, 128, 512)]
CASES += [
    ("single_decode_kv8192", 1, 1, 1, 64, 256, 32, (8192,), False),
    ("page16_tail", 2, 4, 2, 16, 16, 8, (127, 113), False),
    ("page48_crossing", 2, 33, 2, 48, 48, 4, (191, 145), False),
    ("d128_q129_kv_tail", 2, 129, 2, 128, 128, 4, (511, 257), False),
    ("causal_d64_q16", 2, 16, 2, 64, 128, 4, (512, 512), True),
    ("causal_d128_q129", 2, 129, 2, 128, 128, 2, (256, 256), True),
    ("causal_empty_rows_d256", 2, 33, 2, 256, 128, 2, (17, 0), True),
]
MODES = ("cube_general", "vector_v2")


def aggregate(rows, case_info):
    summaries = []
    for key, info in case_info.items():
        selected = [r for r in rows if r["key"] == key and r["status"] == "passed"]
        if not selected:
            continue
        medians = {mode: statistics.median(r["median_us"][mode] for r in selected) for mode in MODES}
        ratios = [r["speedup_cube_over_vector"] for r in selected]
        summaries.append({"key": key, "category": info["category"], "dtype": info["dtype"],
                          "rounds": len(selected), "median_round_us": medians,
                          "speedup_cube_over_vector": medians["vector_v2"] / medians["cube_general"],
                          "min_round_speedup": min(ratios), "max_round_speedup": max(ratios),
                          "cube_faster_rounds": sum(v > 1 for v in ratios)})
    groups = {}
    for group in ("all", "vector_fallback", "stable_rejects_lengths", "cube_control", "float16", "bfloat16"):
        selected = [r for r in summaries if group == "all" or r["category"] == group or r["dtype"] == group]
        if selected:
            values = [r["speedup_cube_over_vector"] for r in selected]
            groups[group] = {"cases": len(values), "cube_faster_cases": sum(v > 1 for v in values),
                             "geomean_speedup": math.exp(statistics.mean(math.log(v) for v in values)),
                             "min_speedup": min(values), "max_speedup": max(values)}
    return summaries, groups


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rounds", type=int, default=6)
    parser.add_argument("--pairs", type=int, default=31)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--dtypes", nargs="+", choices=["float16", "bfloat16"], default=["float16", "bfloat16"])
    parser.add_argument("--cases", nargs="+", choices=[c[0] for c in CASES])
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", default="mha_cube_vector_paired.json")
    parser.add_argument("--list-only", action="store_true")
    args = parser.parse_args()
    if args.rounds < 2 or args.pairs < 3 or args.warmup < 1:
        parser.error("rounds>=2, pairs>=3, warmup>=1 required")
    cases = [c for c in CASES if not args.cases or c[0] in args.cases]
    if args.list_only:
        print(json.dumps({"cases": cases, "dtypes": args.dtypes, "comparisons": len(cases)*len(args.dtypes)}, indent=2))
        return 0
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    report = {"status": "running", "settings": vars(args), "cases": {}, "results": [],
              "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "timing": "Warm-cache NPU Event, one prebuilt callable invocation per sample; includes output/scratch allocation effects, all device kernels, and stream-visible launch gaps. Excludes compilation, H2D, CPU reference and public API metadata validation/host synchronization. Not CPU wall time or isolated Cube-instruction time.",
              "comparison": "Same input tensors/stream/dtype/scale/causal mask/page table/lengths for general Cube and v2 Vector. Cube split=1; v2 retains its existing automatic split/merge or unsplit-v1 selection. Measures their complete device-call strategies.",
              "speedup_policy": "v2_Vector_us / general_Cube_us; >1 means Cube faster. BF16 D128 full-block cases are controls, not a comparison against the optimized default v7 Cube.",
              "phase": "compile_and_accuracy"}

    def save():
        report["summary"], report["groups"] = aggregate(report["results"], report["cases"])
        for dest, compact in ((path, False), (path.with_name(path.stem+"_summary.json"), True)):
            payload = dict(report)
            if compact:
                payload["results"] = [{k: v for k, v in row.items() if k != "samples_us"} for row in report["results"]]
            tmp = dest.with_suffix(dest.suffix+".tmp")
            tmp.write_text(json.dumps(payload, indent=2)+"\n", encoding="utf-8")
            tmp.replace(dest)

    save()
    try:
        import torch
        import torch_npu
        import tilelang
        from operators.mha.kernels import mha_paged_decode_ascend_v7 as v7
        from operators.mha.kernels import mha_paged_decode_ascend_v2 as v2
        from operators.mha.kernels import mha_paged_decode_ascend_v1 as v1
        from operators.mha.kernels import mha_paged_decode_ascend_cube_general as cube
        from benchmarks.mha.bench_mha_mfu import chunked_cpu_reference
        from tests.mha.test_mha_paged_accuracy import make_inputs
        torch.npu.set_device(args.device)
        device = torch.device("npu", args.device)
        report["device"] = str(torch.npu.get_device_properties(args.device))
        report["versions"] = {"torch": torch.__version__, "torch_npu": torch_npu.__version__, "tilelang": tilelang.__version__}
        report["implementation_sha256"] = {name: hashlib.sha256(Path(mod.__file__).read_bytes()).hexdigest()
                                           for name, mod in (("cube", cube), ("v2", v2), ("v1", v1))}
        bundles = []
        for case, dtype_name in itertools.product(cases, args.dtypes):
            name, b, sq, h, d, p, w, lengths_list, causal = case
            key = f"{name}/{dtype_name}"
            report["active_case"] = key
            dtype = getattr(torch, dtype_name)
            static_route = v7.route(dtype, d, p, sq, causal)
            category = ("vector_fallback" if static_route == "fallback_v2" else
                        "stable_rejects_lengths" if any(n <= 0 or n % 128 for n in lengths_list) else "cube_control")
            info = {"case": name, "dtype": dtype_name, "B": b, "Q": sq, "H": h, "D": d,
                    "page_size": p, "table_width": w, "lengths": lengths_list, "causal": causal,
                    "category": category, "correctness": {}}
            report["cases"][key] = info
            print(f"Preparing {key}: {category}", flush=True)
            save()
            q, k, v, table = make_inputs((name, b, sq, h, d, p, w), args.seed, dtype, torch)
            lengths = torch.tensor(lengths_list, dtype=torch.int32)
            live = torch.zeros(k.shape[0], dtype=torch.bool)
            for bi, length in enumerate(lengths_list):
                pos = torch.arange(length)
                physical = table[bi, pos // p].long()*p + pos % p
                live[physical] = True
            k[~live], v[~live] = float("nan"), float("nan")
            expected = chunked_cpu_reference(q, k, v, table, lengths, p, causal, 128)
            inputs = [x.to(device) for x in (q, k, v, table, lengths)]
            programs = {}
            for label, module in (("cube_general", cube), ("vector_v2", v2)):
                report["active_variant"] = label
                save()
                program = module.build_kernel(b, sq, h, d, k.shape[0], w, p, causal, dtype_name, args.device)
                got = program(*inputs).cpu()
                assert got.shape == q.shape and got.dtype == dtype and got.is_contiguous()
                assert torch.isfinite(got).all(), "Nonfinite output"
                rtol, atol = (1e-2, 4e-3) if dtype == torch.bfloat16 else (2e-3, 2e-3)
                torch.testing.assert_close(got, expected, rtol=rtol, atol=atol)
                visible = lengths[:, None].expand(b, sq)
                if causal:
                    visible = (visible-sq+torch.arange(sq)[None, :]+1).clamp_min(0)
                assert torch.count_nonzero(got[visible == 0]).item() == 0
                info["correctness"][label] = {"status": "passed", "rtol": rtol, "atol": atol,
                                              "max_abs_error": (got.float()-expected.float()).abs().max().item()}
                programs[label] = program
            source = programs["cube_general"].direct_kernel.get_kernel_source()
            info["cube_mmad_calls"] = source.count("asc_mmad(")
            assert info["cube_mmad_calls"] >= 2
            info["cube_generated_sha256"] = hashlib.sha256(source.encode()).hexdigest()
            bundles.append((key, inputs, programs))
            save()
        report.pop("active_variant", None)
        report["phase"] = "timing"
        report["correctness"] = "passed"
        start, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)

        def measure(program, inputs):
            start.record()
            output = program(*inputs)
            end.record()
            end.synchronize()
            us = start.elapsed_time(end)*1000
            if not math.isfinite(us) or us <= 0:
                raise ValueError(f"Invalid Event time: {us}")
            return us

        for round_id in range(args.rounds):
            for key, inputs, programs in (bundles if round_id % 2 == 0 else list(reversed(bundles))):
                report["active_case"] = key
                row = {"key": key, "round": round_id+1, "status": "running"}
                report["results"].append(row)
                save()
                for _ in range(args.warmup):
                    for mode in (MODES if round_id % 2 == 0 else tuple(reversed(MODES))):
                        programs[mode](*inputs)
                torch.npu.synchronize()
                samples = {mode: [] for mode in MODES}
                for pair in range(args.pairs):
                    for mode in (MODES if (pair+round_id) % 2 == 0 else tuple(reversed(MODES))):
                        samples[mode].append(measure(programs[mode], inputs))
                medians = {mode: statistics.median(values) for mode, values in samples.items()}
                speedup = medians["vector_v2"]/medians["cube_general"]
                row.update(status="passed", samples_us=samples, median_us=medians,
                           speedup_cube_over_vector=speedup,
                           median_pair_speedup=statistics.median(v/c for v, c in zip(samples["vector_v2"], samples["cube_general"])),
                           p90_us={mode: sorted(values)[int(0.9*(len(values)-1))] for mode, values in samples.items()})
                save()
                print(f"round={round_id+1} {key}: Cube={medians['cube_general']:.3f} us Vector={medians['vector_v2']:.3f} us speedup={speedup:.4f}x", flush=True)
        report.pop("active_case", None)
        report["phase"] = "complete"
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
