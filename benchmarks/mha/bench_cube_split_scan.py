"""Same-process split=1/2/4/8/16/32 scan for the shared general Cube kernel."""
import argparse
import datetime
import functools
import hashlib
import itertools
import json
import math
from pathlib import Path
import statistics
import traceback

from benchmarks.mha.bench_cube_vector import CASES
from benchmarks.mha.bench_cube_ascendc import performance
from benchmarks.mha.bench_mha_mfu import useful_flops
from operators.mha.general_split_policy import SPLITS, select_splits
from benchmarks.mha.general_model_cases import HOLDOUT


COMPONENT_CASES = {
    "d64_q1_kv2048", "d128_q1_kv2048", "single_decode_kv8192",
    "d128_q128_kv2048", "page16_tail", "d128_q129_kv_tail",
    "causal_d128_q129", "causal_empty_rows_d256",
}


def plan(case, requested, evaluate_model=False):
    _, b, sq, h, _, page, width, _, _ = case
    tasks = b*h*((sq+31)//32)
    tiles = (page*width+31)//32
    auto = select_splits(tasks, tiles)
    allowed = sorted({1, auto, *(s for s in requested if s <= tiles)})
    result = {"auto_splits": auto, "splits": allowed,
            "skipped_splits": [{"num_splits": s, "reason": "Exceeds KV tiles in table capacity"}
                               for s in requested if s > tiles],
            "query_tile_tasks": tasks, "capacity_tiles": tiles}
    if evaluate_model:
        from operators.mha.general_split_cost_model import select_model_splits
        model_splits = select_model_splits(b, sq, h, case[4], page, width, case[7])
        result["model_splits"] = model_splits
        result["splits"] = sorted(set(allowed+[model_splits]))
    return result


def aggregate(results, cases, components, peak):
    summaries = []
    for key, info in cases.items():
        rows = [r for r in results if r["key"] == key and r["status"] == "passed"]
        if not rows:
            continue
        medians = {str(s): statistics.median(r["median_us"][str(s)] for r in rows) for s in info["splits"]}
        best = min(info["splits"], key=lambda s: medians[str(s)])
        best_us = medians[str(best)]
        near = [s for s in info["splits"] if medians[str(s)] <= best_us*1.02]
        auto_us = medians[str(info["auto_splits"])]
        variants = {}
        for s in info["splits"]:
            label = str(s)
            speedups = [r["median_us"][str(info["auto_splits"])]/r["median_us"][label] for r in rows]
            variant = performance(info["effective_flops"], medians[label], peak)
            variant.update(speedup_vs_auto=auto_us/medians[label],
                           faster_than_auto_rounds=sum(v > 1 for v in speedups),
                           min_round_speedup_vs_auto=min(speedups), max_round_speedup_vs_auto=max(speedups))
            comp = [c for c in components if c["key"] == key and c["num_splits"] == s and c["status"] == "passed"]
            if comp:
                variant["components_us"] = {part: statistics.median(c["median_us"][part] for c in comp)
                                             for part in ("partial", "merge")}
            variants[label] = variant
        summary = {"key": key, "dtype": info["dtype"], "auto_splits": info["auto_splits"],
                          "best_split": best, "near_best_2pct": near, "smallest_near_best_2pct": min(near),
                          "rounds": len(rows), "auto_regret_percent": (auto_us/best_us-1)*100,
                          "best_speedup_vs_auto": auto_us/best_us,
                          "best_speedup_vs_split1": medians["1"]/best_us, "variants": variants}
        if "model_splits" in info:
            model_us = medians[str(info["model_splits"])]
            summary.update(model_splits=info["model_splits"], model_regret_percent=(model_us/best_us-1)*100,
                           model_speedup_vs_auto=auto_us/model_us)
        summaries.append(summary)
    groups = {}
    for name in ("all", "float16", "bfloat16"):
        selected = [s for s in summaries if name == "all" or s["dtype"] == name]
        if selected:
            groups[name] = {"cases": len(selected),
                            "auto_within_2pct_cases": sum(s["auto_regret_percent"] <= 2 for s in selected),
                            "best_is_split1_cases": sum(s["best_split"] == 1 for s in selected),
                            "geomean_best_speedup_vs_auto": math.exp(statistics.mean(math.log(s["best_speedup_vs_auto"]) for s in selected)),
                            "max_auto_regret_percent": max(s["auto_regret_percent"] for s in selected)}
            modeled = [s for s in selected if "model_splits" in s]
            if modeled:
                groups[name].update(model_within_2pct_cases=sum(s["model_regret_percent"] <= 2 for s in modeled),
                                    max_model_regret_percent=max(s["model_regret_percent"] for s in modeled),
                                    geomean_model_speedup_vs_auto=math.exp(statistics.mean(math.log(s["model_speedup_vs_auto"]) for s in modeled)))
    return summaries, groups


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=["standard", "holdout", "all"], default="standard")
    parser.add_argument("--cases", nargs="+", choices=[c[0] for c in CASES+HOLDOUT])
    parser.add_argument("--evaluate-cost-model", action="store_true")
    parser.add_argument("--dtypes", nargs="+", choices=["float16", "bfloat16"], default=["float16", "bfloat16"])
    parser.add_argument("--splits", nargs="+", type=int, choices=SPLITS, default=list(SPLITS))
    parser.add_argument("--rounds", type=int, default=6)
    parser.add_argument("--pairs", type=int, default=31)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--component-rounds", type=int, default=2)
    parser.add_argument("--component-pairs", type=int, default=11)
    parser.add_argument("--component-warmup", type=int, default=5)
    parser.add_argument("--components", choices=["focus", "all", "none"], default="focus")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--peak-tflops", type=float, default=378.0)
    parser.add_argument("--output", default="mha_cube_split_scan_summary.json")
    parser.add_argument("--save-samples", action="store_true")
    parser.add_argument("--list-only", action="store_true")
    args = parser.parse_args()
    if (args.rounds < 2 or args.pairs < 3 or args.warmup < 1 or args.component_rounds < 1
            or args.component_pairs < 3 or args.component_warmup < 1
            or not math.isfinite(args.peak_tflops) or args.peak_tflops <= 0):
        parser.error("rounds>=2, pairs>=3, warmup>=1, valid component counts and finite positive peak required")
    suite = {"standard": CASES, "holdout": HOLDOUT, "all": CASES+HOLDOUT}[args.suite]
    if args.cases and set(args.cases)-{c[0] for c in suite}:
        parser.error("Selected case is not in the selected suite")
    cases = [c for c in suite if not args.cases or c[0] in args.cases]
    if args.list_only:
        print(json.dumps({"cases": [{"spec": c, **plan(c, args.splits, args.evaluate_cost_model)} for c in cases],
                          "dtype_cases": len(cases)*len(args.dtypes),
                          "compiled_variants": sum(len(plan(c, args.splits, args.evaluate_cost_model)["splits"]) for c in cases)*len(args.dtypes)}, indent=2))
        return 0
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    stem = path.stem.removesuffix("_summary")
    paste_path = path.with_name(stem+"_paste.txt")
    sample_path = path.with_name(stem+"_samples.jsonl")
    report = {"status": "running", "phase": "compile_and_accuracy", "settings": vars(args),
              "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "cases": {}, "results": [], "components": [],
              "timing": "Warm-cache NPU Event per complete prebuilt callable. Includes output/scratch allocation effects and all kernels. Excludes compilation, H2D, reference, host metadata validation. Same tensors/current stream for all splits. Rotate and reverse variant order; reverse case order across rounds.",
              "selection": "Empirical best and smallest split within2% are diagnostic measurements, not a deployed lookup policy. Original fill28 policy remains unchanged. Best observed on this scan is not an independent validation result.",
              "component_policy": "Separate partial and merge Event sessions; merge consumes precomputed fixed FP32 partials. Times are not additive and must not be subtracted from full-call time as an exact overhead decomposition.",
              "flops_policy": "Useful original-D QK+PV only; causal counts visible keys per row; 378 TFLOPS dense Cube peak on verified28-core device",
              "paste_file": paste_path.name}

    def save():
        report["summary"], report["groups"] = aggregate(report["results"], report["cases"], report["components"], args.peak_tflops)
        tmp = path.with_suffix(path.suffix+".tmp")
        tmp.write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
        tmp.replace(path)
        lines = [f"status={report['status']} phase={report['phase']} correctness={report.get('correctness')}",
                 f"active_case={report.get('active_case')} active_split={report.get('active_split')} active_measurement={report.get('active_measurement')}",
                 "GROUPS: "+json.dumps(report["groups"], ensure_ascii=False)]
        if report.get("error"):
            lines.append("ERROR:\n"+report["error"][-8000:])
        for s in report["summary"]:
            timings = ", ".join(f"S{k}={v['us']:.3f}" for k, v in s["variants"].items())
            lines.append(f"{s['key']} auto={s['auto_splits']} best={s['best_split']} near2%={s['near_best_2pct']} regret={s['auto_regret_percent']:.2f}% rounds={s['rounds']} us: {timings}")
            if "model_splits" in s:
                lines.append(f"  model={s['model_splits']} model_regret={s['model_regret_percent']:.2f}% model_vs_auto={s['model_speedup_vs_auto']:.4f}x")
            components = {k: {part: round(us, 3) for part, us in v["components_us"].items()}
                          for k, v in s["variants"].items() if "components_us" in v}
            if components:
                lines.append("  separate_components_us: "+json.dumps(components))
        paste_path.write_text("\n".join(lines)+"\n", encoding="utf-8")

    save()
    try:
        import torch
        import torch_npu
        import tilelang
        from operators.mha.kernels import mha_paged_decode_ascend_cube_general as cube
        from operators.mha import general_split_policy
        from benchmarks.mha.bench_mha_mfu import chunked_cpu_reference
        from tests.mha.test_mha_paged_accuracy import make_inputs
        torch.npu.set_device(args.device)
        props = torch.npu.get_device_properties(args.device)
        if getattr(props, "cube_core_num", None) != 28 and args.peak_tflops == 378.0:
            raise ValueError("378 TFLOPS requires28 Cube cores; override --peak-tflops otherwise")
        device = torch.device("npu", args.device)
        report["device"] = str(props)
        report["versions"] = {"torch": torch.__version__, "torch_npu": torch_npu.__version__, "tilelang": tilelang.__version__}
        report["kernel_sha256"] = hashlib.sha256(Path(cube.__file__).read_bytes()).hexdigest()
        report["split_policy_sha256"] = hashlib.sha256(Path(general_split_policy.__file__).read_bytes()).hexdigest()
        report["benchmark_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        if args.evaluate_cost_model:
            from operators.mha.general_split_cost_model import MODEL_PATH
            report["general_model_sha256"] = hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest()
            report["model_validation_scope"] = "Independent holdout" if args.suite == "holdout" else "Calibration replay" if args.suite == "standard" else "Mixed: inspect cases separately"
        bundles = []
        for case, dtype_name in itertools.product(cases, args.dtypes):
            name, b, sq, h, d, p, w, lengths_list, causal = case
            key = f"{name}/{dtype_name}"
            report["active_case"] = key
            dtype = getattr(torch, dtype_name)
            info = {"case": name, "dtype": dtype_name, "B": b, "Q": sq, "H": h, "D": d,
                    "page_size": p, "table_width": w, "lengths": lengths_list, "causal": causal,
                    "effective_flops": useful_flops(lengths_list, sq, h, d, causal),
                    **plan(case, args.splits, args.evaluate_cost_model), "correctness": {}, "geometry": {}}
            report["cases"][key] = info
            print(f"Preparing {key}: auto={info['auto_splits']} candidates={info['splits']}", flush=True)
            save()
            q, k, v, table = make_inputs((name, b, sq, h, d, p, w), args.seed, dtype, torch)
            lengths = torch.tensor(lengths_list, dtype=torch.int32)
            expected = chunked_cpu_reference(q, k, v, table, lengths, p, causal, 128)
            inputs = [x.to(device) for x in (q, k, v, table, lengths)]
            visible = lengths[:, None].expand(b, sq)
            if causal:
                visible = (visible-sq+torch.arange(sq)[None, :]+1).clamp_min(0)
            programs = {}
            for splits in info["splits"]:
                report["active_split"] = splits
                save()
                if args.evaluate_cost_model and splits == info["model_splits"]:
                    program = cube.build_kernel(b, sq, h, d, k.shape[0], w, p, causal, dtype_name,
                                                args.device, split_policy="cost_model", cache_lengths=tuple(lengths_list))
                    assert program.num_splits == splits, "Public model selector must match offline prediction"
                else:
                    program = cube.build_kernel(b, sq, h, d, k.shape[0], w, p, causal, dtype_name,
                                                args.device, num_splits=splits)
                got = program(*inputs).cpu()
                assert got.shape == q.shape and got.dtype == dtype and got.is_contiguous()
                assert torch.isfinite(got).all()
                rtol, atol = (1e-2, 4e-3) if dtype == torch.bfloat16 else (2e-3, 2e-3)
                torch.testing.assert_close(got, expected, rtol=rtol, atol=atol)
                assert torch.count_nonzero(got[visible == 0]).item() == 0
                source_kernel = program.direct_kernel if splits == 1 else program.partial_kernel
                source = source_kernel.get_kernel_source()
                assert source.count("asc_mmad(") >= 2
                label = str(splits)
                info["correctness"][label] = {"status": "passed", "rtol": rtol, "atol": atol,
                                              "max_abs_error": (got.float()-expected.float()).abs().max().item()}
                info["geometry"][label] = {"compute_grid": program.compute_grid,
                                           "execution_mode": program.execution_mode,
                                           "generated_sha256": hashlib.sha256(source.encode()).hexdigest()}
                programs[label] = program
                save()
            bundles.append((key, inputs, programs))
        report.pop("active_split", None)
        report["correctness"] = "passed"
        report["phase"] = "timing_full_calls"
        start, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)

        def measure(call):
            start.record()
            output = call()
            end.record()
            end.synchronize()
            us = start.elapsed_time(end)*1000
            if not math.isfinite(us) or us <= 0:
                raise ValueError(f"Invalid Event time {us}")
            return us

        def sample(calls, round_id, warmup, pairs):
            labels = tuple(calls)
            for _ in range(warmup):
                for label in (labels if round_id % 2 == 0 else tuple(reversed(labels))):
                    calls[label]()
            torch.npu.synchronize()
            values = {label: [] for label in labels}
            for pair in range(pairs):
                index = pair+round_id
                offset = index % len(labels)
                order = labels[offset:]+labels[:offset]
                if (index//len(labels)) % 2:
                    order = tuple(reversed(order))
                for label in order:
                    report["active_measurement"] = label
                    values[label].append(measure(calls[label]))
            report.pop("active_measurement", None)
            return values

        if args.save_samples:
            sample_path.write_text("", encoding="utf-8")

        def record_samples(payload):
            if args.save_samples:
                with sample_path.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(payload)+"\n")

        for round_id in range(args.rounds):
            for key, inputs, programs in (bundles if round_id % 2 == 0 else list(reversed(bundles))):
                report["active_case"] = key
                row = {"key": key, "round": round_id+1, "status": "running"}
                report["results"].append(row)
                save()
                calls = {label: functools.partial(program, *inputs) for label, program in programs.items()}
                values = sample(calls, round_id, args.warmup, args.pairs)
                row.update(status="passed", median_us={label: statistics.median(v) for label, v in values.items()},
                           p90_us={label: sorted(v)[int(0.9*(len(v)-1))] for label, v in values.items()})
                record_samples({"phase": "full", "key": key, "round": round_id+1, "samples_us": values})
                save()
                print(f"round={round_id+1} {key}: full_us={row['median_us']}", flush=True)

        report["phase"] = "timing_components"
        for round_id in range(args.component_rounds):
            for key, inputs, programs in (bundles if round_id % 2 == 0 else list(reversed(bundles))):
                name = report["cases"][key]["case"]
                if args.components == "none" or (args.components == "focus" and name not in COMPONENT_CASES):
                    continue
                report["active_case"] = key
                for label, program in programs.items():
                    splits = int(label)
                    if splits == 1:
                        continue
                    report["active_split"] = splits
                    row = {"key": key, "num_splits": splits, "round": round_id+1, "status": "running"}
                    report["components"].append(row)
                    save()
                    partial_call = functools.partial(program.partial_kernel, *inputs)
                    parts, lse = partial_call()
                    merge_call = functools.partial(program.merge_kernel, parts, lse)
                    values = sample({"partial": partial_call, "merge": merge_call}, round_id,
                                    args.component_warmup, args.component_pairs)
                    row.update(status="passed", median_us={part: statistics.median(v) for part, v in values.items()})
                    record_samples({"phase": "components", "key": key, "num_splits": splits,
                                    "round": round_id+1, "samples_us": values})
                    save()
                    print(f"components round={round_id+1} {key} S={splits}: {row['median_us']}", flush=True)
                    del merge_call, parts, lse
        report.pop("active_case", None)
        report.pop("active_split", None)
        report["phase"] = "complete"
        report["status"] = "passed"
    except Exception:
        report.update(status="error", error=traceback.format_exc())
        traceback.print_exc()
    finally:
        save()
    print(f"{report['status'].upper()}: {path}; paste output: {paste_path}", flush=True)
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
