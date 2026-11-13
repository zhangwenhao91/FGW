"""Measure unchanged general Cube MFU and paired speedup against native FIA v2."""
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

from benchmarks.mha.bench_cube_vector import CASES as STANDARD
from benchmarks.mha.bench_mha_mfu import useful_flops


LONG_Q = [(f"long_d{d}_q{q}_kv2048", 2, q, 2, d, 128, 16, (2048, 2048), False)
          for d in (64, 128) for q in (4096, 32768)]
LABELS = ("cube_general", "ascendc_fia_v2")
FIA_DOC = "https://github.com/Ascend/op-plugin/blob/master/docs/zh/custom_APIs/torch_npu/torch_npu-npu_fused_infer_attention_score_v2.md"


def fia_skip_reason(case):
    page = case[5]
    if page < 128 or page > 512 or page % 128:
        return "Native FIA paged API documents block_size=128..512, divisible by128; preserve original page layout, no repacking"
    return None


def performance(flops, us, peak_tflops):
    tflops = flops / (us * 1e-6) / 1e12
    return {"us": us, "effective_tflops": tflops, "mfu_percent": 100*tflops/peak_tflops}


def aggregate(results, cases, peak):
    summaries = []
    for key, info in cases.items():
        rows = [r for r in results if r["key"] == key and r["status"] == "passed"]
        if not rows:
            continue
        metrics = {label: performance(info["effective_flops"], statistics.median(r["median_us"][label] for r in rows), peak)
                   for label in rows[0]["median_us"]}
        summary = {"key": key, "dtype": info["dtype"], "suite": info["suite"],
                   "rounds": len(rows), "metrics": metrics,
                   "ascendc_status": info["ascendc_status"]}
        if LABELS[1] in metrics:
            ratios = [r["speedup_cube_over_ascendc"] for r in rows]
            summary.update(speedup_cube_over_ascendc=metrics[LABELS[1]]["us"] / metrics[LABELS[0]]["us"],
                           min_round_speedup=min(ratios), max_round_speedup=max(ratios),
                           cube_faster_rounds=sum(v > 1 for v in ratios))
        summaries.append(summary)
    groups = {}
    for group in ("all", "standard", "long-q", "float16", "bfloat16"):
        selected = [s for s in summaries if group == "all" or s["suite"] == group or s["dtype"] == group]
        if not selected:
            continue
        paired = [s for s in selected if "speedup_cube_over_ascendc" in s]
        best = max(selected, key=lambda s: s["metrics"][LABELS[0]]["mfu_percent"])
        item = {"cube_cases": len(selected), "paired_cases": len(paired),
                "cube_only_cases": len(selected)-len(paired), "max_cube_mfu_percent": best["metrics"][LABELS[0]]["mfu_percent"],
                "max_cube_mfu_case": best["key"]}
        if paired:
            ratios = [s["speedup_cube_over_ascendc"] for s in paired]
            item.update(geomean_speedup=math.exp(statistics.mean(math.log(v) for v in ratios)),
                        cube_faster_cases=sum(v > 1 for v in ratios), min_speedup=min(ratios), max_speedup=max(ratios))
        groups[group] = item
    return summaries, groups


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=["standard", "long-q", "all"], default="all")
    parser.add_argument("--cases", nargs="+", choices=[c[0] for c in STANDARD + LONG_Q])
    parser.add_argument("--dtypes", nargs="+", choices=["float16", "bfloat16"], default=["float16", "bfloat16"])
    parser.add_argument("--rounds", type=int, default=6)
    parser.add_argument("--pairs", type=int, default=31)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--peak-tflops", type=float, default=378.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--output", default="mha_cube_ascendc_mfu_summary.json")
    parser.add_argument("--save-samples", action="store_true", help="Save optional local per-pair JSONL alongside compact summary")
    parser.add_argument("--list-only", action="store_true")
    args = parser.parse_args()
    if (args.rounds < 2 or args.pairs < 3 or args.warmup < 1
            or not math.isfinite(args.peak_tflops) or args.peak_tflops <= 0):
        parser.error("rounds>=2, pairs>=3, warmup>=1, finite positive peak required")
    chosen = {"standard": STANDARD, "long-q": LONG_Q, "all": STANDARD+LONG_Q}[args.suite]
    if args.cases and set(args.cases)-{c[0] for c in chosen}:
        parser.error("Selected case is not in this suite")
    cases = [c for c in chosen if not args.cases or c[0] in args.cases]
    if args.list_only:
        print(json.dumps({"shapes": len(cases), "cube_cases": len(cases)*len(args.dtypes),
                          "paired_cases": sum(fia_skip_reason(c) is None for c in cases)*len(args.dtypes),
                          "cases": [{"spec": c, "ascendc_skip_reason": fia_skip_reason(c)} for c in cases]}, indent=2))
        return 0
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    report = {"status": "running", "phase": "compile_and_accuracy", "settings": vars(args), "cases": {}, "results": [],
              "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "implementation": "Unchanged general Cube: BR32/BC32, split=1; not optimized default v7 Cube",
              "baseline": "Installed CANN native FIA v2 via torch_npu, original paged K/V views; no dense gather/repacking, no source rebuild",
              "input_policy": "Identical finite random Q/K/V pool, shuffled/shared/repeated pages and runtime lengths for both variants; correctness uses only logically visible tokens",
              "timing": "Warm-cache NPU Event per prebuilt full device callable; includes allocation effects, all kernels and stream-visible launch gaps. Excludes compile, H2D, CPU reference, API metadata host synchronization, list/mask construction and zero-copy K/V view creation. Not CPU wall time or msprof timing.",
              "flops_policy": "4*H*D*sum_b,sum_q(visible_keys_bq), original unpadded D; causal visible=max(KV_b-Q+q+1,0). Includes useful QK+PV only, no softmax/padding/masked FLOPs. Same numerator and dense FP16/BF16 Cube peak for both variants.",
              "speedup_policy": "AscendC_us / general_Cube_us; >1 means general Cube faster. Unsupported page layouts are Cube-only, excluded from speedup aggregates.",
              "fia_documentation": FIA_DOC}

    def save():
        report["summary"], report["groups"] = aggregate(report["results"], report["cases"], args.peak_tflops)
        tmp = path.with_suffix(path.suffix+".tmp")
        tmp.write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
        tmp.replace(path)

    save()
    try:
        import torch
        import torch_npu
        import tilelang
        from operators.mha.kernels import mha_paged_decode_ascend_cube_general as cube
        from benchmarks.mha.bench_mha_mfu import chunked_cpu_reference
        from tests.mha.test_mha_paged_accuracy import make_inputs
        torch.npu.set_device(args.device)
        device = torch.device("npu", args.device)
        props = torch.npu.get_device_properties(args.device)
        if getattr(props, "cube_core_num", None) != 28 and args.peak_tflops == 378.0:
            raise ValueError("378 TFLOPS requires confirmed 28 Cube cores; override --peak-tflops otherwise")
        report["device"] = str(props)
        report["versions"] = {"torch": torch.__version__, "torch_npu": torch_npu.__version__, "tilelang": tilelang.__version__}
        report["kernel_sha256"] = hashlib.sha256(Path(cube.__file__).read_bytes()).hexdigest()
        report["benchmark_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        fia = getattr(torch_npu, "npu_fused_infer_attention_score_v2", None)
        if not callable(fia):
            raise RuntimeError("Previously validated torch_npu FIA v2 API is unavailable")
        report["fia_schema"] = str(torch.ops.npu.npu_fused_infer_attention_score_v2._schemas)
        # Compressed right-down causal mode uses a fixed 2048x2048 mask.
        # True means masked. Construct once, outside all timed regions.
        causal_mask = torch.ones((2048, 2048), dtype=torch.bool).triu(1).to(device)
        bundles = []
        for case, dtype_name in itertools.product(cases, args.dtypes):
            name, b, sq, h, d, p, w, lengths_list, causal = case
            key = f"{name}/{dtype_name}"
            reason = fia_skip_reason(case)
            info = {"case": name, "dtype": dtype_name, "suite": "long-q" if case in LONG_Q else "standard",
                    "B": b, "Q": sq, "H": h, "D": d, "page_size": p, "table_width": w,
                    "lengths": lengths_list, "causal": causal,
                    "effective_flops": useful_flops(lengths_list, sq, h, d, causal),
                    "ascendc_status": "skipped" if reason else "pending", "ascendc_skip_reason": reason,
                    "correctness": {}}
            report["cases"][key] = info
            report["active_case"] = key
            print(f"Preparing {key}: AscendC={'skip: '+reason if reason else 'native FIA v2'}", flush=True)
            save()
            dtype = getattr(torch, dtype_name)
            q, k, v, table = make_inputs((name, b, sq, h, d, p, w), args.seed, dtype, torch)
            lengths = torch.tensor(lengths_list, dtype=torch.int32)
            expected = chunked_cpu_reference(q, k, v, table, lengths, p, causal, 128)
            inputs = [x.to(device) for x in (q, k, v, table, lengths)]
            report["active_variant"] = LABELS[0]
            save()
            program = cube.build_kernel(b, sq, h, d, k.shape[0], w, p, causal, dtype_name, args.device)
            calls = {LABELS[0]: functools.partial(program, *inputs)}
            source = program.direct_kernel.get_kernel_source()
            info["cube_mmad_calls"] = source.count("asc_mmad(")
            assert info["cube_mmad_calls"] >= 2
            info["cube_generated_sha256"] = hashlib.sha256(source.encode()).hexdigest()
            if not reason:
                # BnBsH views share the original bytes and physical page IDs.
                paged_k = inputs[1].view(k.shape[0] // p, p, h*d)
                paged_v = inputs[2].view(k.shape[0] // p, p, h*d)
                empty_rows = causal and any(length < sq for length in lengths_list)
                inner_precise = 2 if empty_rows else 0
                kwargs = dict(block_table=inputs[3], num_query_heads=h, num_key_value_heads=h,
                              input_layout="BSND", block_size=p, softmax_scale=d**-0.5,
                              actual_seq_qlen=[sq]*b, actual_seq_kvlen=list(lengths_list),
                              sparse_mode=3 if causal else 0, inner_precise=inner_precise,
                              return_softmax_lse=False)
                if causal:
                    kwargs["atten_mask"] = causal_mask
                native = functools.partial(fia, inputs[0], paged_k, paged_v, **kwargs)
                calls[LABELS[1]] = lambda native=native: native()[0]
                info["fia_config"] = {"sparse_mode": kwargs["sparse_mode"], "inner_precise": inner_precise,
                                      "input_layout": "BSND", "block_size": p, "mask_shape": [2048, 2048] if causal else None}
            visible = lengths[:, None].expand(b, sq)
            if causal:
                visible = (visible-sq+torch.arange(sq)[None, :]+1).clamp_min(0)
            for label, call in calls.items():
                report["active_variant"] = label
                save()
                got = call().cpu()
                assert got.shape == q.shape and got.dtype == dtype and got.is_contiguous()
                assert torch.isfinite(got).all(), "Nonfinite output"
                rtol, atol = (1e-2, 4e-3) if dtype == torch.bfloat16 else (2e-3, 2e-3)
                torch.testing.assert_close(got, expected, rtol=rtol, atol=atol)
                assert torch.count_nonzero(got[visible == 0]).item() == 0, "Empty rows must be zero"
                info["correctness"][label] = {"status": "passed", "rtol": rtol, "atol": atol,
                                              "max_abs_error": (got.float()-expected.float()).abs().max().item()}
            if not reason:
                info["ascendc_status"] = "passed"
            bundles.append((key, calls))
            save()
        report.pop("active_variant", None)
        report["phase"] = "timing"
        report["correctness"] = "passed"
        start, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)

        def measure(call):
            start.record()
            output = call()
            end.record()
            end.synchronize()
            us = start.elapsed_time(end)*1000
            if not math.isfinite(us) or us <= 0:
                raise ValueError(f"Invalid Event time: {us}")
            return us

        sample_path = path.with_name(path.stem+"_samples.jsonl")
        if args.save_samples:
            sample_path.write_text("", encoding="utf-8")
            report["local_samples_file"] = sample_path.name
        for round_id in range(args.rounds):
            for key, calls in (bundles if round_id % 2 == 0 else list(reversed(bundles))):
                report["active_case"] = key
                row = {"key": key, "round": round_id+1, "status": "running"}
                report["results"].append(row)
                save()
                labels = tuple(calls)
                for _ in range(args.warmup):
                    for label in (labels if round_id % 2 == 0 else tuple(reversed(labels))):
                        calls[label]()
                torch.npu.synchronize()
                samples = {label: [] for label in labels}
                for pair in range(args.pairs):
                    for label in (labels if (pair+round_id) % 2 == 0 else tuple(reversed(labels))):
                        samples[label].append(measure(calls[label]))
                medians = {label: statistics.median(values) for label, values in samples.items()}
                metrics = {label: performance(report["cases"][key]["effective_flops"], us, args.peak_tflops)
                           for label, us in medians.items()}
                row.update(status="passed", median_us=medians, metrics=metrics,
                           p90_us={label: sorted(v)[int(0.9*(len(v)-1))] for label, v in samples.items()})
                if LABELS[1] in calls:
                    row["speedup_cube_over_ascendc"] = medians[LABELS[1]] / medians[LABELS[0]]
                    row["median_pair_speedup"] = statistics.median(a/c for a, c in zip(samples[LABELS[1]], samples[LABELS[0]]))
                if args.save_samples:
                    with sample_path.open("a", encoding="utf-8") as stream:
                        stream.write(json.dumps({"key": key, "round": round_id+1, "samples_us": samples})+"\n")
                save()
                print(f"round={round_id+1} {key}: median_us={medians} Cube_MFU={metrics[LABELS[0]]['mfu_percent']:.6f}% speedup={row.get('speedup_cube_over_ascendc', 'N/A')}", flush=True)
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
