"""Runtime Q/pool/page-table shapes versus current static v7 and native FIA."""
import argparse
import functools
import hashlib
import json
import math
from pathlib import Path
import statistics
import traceback
import time

from benchmarks.mha.summarize_v7_b1h1_grid import QS, KVS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--rounds", type=int, default=6)
    parser.add_argument("--pairs", type=int, default=51)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--output", default="mha_runtime_lengths.json")
    parser.add_argument("--list-only", action="store_true")
    parser.add_argument("--probe-split", type=int, choices=[1, 4])
    args = parser.parse_args()
    if args.rounds < 2 or args.pairs < 3 or args.warmup < 1:
        parser.error("rounds>=2, pairs>=3, warmup>=1 required")
    cases = [(129, 512), (257, 2048), (4096, 8192), (8191, 4096)] if args.smoke else [(q, kv) for kv in KVS for q in QS]
    if args.list_only:
        print(json.dumps({"cases": len(cases), "shapes": cases, "B": 1, "H": 1,
                          "variants": ["static", "dynamic", "ascendc"]}))
        return 0
    if args.smoke and args.probe_split is None:
        args.probe_split = 1
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    report = {"status": "running", "phase": "compile_accuracy", "settings": vars(args),
              "cases": [], "results": [], "summary": [],
              "policy": "B1H1 BF16 D128 page128 noncausal; current contiguous manual/stages2/cap28, Fast Math default; identical split and merge-row tile for static/dynamic full measurements; no default promotion",
              "runtime": "Q length, physical pool rows and page-table width use T.dynamic; logical KV length remains cache_seqlens device data; B/H/split/merge-row tile specialize reusable families",
              "accuracy": "Seeds7/19 and live lengths KV and KV-128; shuffled/repeated pages, unused pool pages; CPU FP32 chunked reference, elementwise rtol1e-2/atol4e-3, finite/shape/dtype checks. Timing uses seed7 full-live KV.",
              "timing": "Rotating/reversing three-way warm-cache NPU Event full calls. Includes runtime wrapper shape validation, Q preparation, allocation, dynamic launch and any merge. Excludes host family/split selection, compilation, H2D, references, FIA metadata preparation. Compile wall time reported separately; not steady-state latency.",
              "speedup": "static_us/dynamic_us and AscendC_us/dynamic_us; median paired round ratios; >1 means dynamic faster"}

    def save():
        path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        lines = [f"status={report['status']} phase={report['phase']} correctness={report.get('correctness')} active={report.get('active')}"]
        if report.get("error"):
            lines.append(report["error"])
        for item in report["summary"]:
            lines.append(json.dumps(item))
        if report.get("family_reuse"):
            lines.append("REUSE: " + json.dumps({"dynamic_family_count": report["dynamic_family_count"], "static_shape_count": report["static_shape_count"], "family_cache": report["family_cache"], "families": report["family_reuse"]}))
        if report.get("groups"):
            lines.append("GROUPS: " + json.dumps(report["groups"]))
        if report["status"] == "error":
            lines.extend("CASE " + json.dumps(item) for item in report["cases"])
        path.with_name(path.stem + "_paste.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    save()
    try:
        import torch
        import torch_npu
        import tilelang
        from operators.mha.kernels import mha_paged_decode_ascend_v7 as impl
        from operators.mha.split_cost_model import MODEL_PATH
        from benchmarks.mha.bench_mha_mfu import chunked_cpu_reference
        from tests.mha.test_mha_paged_accuracy import make_inputs
        torch.npu.set_device(args.device)
        report["device"] = str(torch.npu.get_device_properties(args.device))
        report["versions"] = {"torch": torch.__version__, "torch_npu": torch_npu.__version__, "tilelang": tilelang.__version__}
        report["source_sha256"] = hashlib.sha256(Path(impl.__file__).read_bytes()).hexdigest()
        report["split_model_sha256"] = hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest()
        from operators.mha.runtime_lengths import build_runtime_kernel
        build_runtime_kernel.cache_clear()
        families = {}
        # Timing/allocations are bounded to one shape at a time; every variant
        # for that shape must pass both seeds before any performance is measured.
        for index, (sq, kv) in enumerate(cases):
            key = f"q{sq}_kv{kv}"
            info = {"key": key, "Q": sq, "KV": kv, "correctness": {}, "kernels": {}}
            report["cases"].append(info)
            cpu_inputs = make_inputs((key, 1, sq, 1, 128, 128, kv//128), 7, torch.bfloat16, torch)
            pool_rows = cpu_inputs[1].shape[0]
            programs = {}
            info["compile_wall_seconds"] = {}
            report.update(phase="compile_accuracy", active=f"{key}/compile_static")
            save()
            started = time.perf_counter()
            baseline = impl.build_kernel(1, sq, 1, 128, pool_rows, kv//128, 128, False,
                "bfloat16", args.device, full_kv_cache=True, num_splits=args.probe_split)
            programs["static"] = baseline
            info["compile_wall_seconds"]["static"] = time.perf_counter() - started
            split = baseline.num_splits
            merge_rows = min(8, 1 << (sq - 1).bit_length()) if split > 1 else 8
            report["active"] = f"{key}/compile_dynamic"
            save()
            started = time.perf_counter()
            runtime = build_runtime_kernel(1, 1, split, merge_rows, args.device)
            programs["dynamic"] = runtime
            info["compile_wall_seconds"]["dynamic_family_lookup_or_compile"] = time.perf_counter() - started
            family = str(runtime.family_key)
            if family in families:
                assert families[family]["program"] is runtime, "Dynamic family was rebuilt"
            else:
                families[family] = {"program": runtime, "shapes": []}
            families[family]["shapes"].append([sq, kv])
            info.update(split=split, merge_rows=merge_rows, family_key=family,
                        compute_grid=baseline.compute_grid, default_split_policy=baseline.split_policy)
            for label, callable_program in programs.items():
                program = callable_program.program if label == "dynamic" else callable_program
                kernels = {"direct": program.direct_kernel} if split == 1 else {
                    "partial": program.partial_kernel, "merge": program.merge_kernel}
                info["kernels"][label] = {}
                for component, kernel in kernels.items():
                    source = kernel.get_kernel_source()
                    info["kernels"][label][component] = hashlib.sha256(source.encode()).hexdigest()
                    path.with_name(f"{path.stem}_{key}_{label}_{component}.cpp").write_text(source, encoding="utf-8")
            for seed in (7, 19):
                report["active"] = f"{key}/accuracy_seed{seed}"
                save()
                q, k, v, table = cpu_inputs if seed == 7 else make_inputs(
                    (key, 1, sq, 1, 128, 128, kv//128), seed, torch.bfloat16, torch)
                live = kv if seed == 7 else kv - 128
                lengths = torch.tensor([live], dtype=torch.int32)
                expected = chunked_cpu_reference(q, k, v, table, lengths, 128, False, 128)
                inputs = [t.to("npu") for t in (q, k, v, table, lengths)]
                calls = {label: functools.partial(program, *inputs) for label, program in programs.items()}
                native = functools.partial(torch_npu.npu_fused_infer_attention_score_v2,
                    inputs[0], inputs[1].view(-1, 128, 128), inputs[2].view(-1, 128, 128),
                    block_table=inputs[3], num_query_heads=1, num_key_value_heads=1, input_layout="BSND",
                    block_size=128, softmax_scale=128**-0.5, actual_seq_qlen=[sq], actual_seq_kvlen=[live],
                    sparse_mode=0, inner_precise=0, return_softmax_lse=False)
                calls["ascendc"] = lambda native=native: native()[0]
                for label, call in calls.items():
                    got = call().cpu()
                    assert got.shape == q.shape and got.dtype == q.dtype and torch.isfinite(got).all()
                    torch.testing.assert_close(got, expected, rtol=1e-2, atol=4e-3)
                    info["correctness"][f"{label}/seed{seed}"] = {
                        "status": "passed", "max_abs_error": (got.float()-expected.float()).abs().max().item()}
                if seed == 7:
                    timed_calls = calls
            report["phase"] = "timing"
            start, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)

            def measure(call):
                start.record()
                output = call()
                end.record()
                end.synchronize()
                us = start.elapsed_time(end)*1000
                assert math.isfinite(us) and us > 0
                return us

            labels = tuple(timed_calls)
            rows = []
            for rd in range(args.rounds):
                report["active"] = f"{key}/round{rd+1}"
                save()
                for _ in range(args.warmup):
                    for label in (labels if rd % 2 == 0 else tuple(reversed(labels))):
                        timed_calls[label]()
                torch.npu.synchronize()
                samples = {label: [] for label in labels}
                for pair in range(args.pairs):
                    offset = (pair+rd) % len(labels)
                    order = labels[offset:] + labels[:offset]
                    if ((pair+rd)//len(labels)) % 2:
                        order = tuple(reversed(order))
                    for label in order:
                        samples[label].append(measure(timed_calls[label]))
                medians = {label: statistics.median(v) for label, v in samples.items()}
                row = {"key": key, "round": rd+1, "median_us": medians, "samples_us": samples}
                report["results"].append(row)
                rows.append(medians)
                print(f"{index+1}/{len(cases)} {key} round{rd+1}: {medians}", flush=True)
            item = {"key": key, "split": info["split"], "rounds": len(rows),
                    "median_us": {label: statistics.median(r[label] for r in rows) for label in labels},
                    "max_abs_error": {label: max(info["correctness"][f"{label}/seed{seed}"]["max_abs_error"] for seed in (7, 19)) for label in labels},
                    "family_key": family, "compile_wall_seconds": info["compile_wall_seconds"]}
            for control in ("static", "ascendc"):
                ratios = [r[control]/r["dynamic"] for r in rows]
                item[f"dynamic_vs_{control}"] = {"speedup": statistics.median(ratios),
                    "min_round": min(ratios), "max_round": max(ratios), "faster_rounds": sum(v > 1 for v in ratios)}
            report["summary"].append(item)
            save()
        report["groups"] = {}
        for control in ("static", "ascendc"):
            values = [r[f"dynamic_vs_{control}"]["speedup"] for r in report["summary"]]
            report["groups"][f"dynamic_vs_{control}"] = {"cases": len(values),
                "geomean_speedup": math.exp(statistics.mean(math.log(v) for v in values)),
                "faster_cases": sum(v > 1 for v in values),
                "within_2pct_cases": sum(0.98 <= v <= 1.02 for v in values)}
        report["family_reuse"] = {key: {"shape_calls": len(value["shapes"]), "shapes": value["shapes"]}
                                  for key, value in families.items()}
        report["dynamic_family_count"] = len(families)
        report["static_shape_count"] = len(cases)
        report["family_cache"] = build_runtime_kernel.cache_info()._asdict()
        if args.smoke:
            assert len(families) == 1 and len(cases) == 4, "Smoke must reuse exactly one family across four shapes"
        report.update(status="passed", phase="complete", correctness="passed", active=None)
    except Exception:
        report.update(status="error", error=traceback.format_exc())
        traceback.print_exc()
    finally:
        save()
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
