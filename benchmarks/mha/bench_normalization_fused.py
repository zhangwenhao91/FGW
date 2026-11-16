"""Compare fused SIMD normalization/cast with SIMT and separate SIMD controls."""
import argparse
import functools
import hashlib
import json
import math
from pathlib import Path
import statistics
import traceback

from benchmarks.mha.summarize_v7_b1h1_grid import QS, KVS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--rounds", type=int, default=6)
    parser.add_argument("--pairs", type=int, default=51)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--output", default="mha_normalization_fused.json")
    parser.add_argument("--list-only", action="store_true")
    args = parser.parse_args()
    if args.rounds < 2 or args.pairs < 3 or args.warmup < 1:
        parser.error("rounds>=2, pairs>=3, warmup>=1 required")
    cases = [(129, 2048), (4096, 8192)] if args.smoke else [(q, kv) for kv in KVS for q in QS]
    if args.list_only:
        print(json.dumps({"cases": len(cases), "shapes": cases, "B": 1, "H": 1,
                          "variants": ["simt", "simd", "fused", "ascendc"]}))
        return 0
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    report = {"status": "running", "phase": "compile_accuracy", "settings": vars(args),
              "cases": [], "results": [], "summary": [],
              "policy": "B1H1 BF16 D128 page128 noncausal; contiguous manual/stages2/cap28; SIMT default chooses frozen split; separate SIMD and fused reuse the same split. No default promotion",
              "scope": "Direct fused path divides FP32 accumulator, converts and writes a separate BF16 UB buffer without normalized FP32 writeback/reload. Split path retains default FP32 partials/LSE; final merge uses fused SIMD divide/cast. Separate SIMD control retains the previous experiment including SIMD partial normalization. max(total,1e-30), division and merge weights are retained; no reciprocal approximation.",
              "accuracy": "Seeds7/19 on shuffled/repeated pages with unused pool pages; chunked CPU reference with elementwise rtol1e-2/atol4e-3 plus finite/shape/dtype checks. Timing seed7.",
              "timing": "Rotating/reversing four-way warm-cache NPU Event full calls including Q preparation, allocations and all kernels. Excludes compilation, H2D, references, FIA metadata preparation.",
              "speedup": "control_us/fused_us; median paired round ratios; >1 means fused faster"}

    def save():
        path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        lines = [f"status={report['status']} phase={report['phase']} correctness={report.get('correctness')} active={report.get('active')}"]
        if report.get("error"):
            lines.append(report["error"])
        for item in report["summary"]:
            lines.append(json.dumps(item))
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
        # Timing/allocations are bounded to one shape at a time; every variant
        # for that shape must pass both seeds before any performance is measured.
        for index, (sq, kv) in enumerate(cases):
            key = f"q{sq}_kv{kv}"
            info = {"key": key, "Q": sq, "KV": kv, "correctness": {}, "kernels": {}}
            report["cases"].append(info)
            cpu_inputs = make_inputs((key, 1, sq, 1, 128, 128, kv//128), 7, torch.bfloat16, torch)
            pool_rows = cpu_inputs[1].shape[0]
            programs = {}
            for label in ("simt", "simd", "fused"):
                report.update(phase="compile_accuracy", active=f"{key}/compile_{label}")
                save()
                split = None if label == "simt" else programs["simt"].num_splits
                program = impl.build_kernel(1, sq, 1, 128, pool_rows, kv//128, 128, False,
                    "bfloat16", args.device, full_kv_cache=True, num_splits=split, output_normalization=label)
                programs[label] = program
                assert program.num_splits == programs["simt"].num_splits
                assert program.compute_grid == programs["simt"].compute_grid
                kernels = {"direct": program.direct_kernel} if program.num_splits == 1 else {
                    "partial": program.partial_kernel, "merge": program.merge_kernel}
                info["kernels"][label] = {}
                for component, kernel in kernels.items():
                    source = kernel.get_kernel_source()
                    info["kernels"][label][component] = hashlib.sha256(source.encode()).hexdigest()
                    path.with_name(f"{path.stem}_{key}_{label}_{component}.cpp").write_text(source, encoding="utf-8")
            info.update(split=programs["simt"].num_splits, compute_grid=programs["simt"].compute_grid,
                        baseline_split_policy=programs["simt"].split_policy)
            for seed in (7, 19):
                report["active"] = f"{key}/accuracy_seed{seed}"
                save()
                q, k, v, table = cpu_inputs if seed == 7 else make_inputs(
                    (key, 1, sq, 1, 128, 128, kv//128), seed, torch.bfloat16, torch)
                lengths = torch.tensor([kv], dtype=torch.int32)
                expected = chunked_cpu_reference(q, k, v, table, lengths, 128, False, 128)
                inputs = [t.to("npu") for t in (q, k, v, table, lengths)]
                calls = {label: functools.partial(program, *inputs) for label, program in programs.items()}
                native = functools.partial(torch_npu.npu_fused_infer_attention_score_v2,
                    inputs[0], inputs[1].view(-1, 128, 128), inputs[2].view(-1, 128, 128),
                    block_table=inputs[3], num_query_heads=1, num_key_value_heads=1, input_layout="BSND",
                    block_size=128, softmax_scale=128**-0.5, actual_seq_qlen=[sq], actual_seq_kvlen=[kv],
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
                    "source_equal_simt_fused": info["kernels"]["simt"] == info["kernels"]["fused"]}
            for control in ("simt", "simd", "ascendc"):
                ratios = [r[control]/r["fused"] for r in rows]
                item[f"fused_vs_{control}"] = {"speedup": statistics.median(ratios),
                    "min_round": min(ratios), "max_round": max(ratios), "faster_rounds": sum(v > 1 for v in ratios)}
            report["summary"].append(item)
            save()
        report["groups"] = {}
        for control in ("simt", "simd", "ascendc"):
            values = [r[f"fused_vs_{control}"]["speedup"] for r in report["summary"]]
            report["groups"][f"fused_vs_{control}"] = {"cases": len(values),
                "geomean_speedup": math.exp(statistics.mean(math.log(v) for v in values)),
                "faster_cases": sum(v > 1 for v in values),
                "within_2pct_cases": sum(0.98 <= v <= 1.02 for v in values)}
        report.update(status="passed", phase="complete", correctness="passed", active=None)
    except Exception:
        report.update(status="error", error=traceback.format_exc())
        traceback.print_exc()
    finally:
        save()
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
