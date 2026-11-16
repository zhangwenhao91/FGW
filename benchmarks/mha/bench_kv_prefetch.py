"""Compare isolated K/V prefetch timing, with a V capacity control."""
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
    parser.add_argument("--output", default="mha_kv_prefetch.json")
    parser.add_argument("--list-only", action="store_true")
    args = parser.parse_args()
    if args.rounds < 2 or args.pairs < 3 or args.warmup < 1:
        parser.error("rounds>=2, pairs>=3, warmup>=1 required")
    cases = [(129, 2048), (4096, 8192)] if args.smoke else [(q, kv) for kv in KVS for q in QS]
    if args.list_only:
        print(json.dumps({"cases": len(cases), "shapes": cases, "B": 1, "H": 1,
                          "variants": ["baseline", "k_early", "v_buffer4", "v_early", "ascendc"]}))
        return 0
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    report = {"status": "running", "phase": "compile_accuracy", "settings": vars(args),
              "cases": [], "results": [], "summary": [],
              "policy": "Q128 manual, frozen baseline split reused by all candidates; defaults unchanged",
              "scope": "baseline: K0/QK1/score3/softmax5/P6/V3/PV6/acc7, V3 buffers. k_early: K0/all other stages+1, same buffers. v_buffer4: baseline stages, V4 buffers. v_early: V2, other baseline stages, V4 buffers. Compare v_early to v_buffer4 to isolate timing.",
              "accuracy": "Both seeds7/19, shuffled/repeated pages, CPU reference rtol.01 atol.004",
              "timing": "Rotating/reversing five-way warm-cache Event full calls; includes all kernels and Q preparation"}

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
            for label in ("baseline", "k_early", "v_buffer4", "v_early"):
                report.update(phase="compile_accuracy", active=f"{key}/compile_{label}")
                save()
                split = None if label == "baseline" else programs["baseline"].num_splits
                program = impl.build_kernel(1, sq, 1, 128, pool_rows, kv//128, 128, False,
                    "bfloat16", args.device, full_kv_cache=True, num_splits=split, kv_prefetch=label)
                programs[label] = program
                assert program.num_splits == programs["baseline"].num_splits
                assert program.compute_grid == programs["baseline"].compute_grid
                kernels = {"direct": program.direct_kernel} if program.num_splits == 1 else {
                    "partial": program.partial_kernel, "merge": program.merge_kernel}
                info["kernels"][label] = {}
                for component, kernel in kernels.items():
                    source = kernel.get_kernel_source()
                    info["kernels"][label][component] = hashlib.sha256(source.encode()).hexdigest()
                    path.with_name(f"{path.stem}_{key}_{label}_{component}.cpp").write_text(source, encoding="utf-8")
            info.update(split=programs["baseline"].num_splits, compute_grid=programs["baseline"].compute_grid,
                        baseline_split_policy=programs["baseline"].split_policy)
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
                    "source_equal_baseline_v_early": info["kernels"]["baseline"] == info["kernels"]["v_early"]}
            for candidate, control in (("k_early", "baseline"), ("v_buffer4", "baseline"),
                                       ("v_early", "baseline"), ("v_early", "v_buffer4"),
                                       ("k_early", "ascendc"), ("v_early", "ascendc")):
                ratios = [r[control]/r[candidate] for r in rows]
                item[f"{candidate}_vs_{control}"] = {"speedup": statistics.median(ratios),
                    "min_round": min(ratios), "max_round": max(ratios), "faster_rounds": sum(v > 1 for v in ratios)}
            report["summary"].append(item)
            save()
        report["groups"] = {}
        for comparison in ("k_early_vs_baseline", "v_buffer4_vs_baseline", "v_early_vs_baseline",
                           "v_early_vs_v_buffer4", "k_early_vs_ascendc", "v_early_vs_ascendc"):
            values = [r[comparison]["speedup"] for r in report["summary"]]
            report["groups"][comparison] = {"cases": len(values),
                "geomean_speedup": math.exp(statistics.mean(math.log(v) for v in values)),
                "faster_cases": sum(v > 1 for v in values)}
        report.update(status="passed", phase="complete", correctness="passed", active=None)
    except Exception:
        report.update(status="error", error=traceback.format_exc())
        traceback.print_exc()
    finally:
        save()
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
