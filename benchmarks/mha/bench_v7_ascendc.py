"""Compare current v7 with installed CANN AscendC fused infer attention on identical paged data."""
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

Q_LENGTHS = (1, 4, 16, 32, 64, 127, 128, 129, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--q-lengths", nargs="+", type=int, default=list(Q_LENGTHS))
    parser.add_argument("--fia-api", choices=["auto", "v1", "v2", "both"], default="auto",
                        help="auto prefers v2 if exposed; no silent retry after runtime/accuracy failures")
    parser.add_argument("--split-comparison", action="store_true",
                        help="Also compare forced v7 num_splits=1 against auto split and FIA")
    parser.add_argument("--bh", nargs="+", default=["2x8", "1x1"])
    parser.add_argument("--compare-persistent", action="store_true", help="Also compare experimental 28-program scheduling")
    parser.add_argument("--compare-legacy", action="store_true")
    parser.add_argument("--kv", type=int, default=2048)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--repeat", type=int, default=51)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--peak-tflops", type=float, default=378.0)
    parser.add_argument("--list-only", action="store_true")
    parser.add_argument("--output", default="mha_v7_ascendc_speedup.json")
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
    if args.list_only:
        print(json.dumps(cases, indent=2))
        return 0
    report = {"status": "running", "settings": vars(args), "results": [],
        "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "baseline": "Installed CANN FusedInferAttentionScore AscendC kernels via torch_npu; not a source build of current ops-transformer master",
        "speedup_policy": "speedup_v7_over_ascendc = AscendC median latency / v7 median latency, paired by case and round; >1 means v7 faster",
        "sources": ["https://gitcode.com/cann/ops-transformer/tree/master/attention/fused_infer_attention_score",
                    "https://github.com/Ascend/op-plugin/blob/master/op_plugin/config/op_plugin_functions.yaml"],
        "flops_policy": "4*B*H*Q*KV*128, noncausal useful QK+PV only; no padding/softmax FLOPs",
        "timing": "Full native callable, warm-cache NPU Event median; one call per sample. Excludes compile, H2D, CPU reference, metadata/list preparation and zero-copy K/V view creation. Includes output allocation and stream-visible launch gaps. v7 includes Q conversion and merge; FIA uses original paged storage without gather, same BSND Q/O.",
        "v7_variant_policy": "v7_manual_bf16=guarded current default; v7_legacy_bf16=old automatic strategy; v7_split1_bf16=explicit split1",
        "comparison": "All cases match B/H/Q/KV, values, page table and per-request lengths. BF16 Q/K/V/O, D128, page128, noncausal, scale=128**-0.5, seed7. v7 manual/stages2/Q resident. FIA inner_precise=0, sparse_mode=0, no mask, no quantization. No source installation or environment modification."}
    if args.split_comparison:
        report["comparison"] += " Includes forced num_splits=1 candidate, same tuning otherwise; measures whole strategy change, not isolated merge time. Default auto policy selects cost_model only for eligible full-cache configurations; otherwise legacy."
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
        from operators.mha.split_cost_model import MODEL_PATH
        report["cost_model_sha256"] = hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest()
        report["cost_model_weights"] = json.loads(MODEL_PATH.read_text(encoding="utf-8"))["weights"]
        report["v7_sha256"] = hashlib.sha256(Path(v7.__file__).read_bytes()).hexdigest()
        report["api_probe"] = {}
        names = {"v2": "npu_fused_infer_attention_score_v2", "v1": "npu_fused_infer_attention_score"}
        available = {}
        for api, name in names.items():
            function = getattr(torch_npu, name, None)
            available[api] = callable(function)
            entry = {"python_name": name, "available": available[api]}
            try:
                entry["schemas"] = str(getattr(torch.ops.npu, name)._schemas)
            except Exception as exc:
                entry["schema_probe_error"] = str(exc)
            report["api_probe"][api] = entry
        if args.fia_api == "auto":
            selected = [next((api for api in ("v2", "v1") if available[api]), "v2")]
        elif args.fia_api == "both":
            selected = ["v2", "v1"]
        else:
            selected = [args.fia_api]
        report["selected_apis"] = selected
        report["cann_environment"] = {key: os.environ.get(key) for key in
            ("ASCEND_HOME_PATH", "ASCEND_TOOLKIT_HOME", "ASCEND_OPP_PATH", "ASCEND_RT_VISIBLE_DEVICES")}
        print(f"FIA API availability={available}; selected={selected}", flush=True)
        save()
        if not any(available.values()):
            raise RuntimeError("No torch_npu fused infer attention interface is exposed; see api_probe")

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
            geometry = {}
            variants = (("v7_manual_bf16", "v7_split1_bf16") if args.split_comparison else ("v7_manual_bf16",)) + tuple("ascendc_fia_" + api for api in selected)
            if args.compare_persistent:
                variants += ("v7_persistent28_bf16",)
            if args.compare_legacy:
                variants += ("v7_legacy_bf16",)
            # BnBsH physical layout: views only, same bytes and physical page IDs.
            paged_k = inputs[1].view(b * width, 128, h * 128)
            paged_v = inputs[2].view(b * width, 128, h * 128)
            kv_lengths = lengths.tolist()
            q_lengths = [sq] * b
            for variant in variants:
                try:
                    if variant in ("v7_manual_bf16", "v7_split1_bf16", "v7_legacy_bf16", "v7_persistent28_bf16"):
                        kernel = v7.build_kernel(b, sq, h, 128, k.shape[0], width, 128, False,
                            "bfloat16", args.device, schedule_mode="manual",
                            num_splits=1 if variant == "v7_split1_bf16" else None,
                            split_policy="legacy" if variant == "v7_legacy_bf16" else "auto", full_kv_cache=True,
                            persistent_programs=28 if variant == "v7_persistent28_bf16" else None)
                        call = functools.partial(kernel, *inputs)
                        got = call().cpu()
                        kernel_mode = kernel.execution_mode
                        geometry[variant] = {"split_policy": kernel.split_policy, "num_splits": kernel.num_splits,
                            "query_tile_tasks": kernel.query_tile_tasks, "compute_grid": kernel.compute_grid,
                            "compute_work": kernel.compute_work, "persistent_programs": kernel.persistent_programs}
                    else:
                        api = variant.rsplit("_", 1)[1]
                        if not available[api]:
                            raise RuntimeError(f"torch_npu.{names[api]} unavailable; see api_probe")
                        kwargs = dict(block_table=inputs[3], num_key_value_heads=h,
                            input_layout="BSND", block_size=128, sparse_mode=0, inner_precise=0)
                        if api == "v2":
                            kwargs.update(num_query_heads=h, softmax_scale=128**-0.5,
                                actual_seq_qlen=q_lengths, actual_seq_kvlen=kv_lengths,
                                return_softmax_lse=False)
                        else:
                            kwargs.update(num_heads=h, scale=128**-0.5,
                                actual_seq_lengths=q_lengths, actual_seq_lengths_kv=kv_lengths,
                                softmax_lse_flag=False)
                        native_call = functools.partial(getattr(torch_npu, names[api]), inputs[0], paged_k, paged_v, **kwargs)
                        def call(native=native_call):
                            return native()[0]
                        got = call().cpu()
                        kernel_mode = "installed_cann_fused_infer_paged"
                    if got.dtype != torch.bfloat16 or tuple(got.shape) != tuple(expected.shape):
                        raise ValueError(f"Unexpected output dtype/shape: {got.dtype} {got.shape}")
                    torch.testing.assert_close(got.float(), expected.float(), rtol=1e-2, atol=4e-3)
                    correctness[variant] = {"max_abs_error": (got.float()-expected.float()).abs().max().item(),
                        "rtol": 1e-2, "atol": 4e-3}
                    programs[variant] = (call, kernel_mode)
                    if variant == "v7_persistent28_bf16" and (b,h,sq,args.kv)==(2,8,4096,2048):
                        source = kernel.direct_kernel.get_kernel_source()
                        source_path = path.with_name(path.stem+"_persistent_device.cpp")
                        source_path.write_text(source,encoding="utf-8")
                        report["persistent_generated_source"] = str(source_path)
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
                        "kernel_mode": programs[label][1], "geometry": geometry.get(label), "median_us": median*1000,
                        "effective_flops": flops, "effective_tflops": tflops,
                        "mfu_percent": 100*tflops/args.peak_tflops, "samples_ms": samples[label]}
                    report["results"].append(row)
                    print(f"{track} Q={sq} {label} round={round_id+1}: {median*1000:.3f} us MFU={row['mfu_percent']:.4f}%", flush=True)
                for v7_label in ("v7_manual_bf16", "v7_split1_bf16", "v7_legacy_bf16", "v7_persistent28_bf16"):
                    if v7_label not in samples:
                        continue
                    v7_ms = statistics.median(samples[v7_label])
                    for api in selected:
                        label = "ascendc_fia_" + api
                        if label in samples:
                            fia_ms = statistics.median(samples[label])
                            comparison = {"track": track, "B": b, "H": h, "Q": sq, "KV": args.kv,
                                "round": round_id+1, "candidate": v7_label, "geometry": geometry[v7_label],
                                "baseline": label, "v7_median_us": v7_ms*1000,
                                "ascendc_median_us": fia_ms*1000, "speedup_v7_over_ascendc": fia_ms/v7_ms,
                                "v7_latency_change_percent": 100*(v7_ms/fia_ms-1)}
                            report.setdefault("comparisons", []).append(comparison)
                            print(f"COMPARE {track} Q={sq} KV={args.kv} round={round_id+1} {v7_label} / {label}: V7 speedup={fia_ms/v7_ms:.4f}x", flush=True)
                if all(label in samples for label in ("v7_manual_bf16", "v7_split1_bf16")):
                    auto_ms = statistics.median(samples["v7_manual_bf16"])
                    forced_ms = statistics.median(samples["v7_split1_bf16"])
                    report.setdefault("split_comparisons", []).append({"track": track, "B": b, "H": h,
                        "Q": sq, "KV": args.kv, "round": round_id+1,
                        "auto_num_splits": geometry["v7_manual_bf16"]["num_splits"],
                        "auto_median_us": auto_ms*1000, "split1_median_us": forced_ms*1000,
                        "speedup_split1_over_auto": auto_ms/forced_ms,
                        "latency_saved_us": (auto_ms-forced_ms)*1000})
                if all(label in samples for label in ("v7_manual_bf16", "v7_legacy_bf16")):
                    current_ms = statistics.median(samples["v7_manual_bf16"])
                    legacy_ms = statistics.median(samples["v7_legacy_bf16"])
                    report.setdefault("policy_comparisons", []).append({"track": track, "Q": sq, "KV": args.kv,
                        "round": round_id+1, "current_policy": geometry["v7_manual_bf16"],
                        "legacy_policy": geometry["v7_legacy_bf16"],
                        "current_median_us": current_ms*1000, "legacy_median_us": legacy_ms*1000,
                        "speedup_current_over_legacy": legacy_ms/current_ms})
                if all(label in samples for label in ("v7_manual_bf16", "v7_persistent28_bf16")):
                    normal_ms = statistics.median(samples["v7_manual_bf16"])
                    persistent_ms = statistics.median(samples["v7_persistent28_bf16"])
                    report.setdefault("persistent_comparisons", []).append({"track": track, "Q": sq, "KV": args.kv,
                        "round": round_id+1, "normal_geometry": geometry["v7_manual_bf16"],
                        "persistent_geometry": geometry["v7_persistent28_bf16"],
                        "normal_median_us": normal_ms*1000,"persistent_median_us":persistent_ms*1000,
                        "speedup_persistent_over_normal": normal_ms/persistent_ms})
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
