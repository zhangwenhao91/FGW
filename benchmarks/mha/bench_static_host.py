"""Static host plan vs default v7, previously best uniform split, and installed FIA."""

import argparse
import functools
import hashlib
import json
import math
from pathlib import Path
import statistics
import traceback
import os
from benchmarks.mha.bench_split_breakdown import compiler_temp
from operators.mha.static_schedule import make_plan
from benchmarks.mha.resource_guard import cpu_environment, progress, runtime_policy

CASES = ((8192, 32768, 2), (4096, 8192, 1), (2048, 24576, 4), (3584, 32768, 1), (7168, 32768, 1))
SMOKE = ((1, 512, 1), (17, 512, 1), (129, 512, 1), (3712, 512, 1), (3584, 512, 1))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--smoke", action="store_true", help="Accuracy only, including tails/mixed/direct/no-merge"
    )
    parser.add_argument("--pairs", type=int, default=7)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--output", default="mha_static_host.json")
    parser.add_argument("--list-only", action="store_true")
    args = parser.parse_args()
    if args.pairs < 3 or args.warmup < 1:
        parser.error("pairs>=3, warmup>=1 required")
    cases = SMOKE if args.smoke else CASES
    if args.list_only:
        print(json.dumps([dict(**make_plan(q, kv).describe(), uniform_split=s) for q, kv, s in cases]))
        return 0
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    report = dict(
        status="running",
        phase="setup",
        active=None,
        settings=vars(args),
        cases=[],
        rounds=[],
        summary=[],
        scope="B1H1 BF16 D128/page128 full live KV, noncausal, static contiguous host plan, no device queue. Default code/model unchanged.",
        timing="Full calls include Q preparation, workspace/output allocations, all kernels. Host plan construction/H2D upload and compilation excluded. Immediate equal warmup per Event sample.",
        local_contract="static/local is the local JIT on prepared Q with normal allocations; static/merge updates only boundary rows in a fixed valid final-output fixture in place; static/chain runs local then merge on prepared Q. No-merge plans omit merge/chain timing.",
        prototype_cost="Local JIT allocates Parts/LSE/final O each call. No-fragment plan uses unused one-slot dummy workspace. No merge kernel is launched for it.",
    )

    last_stage = [None]
    def save():
        stage = (report['phase'], report['active'])
        if stage != last_stage[0]:
            progress(path, f"{stage[0]} active={stage[1]}")
            last_stage[0] = stage
        path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf8")
        lines = [
            f"status={report['status']} phase={report['phase']} active={report['active']} completed={len(report['summary'])}/{len(cases)}"
        ]
        for c in report["cases"]:
            lines.append(
                "CASE " + json.dumps({k: v for k, v in c.items() if k not in ("accuracy", "builds")})
            )
        lines.extend("RESULT " + json.dumps(r) for r in report["summary"])
        if report.get("error"):
            lines.append(report["error"])
        path.with_name(path.stem + "_paste.txt").write_text("\n".join(lines) + "\n", encoding="utf8")

    save()
    try:
        report["compiler_tmp"] = compiler_temp(path)
        os.environ["TILELANG_DISABLE_CACHE"] = "1"
        threads = cpu_environment()
        report['resource_policy'] = runtime_policy()
        report.update(phase='import_torch',active=None);save()
        import torch
        torch.set_num_threads(threads)
        report['cpu_threads'] = torch.get_num_threads()
        report.update(phase='import_torch_npu');save()
        import torch_npu
        report.update(phase='import_tilelang');save()
        import tilelang
        from tilelang.env import env
        from operators.mha.kernels import mha_static_host_candidate as candidate
        from operators.mha.kernels import mha_paged_decode_ascend_v7 as default
        from operators.mha import static_schedule
        from operators.mha.split_cost_model import MODEL_PATH
        from benchmarks.mha.bench_mha_mfu import chunked_cpu_reference
        from tests.mha.test_mha_paged_accuracy import make_inputs

        env.disable_cache()
        assert not env.is_cache_enabled()
        report.update(phase='npu_set_device');save()
        torch.npu.set_device(args.device)
        props = torch.npu.get_device_properties(args.device)
        assert getattr(props, "cube_core_num", None) == 28
        report.update(
            device=str(props),
            versions=dict(
                torch=torch.__version__, torch_npu=torch_npu.__version__, tilelang=tilelang.__version__
            ),
            source_sha256={
                name: hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest()
                for name, m in [("candidate", candidate), ("default", default), ("plan", static_schedule)]
            },
            model_sha256=hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest(),
        )
        start, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)

        def measure(call):
            for _ in range(args.warmup):
                call()
            torch.npu.synchronize()
            start.record()
            output = call()
            end.record()
            end.synchronize()
            us = start.elapsed_time(end) * 1000
            assert math.isfinite(us) and us > 0
            return us

        for qlen, kv, best in cases:
            key = f"q{qlen}_kv{kv}"
            plan = make_plan(qlen, kv)
            info = dict(key=key, plan=plan.describe(), uniform_split=best, accuracy=[], builds=[])
            report["cases"].append(info)
            path.with_name(f"{path.stem}_{key}_plan.json").write_text(
                json.dumps(
                    dict(programs=plan.programs, merges=plan.merges, summary=plan.describe()), indent=2
                )
            )
            fixtures = {}
            for seed in (7, 19):
                report.update(phase="reference", active=f"{key}/seed{seed}")
                save()
                q, k, v, table = make_inputs(
                    (key, 1, qlen, 1, 128, 128, kv // 128), seed, torch.bfloat16, torch
                )
                lengths = torch.tensor([kv], dtype=torch.int32)
                expected = chunked_cpu_reference(q, k, v, table, lengths, 128, False, 128)
                fixtures[seed] = ([x.to("npu") for x in (q, k, v, table, lengths)], expected)
            rows = []
            for rd in range(1 if args.smoke else 2):
                report.update(phase="compile_accuracy", active=f"{key}/round{rd+1}")
                save()
                compiler_temp(path)
                candidate.build_kernel.cache_clear()
                default.build_kernel.cache_clear()
                pool = fixtures[7][0][1].shape[0]
                static = candidate.build_kernel(qlen, kv, pool)
                base = default.build_kernel(
                    1, qlen, 1, 128, pool, kv // 128, 128, False, "bfloat16", args.device, full_kv_cache=True
                )
                default.build_kernel.cache_clear()
                compiler_temp(path)
                uniform = (
                    base
                    if base.num_splits == best
                    else default.build_kernel(
                        1,
                        qlen,
                        1,
                        128,
                        pool,
                        kv // 128,
                        128,
                        False,
                        "bfloat16",
                        args.device,
                        full_kv_cache=True,
                        num_splits=best,
                    )
                )
                info["default_split"] = base.num_splits
                kernels = {"static_local": static.local_kernel}
                if static.merge_kernel is not None:
                    kernels["static_merge"] = static.merge_kernel
                for label, program in [("default", base), ("uniform", uniform)]:
                    if program.num_splits == 1:
                        kernels[label + "_direct"] = program.direct_kernel
                    else:
                        kernels.update(
                            {
                                label + "_partial": program.partial_kernel,
                                label + "_merge": program.merge_kernel,
                            }
                        )
                hashes = {}
                for label, kernel in kernels.items():
                    source = kernel.get_kernel_source()
                    hashes[label] = hashlib.sha256(source.encode()).hexdigest()
                    path.with_name(f"{path.stem}_{key}_r{rd+1}_{label}.cpp").write_text(
                        source, encoding="utf8"
                    )
                info["builds"].append(dict(round=rd + 1, source_sha256=hashes))
                for seed, (inputs, expected) in fixtures.items():
                    report["active"] = f"{key}/round{rd+1}/seed{seed}"
                    save()
                    static.validate_inputs(*inputs)
                    counts, tasks, merges = static.prepare_plan(inputs[0].device)
                    prepared = inputs[0].permute(0, 2, 1, 3).contiguous()
                    local = functools.partial(
                        static.local_kernel, prepared, inputs[1], inputs[2], inputs[3], counts, tasks
                    )
                    parts, lse, output = local()
                    calls = {
                        "default": functools.partial(base, *inputs),
                        "static": functools.partial(static, *inputs),
                    }
                    if uniform is not base:
                        calls["uniform_best"] = functools.partial(uniform, *inputs)
                    if static.merge_kernel is not None:
                        calls["static/merge"] = functools.partial(static.finish, parts, lse, output)

                        def chain(local=local, static=static):
                            return static.finish(*local())

                        calls["static/chain"] = chain
                    native = functools.partial(
                        torch_npu.npu_fused_infer_attention_score_v2,
                        inputs[0],
                        inputs[1].view(-1, 128, 128),
                        inputs[2].view(-1, 128, 128),
                        block_table=inputs[3],
                        num_query_heads=1,
                        num_key_value_heads=1,
                        input_layout="BSND",
                        block_size=128,
                        softmax_scale=128**-0.5,
                        actual_seq_qlen=[qlen],
                        actual_seq_kvlen=[kv],
                        sparse_mode=0,
                        inner_precise=0,
                        return_softmax_lse=False,
                    )
                    calls["ascendc"] = lambda native=native: native()[0]
                    for label, call in calls.items():
                        got = call().cpu()
                        assert (
                            got.shape == expected.shape
                            and got.dtype == expected.dtype
                            and torch.isfinite(got).all()
                        )
                        torch.testing.assert_close(got, expected, rtol=0.01, atol=0.004)
                        info["accuracy"].append(
                            dict(
                                round=rd + 1,
                                seed=seed,
                                label=label,
                                max_abs_error=(got.float() - expected.float()).abs().max().item(),
                            )
                        )
                    if seed == 7:
                        calls["static/local"] = local
                        timed_calls = calls
                if args.smoke:
                    continue
                report.update(phase="timing", active=f"{key}/round{rd+1}")
                save()
                labels = tuple(timed_calls)
                samples = {label: [] for label in labels}
                for pair in range(args.pairs):
                    offset = (pair + rd) % len(labels)
                    order = labels[offset:] + labels[:offset]
                    if (pair + rd) % 2:
                        order = tuple(reversed(order))
                    for label in order:
                        samples[label].append(measure(timed_calls[label]))
                medians = {k: statistics.median(v) for k, v in samples.items()}
                rows.append(medians)
                report["rounds"].append(dict(key=key, round=rd + 1, median_us=medians, samples_us=samples))
                save()
                print(key, "round", rd + 1, medians, flush=True)
            info["accuracy_checks"] = len(info["accuracy"])
            info["max_abs_error"] = max(a["max_abs_error"] for a in info["accuracy"])
            if args.smoke:
                report["summary"].append(dict(key=key, status="accuracy_passed", plan=plan.describe()))
            else:
                result = dict(
                    key=key,
                    median_us={k: statistics.median(r[k] for r in rows) for k in rows[0]},
                    round_us=rows,
                    static_builds=len(info["builds"]),
                    plan=plan.describe(),
                )
                for control in ("default", "uniform_best", "ascendc"):
                    if control not in rows[0]:
                        continue
                    ratios = [r[control] / r["static"] for r in rows]
                    result["static_vs_" + control] = dict(
                        speedup=statistics.median(ratios),
                        min_round=min(ratios),
                        max_round=max(ratios),
                        faster_rounds=sum(x > 1 for x in ratios),
                    )
                report["summary"].append(result)
            save()
        report.update(status="passed", phase="complete", active=None, correctness="passed")
    except Exception:
        report.update(status="error", error=traceback.format_exc())
        traceback.print_exc()
    finally:
        save()
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
