"""Paired causal/noncausal repeated timing, using identical single-query inputs.

Component timings are isolated diagnostics, not additive pipeline attribution.
"""
import argparse
import datetime
import json
import pathlib
import statistics
import time
import traceback


def summary(values):
    return {"median": statistics.median(values), "min": min(values), "max": max(values)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rounds", type=int, default=6)
    parser.add_argument("--repeat", type=int, default=51)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--output", default="mha_v4_profile_report.json")
    args = parser.parse_args()
    if args.rounds < 2 or args.repeat < 3 or args.warmup < 1:
        parser.error("rounds>=2, repeat>=3, warmup>=1 required")
    report = {"status": "running", "settings": vars(args), "results": [],
              "time_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "shape": {"B": 1, "Sq": 1, "H": 16, "D": 128, "KV": 512, "page_size": 128},
              "notes": ["Warm-cache NPU Event samples; compiles/H2D/reference excluded.",
                        "Each event interval includes stream-visible launch gaps and allocation effects.",
                        "Isolated component medians do not sum to full-call median.",
                        "Q conversion may be a view for Sq=1; event readings then measure empty-interval overhead.",
                        "No empty-event overhead is subtracted."]}
    path = pathlib.Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)

    def save():
        path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    save()
    try:
        import torch
        import torch_npu
        import tilelang
        from operators.mha.kernels.mha_paged_decode_ascend_v4 import build_kernel
        from operators.mha.kernels.mha_paged_decode_ascend_v0 import reference
        torch.npu.set_device(args.device)
        device = f"npu:{args.device}"
        generator = torch.Generator().manual_seed(7)
        q, k, v = [torch.randn(*shape, generator=generator).bfloat16()
                   for shape in ((1, 1, 16, 128), (512, 16, 128), (512, 16, 128))]
        table = torch.tensor([[2, 0, 3, 1]], dtype=torch.int32)
        lengths = torch.tensor([512], dtype=torch.int32)
        inputs = [x.to(device) for x in (q, k, v, table, lengths)]
        qn, kn, vn, tn, ln = inputs
        report["device"] = str(torch.npu.get_device_properties(args.device))
        report["versions"] = {"torch": torch.__version__, "torch_npu": torch_npu.__version__, "tilelang": tilelang.__version__}
        programs = {causal: build_kernel(1, 1, 16, 128, 512, 4, 128, causal,
                                        "bfloat16", args.device) for causal in (False, True)}
        expected = reference(q, k, v, table, lengths, 128, False)
        outputs = {}
        functions = {}
        for causal, program in programs.items():
            outputs[causal] = program(*inputs).cpu()
            torch.testing.assert_close(outputs[causal], expected, rtol=1e-2, atol=4e-3)
            prepared = program.prepare_q(qn)
            parts, lse = program.partial_kernel(prepared, kn, vn, tn, ln)
            torch.npu.synchronize()
            functions[causal] = {
                "full": lambda p=program: p(*inputs),
                "q_layout": lambda p=program: p.prepare_q(qn),
                "partial": lambda p=program, qp=prepared: p.partial_kernel(qp, kn, vn, tn, ln),
                "merge": lambda p=program, a=parts, b=lse: p.merge_kernel(a, b),
            }
            report.setdefault("q_layout", {})[str(causal)] = {
                "shares_storage": prepared.untyped_storage().data_ptr() == qn.untyped_storage().data_ptr(),
                "input_stride": list(qn.stride()), "prepared_stride": list(prepared.stride()),
                "prepared_contiguous": prepared.is_contiguous()}
        torch.testing.assert_close(outputs[False], outputs[True], rtol=1e-2, atol=4e-3)
        report["causal_noncausal_max_abs_difference"] = (outputs[False].float() - outputs[True].float()).abs().max().item()
        report["correctness"] = "passed"

        def measure(fn):
            for _ in range(args.warmup):
                fn()
            torch.npu.synchronize()
            start, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
            values = []
            for _ in range(args.repeat):
                start.record()
                result = fn()  # Keep outputs alive through completion.
                end.record()
                end.synchronize()
                values.append(start.elapsed_time(end) * 1000)
            return values

        report["empty_event_us"] = summary(measure(lambda: None))
        stages = ["full", "partial", "merge", "q_layout"]
        for round_id in range(args.rounds):
            order = (False, True) if round_id % 2 == 0 else (True, False)
            # Rotate component order too, to avoid always timing one stage first.
            ordered_stages = stages[round_id % 4:] + stages[:round_id % 4]
            for stage in ordered_stages:
                for causal in order:
                    samples = measure(functions[causal][stage])
                    row = {"round": round_id + 1, "causal": causal, "stage": stage,
                           "unit": "us", "samples": samples, **summary(samples)}
                    report["results"].append(row)
                    save()
                    print(f"round={round_id+1} causal={causal} {stage}: median={row['median']:.3f} us", flush=True)
            # Host cost of Q view/contiguous without device synchronization;
            # relevant when no GPU operation exists to time with events.
            for causal in order:
                fn = functions[causal]["q_layout"]
                start_ns = time.perf_counter_ns()
                for _ in range(1000):
                    result = fn()
                elapsed = (time.perf_counter_ns() - start_ns) / 1000 / 1000
                report.setdefault("q_layout_host_us_per_call", []).append({"round": round_id+1, "causal": causal, "us": elapsed})
        report["summary"] = [{"causal": causal, "stage": stage, "unit": "us",
                              "round_medians": [r["median"] for r in report["results"] if r["causal"] == causal and r["stage"] == stage],
                              **summary([r["median"] for r in report["results"] if r["causal"] == causal and r["stage"] == stage])}
                             for stage in stages for causal in (False, True)]
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
