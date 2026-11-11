"""Standalone NPU accuracy regression; writes a checkpoint after every result.

Run with Python directly, not pytest. --list-only needs only the standard library.
The unoptimized token-serial kernel can be slow on long caches.
"""
import argparse
import datetime
import itertools
import importlib
import json
import pathlib
import time
import traceback


# label, batch, queries, heads, head_dim, page_size, table_width
QUICK = [
    ("tiny", 2, 2, 2, 16, 16, 2),
    ("non_power_two_dim", 3, 4, 2, 80, 32, 3),
    ("wide_head", 2, 2, 4, 256, 128, 2),
]
STANDARD = QUICK + [
    ("single_decode", 1, 1, 8, 64, 128, 4),
    ("dim32", 4, 2, 4, 32, 32, 4),
    ("dim128", 2, 4, 4, 128, 64, 4),
    ("many_queries", 2, 8, 2, 64, 16, 8),
    ("many_requests", 8, 2, 2, 64, 64, 4),
    ("page256", 2, 2, 4, 128, 256, 4),
    ("cache2048", 2, 2, 2, 64, 128, 16),
]
EXTENDED = STANDARD + [
    ("dim48", 2, 2, 2, 48, 48, 3),
    ("dim96", 2, 4, 2, 96, 64, 8),
    ("dim192", 2, 2, 2, 192, 128, 4),
    ("cache4096", 2, 2, 2, 128, 256, 16),
    ("cache8192", 2, 2, 2, 64, 256, 32),
]


def lengths_for(case, pattern, seed):
    _, b, sq, _, _, p, w = case
    capacity = p * w
    if pattern == "empty":
        return [0] * b
    if pattern == "full":
        return [capacity] * b
    # Rotate through zero, shorter than Q, page boundaries, partial last page,
    # and full capacity. Multiple seeds shift which requests see which lengths.
    choices = [0, 1, max(0, sq - 1), sq, p - 1, p, p + 1, capacity - 1, capacity]
    return [min(capacity, choices[(i + seed) % len(choices)]) for i in range(b)]


def make_inputs(case, seed, dtype, torch):
    _, b, sq, h, d, p, w = case
    generator = torch.Generator().manual_seed(seed)
    # A pool larger than any request's capacity, with one page never referenced.
    pool_pages = max(w + 3, b * w + 1)
    table = torch.stack([torch.randperm(pool_pages - 1, generator=generator)[:w]
                         for _ in range(b)]).to(torch.int32)
    if b > 1:
        table[1:, 0] = table[0, 0]  # Explicit sharing between requests.
    if w > 1:
        # Repeated physical pages within a request exercise logical-vs-physical indexing.
        table[-1, -1] = table[-1, 0]
    q = torch.randn(b, sq, h, d, generator=generator).to(dtype)
    k = torch.randn(pool_pages * p, h, d, generator=generator).to(dtype)
    v = torch.randn(pool_pages * p, h, d, generator=generator).to(dtype)
    return q, k, v, table


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=["quick", "standard", "extended"], default="standard")
    parser.add_argument("--seeds", nargs="+", type=int, default=[7, 42, 2026])
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--impl", choices=["v0", "v1", "v2"], default="v0")
    parser.add_argument("--output", default=None)
    parser.add_argument("--list-only", action="store_true")
    parser.add_argument("--cases", nargs="+", choices=[c[0] for c in EXTENDED],
                        help="Run only these named cases from the selected suite")
    args = parser.parse_args()
    if args.output is None:
        args.output = "mha_paged_accuracy_report.json" if args.impl == "v0" else f"mha_paged_accuracy_{args.impl}_report.json"
    cases = {"quick": QUICK, "standard": STANDARD, "extended": EXTENDED}[args.suite]
    if args.cases:
        missing = set(args.cases) - {c[0] for c in cases}
        if missing:
            parser.error(f"Cases not in selected suite: {sorted(missing)}")
        cases = [c for c in cases if c[0] in args.cases]
    patterns = ("mixed", "full", "empty")
    total = len(cases) * len(args.seeds) * 2 * 2 * len(patterns)
    if args.list_only:
        print(json.dumps({"cases": cases, "seeds": args.seeds,
                          "patterns": patterns, "total_checks": total}, indent=2))
        return 0

    output = pathlib.Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    report = {"status": "running", "suite": args.suite, "seeds": args.seeds,
              "selected_cases": [c[0] for c in cases], "implementation": args.impl,
              "total_checks": total, "results": [], "logical_device": args.device,
              "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "note": "Timing includes launch/copies/checks and is not a kernel benchmark."}

    def save():
        temporary = output.with_suffix(output.suffix + ".tmp")
        temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        temporary.replace(output)

    save()
    try:
        import torch
        import torch_npu
        import tilelang
        from mha_paged_decode_ascend_v0 import reference
        mha_paged_decode = importlib.import_module(f"mha_paged_decode_ascend_{args.impl}").mha_paged_decode
        torch.npu.set_device(args.device)
        device = f"npu:{args.device}"
        report["versions"] = {"torch": torch.__version__, "torch_npu": torch_npu.__version__,
                              "tilelang": tilelang.__version__}
        report["device_name"] = torch.npu.get_device_name(args.device)
        for case, seed, dtype in itertools.product(cases, args.seeds, (torch.float16, torch.bfloat16)):
            q, k_base, v_base, table = make_inputs(case, seed, dtype, torch)
            _, b, sq, h, d, p, w = case
            # Reuse one compiled specialization across changing runtime lengths/data.
            for causal, pattern in itertools.product((False, True), patterns):
                lengths = torch.tensor(lengths_for(case, pattern, seed), dtype=torch.int32)
                entry = {"case": case[0], "shape": {"B": b, "Sq": sq, "H": h, "D": d,
                         "page_size": p, "table_width": w, "pool_rows": k_base.shape[0]},
                         "seed": seed, "dtype": str(dtype), "causal": causal,
                         "pattern": pattern, "lengths": lengths.tolist(), "status": "running"}
                report["results"].append(entry)
                save()  # Preserve active case even after a native runtime crash.
                start = time.perf_counter()
                try:
                    live = torch.zeros(k_base.shape[0], dtype=torch.bool)
                    for bi in range(b):
                        pos = torch.arange(int(lengths[bi]))
                        physical = table[bi, pos // p].long() * p + pos % p
                        live[physical] = True
                    k, v = k_base.clone(), v_base.clone()
                    k[~live], v[~live] = float("nan"), float("nan")
                    inputs = [x.to(device) for x in (q, k, v, table, lengths)]
                    got = mha_paged_decode(*inputs, page_size=p, is_causal=causal).cpu()
                    expected = reference(q, k, v, table, lengths, p, causal)
                    assert got.shape == q.shape and got.dtype == dtype
                    assert torch.isfinite(got).all(), "Non-finite output"
                    # Keep the original v0 tolerances; do not loosen them for long caches.
                    rtol, atol = (1e-2, 4e-3) if dtype == torch.bfloat16 else (2e-3, 2e-3)
                    error = (got.float() - expected.float()).abs()
                    bound = atol + rtol * expected.float().abs()
                    entry.update(max_abs_error=error.max().item(), mean_abs_error=error.mean().item(),
                                 max_tolerance_ratio=(error / bound).max().item(), rtol=rtol, atol=atol)
                    torch.testing.assert_close(got, expected, rtol=rtol, atol=atol)
                    visible = lengths[:, None].expand(b, sq)
                    if causal:
                        visible = (visible - sq + torch.arange(sq)[None, :] + 1).clamp_min(0)
                    assert torch.count_nonzero(got[visible == 0]).item() == 0, "Empty rows must be exactly zero"
                    entry["status"] = "passed"
                except AssertionError:
                    entry.update(status="failed", error=traceback.format_exc())
                except Exception:
                    # An NPU runtime failure may poison the context; stop rather than
                    # presenting a cascade of dependent failures as separate findings.
                    entry.update(status="error", error=traceback.format_exc())
                    raise
                finally:
                    entry["wall_seconds"] = time.perf_counter() - start
                    save()
                    print(f"[{len(report['results'])}/{total}] {case[0]} seed={seed} {dtype} "
                          f"causal={causal} {pattern}: {entry['status']} "
                          f"max_error={entry.get('max_abs_error', 'n/a')}", flush=True)
        failed = sum(r["status"] != "passed" for r in report["results"])
        report["status"] = "failed" if failed else "passed"
    except Exception:
        report.update(status="error", error=traceback.format_exc())
        traceback.print_exc()
    finally:
        report["passed"] = sum(r["status"] == "passed" for r in report["results"])
        report["failed_or_incomplete"] = sum(r["status"] != "passed" for r in report["results"])
        save()
    print(f"{report['status'].upper()}: {report['passed']}/{total} passed; report={output}", flush=True)
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
