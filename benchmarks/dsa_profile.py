"""Capture and summarize NPU traces for the DSA diagnostic."""

import csv
import json
import statistics
import tempfile
from pathlib import Path


def summarize(directory):
    """Keep per-kernel medians, leaving full trace and CSV files on disk."""
    groups = {}
    raw = sorted(directory.rglob("op_summary_*.csv"))
    for path in raw or sorted(directory.rglob("op_summary.csv")):
        with path.open(encoding="utf-8-sig", newline="") as stream:
            for row in csv.DictReader(stream):
                name = row.get("OP Name") or row.get("Op Name") or row.get("OP Type")
                if not name:
                    continue
                group = groups.setdefault(name, {"rows": 0, "columns": {}})
                group["rows"] += 1
                for field, value in row.items():
                    if not field or not any(
                        token in field.lower()
                        for token in (
                            "time",
                            "duration",
                            "ratio",
                            "mte",
                            "cube",
                            "vec",
                            "block",
                        )
                    ):
                        continue
                    try:
                        number = float(value)
                    except (TypeError, ValueError):
                        continue
                    group["columns"].setdefault(field, []).append(number)
    return {
        name: {
            "rows": group["rows"],
            "median": {
                field: statistics.median(values)
                for field, values in group["columns"].items()
            },
        }
        for name, group in groups.items()
    }


def capture(calls, root, *, case, shape, dtype):
    import torch
    import torch_npu.profiler as profiler

    parent = Path(root)
    parent.mkdir(parents=True, exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix=f"{case}-", dir=parent))
    report = {
        "case": case,
        "shape": dict(
            zip(
                ("batch", "q", "kv", "heads", "kv_heads", "topk", "d", "tail"),
                shape,
                strict=True,
            )
        ),
        "dtype": dtype,
        "profiles": {},
    }
    for name, call in calls.items():
        print(f"Profiling {name} in {directory}", flush=True)
        for _ in range(5):
            call()
        torch.npu.synchronize()
        target = directory / name
        with profiler.profile(
            activities=[profiler.ProfilerActivity.CPU, profiler.ProfilerActivity.NPU],
            schedule=profiler.schedule(
                wait=0, warmup=0, active=1, repeat=1, skip_first=0
            ),
            on_trace_ready=profiler.tensorboard_trace_handler(str(target)),
            experimental_config=profiler._ExperimentalConfig(
                profiler_level=profiler.ProfilerLevel.Level1,
                aic_metrics=profiler.AiCMetrics.PipeUtilization,
                l2_cache=False,
                data_simplification=False,
            ),
        ) as prof:
            with torch.profiler.record_function(f"FGW_DSA_{name}"):
                for _ in range(5):
                    call()
                torch.npu.synchronize()
            prof.step()
        report["profiles"][name] = {
            "files": [
                str(p.relative_to(directory))
                for p in sorted(target.rglob("*"))
                if p.is_file()
                and p.name in ("trace_view.json", "op_summary.csv", "op_statistic.csv")
            ],
            "kernels": summarize(target),
        }
    report["status"] = (
        "passed"
        if all(item["kernels"] for item in report["profiles"].values())
        else "missing_op_summary"
    )
    summary = directory / "summary.json"
    summary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(summary, flush=True)
    print(summary.read_text(encoding="utf-8"))
