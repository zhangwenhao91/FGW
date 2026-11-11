#!/usr/bin/env python3
"""FGW operator development entry point; no NPU imports until a command runs."""
import argparse
import os
from pathlib import Path
import runpy
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
COMMANDS = {
    "env": {"collect": "tools.env.collect_env", "smoke": "tools.env.verify_tilelang_ascend"},
    "mha": {
        "accuracy": "tests.mha.test_mha_paged_accuracy",
        "bench": "benchmarks.mha.bench_mha_mfu",
        "msprof": "benchmarks.mha.profile_mha_msprof",
        "paired": "benchmarks.mha.profile_q_resident_paired",
        "event-profile": "benchmarks.mha.profile_mha_v4",
    },
}


def main():
    parser = argparse.ArgumentParser(description=__doc__, epilog="Examples: python fgw.py env collect; python fgw.py mha bench --impl v7 --suite large-q")
    parser.add_argument("area", choices=COMMANDS)
    parser.add_argument("command", help="env: collect/smoke; mha: accuracy/bench/msprof/paired/event-profile/run")
    args = parser.parse_args(sys.argv[1:3])
    forwarded = sys.argv[3:]
    if args.area == "mha" and args.command == "run":
        if len(forwarded) != 1:
            parser.error("Use: python fgw.py mha run <script-name-without-.sh>")
        name = forwarded[0]
        if Path(name).name != name or not name.startswith("run_mha_"):
            parser.error("Expected a run_mha_* script name")
        script = ROOT / "scripts" / "mha" / (name + ".sh")
        if not script.is_file():
            parser.error(f"Unknown script: {name}")
        return subprocess.call(["bash", str(script)], cwd=ROOT)
    module = COMMANDS[args.area].get(args.command)
    if module is None:
        parser.error(f"Unknown command for {args.area}: {args.command}")
    # v7 is current; historical versions are opt-in experiments.
    if args.area == "mha" and args.command in ("accuracy", "bench", "msprof"):
        if not any(a == "--impl" or a.startswith("--impl=") for a in forwarded):
            forwarded = ["--impl", "v7", *forwarded]
    out = ROOT / "reports" / ("environment" if args.area == "env" else args.area) / "latest"
    out.mkdir(parents=True, exist_ok=True)
    os.chdir(out)
    sys.argv = [module, *forwarded]
    runpy.run_module(module, run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
