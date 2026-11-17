"""Read host crash evidence without importing Torch or calling the NPU driver.

Only report files are written. No cleanup, device reset or benchmark execution.
"""

import argparse
import datetime
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[2]
PATTERN = re.compile(
    r"out of memory|oom|killed process|panic|watchdog|thermal|devdrv|ascend|npu|aer:|pcie|segfault|fatal|no space",
    re.I,
)


def command(argv):
    try:
        result = subprocess.run(
            argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace", timeout=8
        )
        return result.stdout
    except (OSError, subprocess.TimeoutExpired) as e:
        return str(e)


def collect():
    report = dict(
        utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        platform=platform.platform(),
        scope="Host reads only, no Torch import, no NPU API/npu-smi, no cleanup/reset",
    )
    for name in ("meminfo", "uptime", "sys/kernel/random/boot_id", "loadavg", "1/comm", "1/cgroup"):
        try:
            report[name] = Path("/proc", name).read_text()
        except OSError as e:
            report[name] = str(e)
    report["filesystems"] = {}
    for directory in ("/", "/tmp", str(ROOT)):
        try:
            p = Path(directory)
            st = p.stat()
            usage = shutil.disk_usage(p)
            report["filesystems"][directory] = dict(
                realpath=str(p.resolve()),
                device=st.st_dev,
                is_symlink=p.is_symlink(),
                total_bytes=usage.total,
                free_bytes=usage.free,
            )
        except OSError as e:
            report["filesystems"][directory] = dict(error=str(e))
    report["mounts"] = command(["findmnt", "-T", str(ROOT), "-o", "TARGET,SOURCE,FSTYPE,OPTIONS"])
    report["inodes"] = command(["df", "-i", "/", "/tmp", str(ROOT)])
    report["boot_history"] = command(["last", "-x", "-n", "12"])
    report["top_rss_processes"] = command(["ps", "-eo", "pid,ppid,stat,comm,rss,nlwp", "--sort=-rss"]).splitlines()[:24]
    for name, argv in (
        ("previous_boot_kernel", ["journalctl", "-k", "-b", "-1", "--no-pager", "-n", "2000"]),
        ("current_boot_kernel", ["journalctl", "-k", "-b", "0", "--no-pager", "-n", "2000"]),
        ("dmesg", ["dmesg", "-T"]),
    ):
        text = command(argv)
        selected = [l for l in text.splitlines() if PATTERN.search(l)]
        report[name] = selected[-100:] if selected else text[-4000:]
    report["cgroup_v2"] = {}
    try:
        for line in Path("/proc/self/cgroup").read_text().splitlines():
            if line.startswith("0::"):
                base = Path("/sys/fs/cgroup") / line[3:].lstrip("/")
                for name in (
                    "memory.current",
                    "memory.max",
                    "memory.peak",
                    "memory.events",
                    "pids.current",
                    "pids.max",
                    "cpu.max",
                ):
                    try:
                        report["cgroup_v2"][name] = (base / name).read_text()
                    except OSError as e:
                        report["cgroup_v2"][name] = str(e)
    except OSError:
        pass
    report["latest_progress"] = {}
    latest = ROOT / "reports/mha/latest"
    if latest.exists():
        folders = sorted(
            (
                p
                for p in latest.iterdir()
                if p.is_dir()
                and p.name.startswith(("mha_split_breakdown.", "mha_merge_granularity.", "mha_static_host."))
            ),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )[:6]
        for folder in folders:
            details = {}
            for name in ("full.json", "smoke.json"):
                p = folder / name
                if not p.exists():
                    continue
                try:
                    d = json.loads(p.read_text(encoding="utf8"))
                    details[name] = {
                        k: d.get(k)
                        for k in ("status", "phase", "active", "compiler_tmp", "device", "versions")
                    }
                    details[name]["completed_cases"] = len(d.get("summary", []))
                    details[name]["error_tail"] = d.get("error", "")[-5000:]
                except (OSError, ValueError) as e:
                    details[name] = str(e)
            for p in folder.glob("*resources.jsonl"):
                try:
                    details[p.name] = p.read_text(encoding="utf8").splitlines()[-8:]
                except OSError as e:
                    details[p.name] = str(e)
            for p in folder.glob("guard*.jsonl"):
                try:
                    details[p.name] = p.read_text(encoding="utf8").splitlines()[-8:]
                except OSError as e:
                    details[p.name] = str(e)
            report["latest_progress"][folder.name] = details
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "reports/diagnostics/crash_context.json")
    args = parser.parse_args()
    print("Collecting host crash evidence only; Torch and NPU are not loaded.", flush=True)
    data = collect()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf8")
    text = args.output.with_suffix(".txt")
    text.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf8")
    print(text, flush=True)
