"""Host-only benchmark supervisor; never imports torch or queries the NPU."""

import argparse
import datetime
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time


def cpu_environment():
    threads = int(os.environ.get("FGW_CPU_THREADS", "4"))
    if not 1 <= threads <= 8:
        raise ValueError("FGW_CPU_THREADS must be 1..8")
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = str(threads)
    return threads


def runtime_policy():
    return dict(
        cpu_threads=int(os.environ.get("FGW_CPU_THREADS", "4")),
        supervised=os.environ.get("FGW_RESOURCE_GUARD") == "1",
        nice=os.getpriority(os.PRIO_PROCESS, 0) if hasattr(os, "getpriority") else None,
    )


def host_snapshot():
    snapshot = dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    if Path("/proc/meminfo").exists():
        mem = dict(
            (line.split(":")[0], int(line.split()[1]))
            for line in Path("/proc/meminfo").read_text().splitlines()
            if ":" in line and line.split()[1].isdigit()
        )
        snapshot.update(
            mem_available_mb=mem.get("MemAvailable", 0) / 1024, swap_free_mb=mem.get("SwapFree", 0) / 1024
        )
    return snapshot


def progress(path, phase):
    """One bounded host snapshot per phase; no device synchronization or query."""
    snapshot = host_snapshot()
    snapshot.update(phase=phase, pid=os.getpid())
    status = Path("/proc/self/status")
    if status.exists():
        for line in status.read_text().splitlines():
            if line.startswith(("VmRSS:", "VmHWM:", "Threads:")):
                key, value = line.split(":", 1)
                snapshot[key] = value.strip()
    print("[FGW stage] " + phase, flush=True)
    with path.with_name(path.stem + "_resources.jsonl").open("a", encoding="utf8") as f:
        f.write(json.dumps(snapshot) + "\n")
        f.flush()
        os.fsync(f.fileno())


def process_tree(root_pid):
    entries = {}
    pending = [root_pid]
    selected = set()
    while pending:
        pid = pending.pop()
        if pid in selected:
            continue
        selected.add(pid)
        directory = Path('/proc')/str(pid)
        try:
            values = {}
            for line in (directory / "status").read_text().splitlines():
                if ":" in line:
                    key, value = line.split(":", 1)
                    values[key] = value.strip()
            entries[pid] = (
                int(values["PPid"]),
                int(values.get("VmRSS", "0 kB").split()[0]),
                values.get("Name"),
            )
            # Follow only this worker's descendants, including children spawned
            # by non-main threads; do not scan every process on a shared server.
            for thread in (directory/'task').iterdir():
                try:pending.extend(map(int,(thread/'children').read_text().split()))
                except (OSError,ValueError):continue
        except (OSError, ValueError, KeyError):
            continue
    return dict(
        pids=sorted(selected),
        rss_mb=sum(entries.get(pid, (0, 0, ""))[1] for pid in selected) / 1024,
        names={str(pid): entries[pid][2] for pid in selected if pid in entries},
    )


def stop_group(process):
    if process.poll() is not None:
        return
    # Only the dedicated session created by this supervisor is targeted.
    try:
        if os.getpgid(process.pid) != process.pid:
            raise RuntimeError("Refusing to signal a process group not owned by this worker")
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            return
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("missing worker command")
    if os.name != "posix" or not Path("/proc").is_dir():
        parser.error("Linux /proc required")
    threads = cpu_environment()
    max_rss = int(os.environ.get("FGW_MAX_TREE_RSS_MB", "4096"))
    timeout = int(os.environ.get("FGW_TEST_TIMEOUT_SECONDS", "1800"))
    if max_rss <= 0 or timeout <= 0:
        parser.error("positive memory/time budgets required")
    print("[FGW guard] starting host-only preflight", flush=True)
    args.log.parent.mkdir(parents=True, exist_ok=True)

    def log(record):
        with args.log.open("a", encoding="utf8") as f:
            f.write(json.dumps(record) + "\n")
            f.flush()
            os.fsync(f.fileno())

    start = time.monotonic()
    log(
        dict(
            event="start",
            command=command,
            cpu_threads=threads,
            max_tree_rss_mb=max_rss,
            timeout_seconds=timeout,
            **host_snapshot(),
        )
    )
    print(
        f"[FGW guard] Torch/BLAS thread cap={threads}; process-tree RSS budget={max_rss} MiB; timeout={timeout}s",
        flush=True,
    )
    if (
        host_snapshot().get("mem_available_mb", 2048) < 1024
        or shutil.disk_usage(args.log.parent).free < 1024**3
    ):
        log(dict(event="refused", reason="insufficient host memory/disk before launch", **host_snapshot()))
        print("[FGW guard] refusing worker: less than 1 GiB available host RAM or disk", flush=True)
        return 125
    os.environ["FGW_RESOURCE_GUARD"] = "1"
    process = subprocess.Popen(command, start_new_session=True)
    try:
        os.setpriority(os.PRIO_PROCESS, process.pid, 10)
    except (OSError, AttributeError):
        pass
    reason = None
    last_print = 0
    try:
        while process.poll() is None:
            tree = process_tree(process.pid)
            snapshot = host_snapshot()
            snapshot.update(event="sample", elapsed_seconds=round(time.monotonic() - start, 1), **tree)
            snapshot["disk_free_mb"] = shutil.disk_usage(args.log.parent).free / 1024**2
            log(snapshot)
            if tree["rss_mb"] > max_rss:
                reason = "process-tree RSS budget exceeded"
            elif snapshot.get("mem_available_mb", 1024) < 512:
                reason = "host MemAvailable below 512 MiB"
            elif snapshot["disk_free_mb"] < 1024:
                reason = "report/scratch filesystem free space below 1 GiB"
            elif time.monotonic() - start > timeout:
                reason = "worker wall timeout exceeded"
            if reason:
                print("[FGW guard] stopping worker: " + reason, flush=True)
                stop_group(process)
                break
            if time.monotonic() - last_print >= 15:
                print(
                    f"[FGW guard] elapsed={snapshot['elapsed_seconds']}s tree_RSS={tree['rss_mb']:.0f}MiB host_available={snapshot.get('mem_available_mb',0):.0f}MiB",
                    flush=True,
                )
                last_print = time.monotonic()
            time.sleep(1)
    finally:
        if process.poll() is None:
            stop_group(process)
        log(dict(event="end", returncode=process.poll(), reason=reason, **host_snapshot()))
    return (
        124
        if reason
        else (process.returncode if process.returncode is not None and process.returncode >= 0 else 1)
    )


if __name__ == "__main__":
    raise SystemExit(main())
