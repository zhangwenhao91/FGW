# Test-related loss of server access: self-audit

2026-11-17. User reports two outages after launching split-breakdown resume and
merge-granularity runners; terminal appeared silent, other users reported a crash.
Whether this was a host reboot, container loss, SSH starvation or driver hang is
not established. No NPU test was run for this audit.

## What the code review establishes

- These two commands do not invoke the new static-host candidate kernel.
- Their Python loops and shell invocations are sequential and bounded. No
  script-created worker pool, fork bomb, reboot command, device-reset command,
  system tuning or recursive deletion of global `/tmp` was found in these paths.
- `compiler_temp()` creates a unique probe below each run's compiler_tmp and
  removes only that probe. It neither deletes nor repairs global `/tmp`.
- The previous ENOENT was creating `/tmp/tmp...`, not a numerical failure.
  It establishes path resolution failure at that instant; it does not establish
  who removed a path or whether a mount, container or host event was responsible.
- CPU references use Q chunks128, not a full Q-by-KV score matrix. At KV32768,
  one FP32 score chunk is16 MiB. Local CPU-only Q8192/KV32768 reference at four
  threads had process peak RSS392.8 MiB (220.0 MiB before fixture/reference).
  Local Torch2.14.1 differs from server2.13; these figures exclude CANN, compiler,
  NPU runtime, binary registration and device allocations.
- Host and NPU inputs/fixtures are retained across timing intentionally. R8/16/32
  merge fixtures are separate but bounded, and summaries retain scalar values,
  not all historical outputs. This does not prove native compiler/runtime memory
  is reclaimed when the Python builder LRU is cleared.
- Warmup and Event measurement synchronize before moving to the next compile
  round. No obvious use-after-free through an unfinished timed call was found.
- Merge row caps divide128; task/slot/output bounds for the tested aligned shapes
  have no obvious index overflow. Simulation and numerical tests cannot rule out
  compiler/runtime/driver faults or establish system stability.

## Confirmed shortcomings and containment changes

1. CPU reference threading was inherited/unbounded: cap OpenMP/MKL/OpenBLAS/
   NumExpr and Torch compute to4 threads (FGW_CPU_THREADS1..8).
2. Preparation/import/compilation had long silent periods: print/checkpoint the
   exact phase, including torch auto-backend import, explicit torch_npu import,
   tilelang import and device selection; retain host RSS/thread counts.
3. No external watchdog or resource record: the three recent runners now launch
   the worker in its own session through a host-only supervisor, log process-tree
   RSS/host MemAvailable/disk space every second and print a heartbeat every15s.
   Default tree RSS budget4 GiB and wall budget30 minutes, plus low-memory/disk
   stops. Lower worker nice priority to10 when permitted. Only that session is
   signaled; no global reset or unrelated process termination is performed.

RSS polling is not a hard cgroup limit. Shared pages can be counted repeatedly;
a rapid spike can outrun polling. It does not monitor NPU device memory and
the Torch/BLAS thread settings do not impose an OS CPU quota on the compiler or
CANN's own internal worker threads. The supervisor
cannot prevent kernel panic, uninterruptible driver I/O, hardware faults or
external resource exhaustion. The worker still has a multi-compile lifetime;
case/round subprocess isolation remains a potential next step after diagnosis.
These changes are containment/observability, NOT a proven root-cause fix.
Resource-policy changes are recorded and incompatible old resume data is refused
before importing Torch. Do not mix guarded and previous timing reports.

Host-only checks validated normal exit, process RSS-budget termination, wall-time
termination and preservation of an unrelated child process. No NPU test was used.

## Collect evidence before any more NPU tests

After the server is accessible, from FGW root:

```bash
git pull --ff-only
python tools/mha/collect_crash_context.py
cat reports/diagnostics/crash_context.txt
```

This collector does NOT import Torch, execute npu-smi, call a device API, compile
kernels, run tests, delete files or reset a device. It reads host RAM/disk/inode,
mount, boot-id/uptime, filtered kernel journal/dmesg, cgroup OOM counters and saved
test progress, writing only its report. Previous-boot logs may be unavailable or
container-limited; in that case administrator/BMC/host logs may be necessary.

Look for OOM/killed-process/cgroup counters; Ascend/devdrv/PCIe/watchdog/panic;
filesystem disappearance/fullness; changed boot id and reboot history. If SSH
was merely unresponsive, host load/RSS and the exact last saved stage are useful.
Do not attribute the outage to CPU reference, JIT accumulation or a device kernel
without the corresponding evidence, and do not launch a full stress sweep merely
to reproduce the outage.
