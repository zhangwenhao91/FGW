# Raw local kernel NPU profiling

Run `python fgw.py mha run run_mha_raw_local_profile` and return
`reports/mha/latest/mha_raw_local_profile_paste.txt`.

This profiles the current `stream` host plan and `rawsimd` candidate at
B1H1, BF16, D128, noncausal, page128, Q32768, KV4096/8192. It changes no
kernel implementation. Each shape compiles once and checks the full output
against the CPU reference with seeds 7 and 19 before capture.

Each shape has three independent profiler sessions. Each session runs 100
local warmups, synchronizes, then captures five local calls with Level1
PipeUtilization. Q preparation and task-table upload happen before capture;
merge and AscendC are excluded. Unlike TileLang's cache-flushing
`msprof_detail` benchmark, this collector does not deliberately flush L2.
Profiler instrumentation still changes execution: these durations do not
replace the paired warm-cache Event benchmark.

Each run has a private directory containing JSON reports, console/resource
logs, generated local/merge sources, installed TileLang profiler source and
hashes, retained raw profiling data, and the profiler's exported trace/CSV.
The trace JSON can be opened in Perfetto. The supervisor applies the existing
CPU, memory and timeout policy; it cannot guarantee protection against a
device or driver failure.

The compact summary reports AIC/AIV counters separately, including
cycle-weighted busy ratios and record ranges. Ratios can overlap and must
not be added. Counter records are not physical-core counts. Record durations
are not Event latency. A low busy ratio alone does not measure synchronization
wait time or distinguish dependency stalls from insufficient work.

The binary parser follows the inspected TileLang FFTS layout. It filters
`local_split` tasks and rejects missing AIC/AIV data, unsupported record sizes
or implausible counters. On parsing failure, inspect the retained raw files
and saved installed parser source before adapting the interpretation; do not
treat a failed parse as measured zero utilization. CPU synthetic fixtures
verify filtering, weighting and rejection behavior, not compatibility with
every CANN version. Real NPU capture is still required to verify compatibility.
