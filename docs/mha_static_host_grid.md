# Static host 80-shape validation

Run on the NPU host from the repository root:

```bash
git pull --ff-only
python fgw.py mha run run_mha_static_host_grid
cat reports/mha/latest/mha_static_host_grid_paste.txt
```

Q: 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768.
KV: 512, 1024, 2048, 4096, 8192, 16384, 24576, 32768.
All 80 Cartesian pairs match the previous clean-v7 grid. B1H1, BF16,
D128, page128, noncausal, full live KV. Default v7 remains unchanged.

Host unit checks and the existing five-case accuracy smoke precede the grid.
Each shape runs in its own supervised process with the existing thread/RSS/time
budgets. Failure stops subsequent shapes; completed per-shape reports remain.
No old reports are reused. Each shape has one compiled round,
two correctness seeds (7/19), three rotated timing pairs and five
immediate warmups for every sample, including native AscendC FIA. Compilation
is excluded from timing; timed calls reuse their compiled kernels. Static host
plan creation and one-time metadata upload are also excluded. Calls include
normal Q preparation, output/workspace allocations and all required kernels.

The three full-call controls are default v7, static host, and AscendC FIA.
No previously-best uniform split is assumed for unscanned shapes. Local/merge/
chain timing is also recorded when applicable. Per-shape JSON, generated C++,
plans, guard logs and timing samples live in the printed unique run directory.
The compact paste file updates after every completed shape and on exit;
missing/failed shapes are never reported as passed. Ratios are medians of the
per-round speedups, with the AscendC ratio range retained (one round gives an identical min/max).
Terminal output shows shape index out of 80, compilation/accuracy stages, and each timing sample.

Local checks validate exact grid coverage, argument selection, summary handling
and shell syntax. NPU correctness/performance require running the command above.
