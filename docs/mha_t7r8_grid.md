# t7r8 vs installed AscendC FIA: original 80-shape grid

Run after pulling main:

```bash
python fgw.py mha run run_mha_t7r8_grid
cat reports/mha/latest/mha_t7r8_grid_paste.txt
```

Uses the same 10 Q lengths and 8 KV lengths as the previous clean-v7 grid.
Only t7r8 and installed AscendC FIA are compiled/invoked as comparison controls;
v7 and uniform-split builds and local/merge/chain diagnostic timing are omitted.
The existing tail policy is unchanged: split final-wave Q tiles into seven KV
pieces (capped at the KV block count), merge row group eight. As in the prior
experiment, when there is no full Q wave, no remainder, or fewer than two KV
blocks, this policy uses whole-Q direct output. This is the t7r8 policy's
existing boundary behavior, not a fallback selected from measured timings.

B1H1 BF16 D128 page128 noncausal, two correctness seeds, one independent
compilation round, three timing pairs. Each sample is immediately preceded by
five warmups for that same control. Pair order alternates TL/AC, AC/TL, TL/AC.
Compilation and one-time plan/metadata upload are excluded; ordinary per-call
output/workspace allocations and all required kernels are included. No result
reuse or per-shape best-of-candidate selection. Host unit checks and five-case
accuracy smoke precede the grid. A failed case stops subsequent work.

Terminal progress, independent guarded processes, unique run folders, per-case
JSON/source exports and incremental compact summaries are retained. All 80
cases must complete for the summary to say passed. Near-1x measurements with
only three samples remain provisional.
