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

## Cost-aware candidate

The grid runner now passes `--plan-policy cost` (the benchmark default remains
`balanced` for reproduction). Both policies use the same local/merge kernel
source. The host evaluates a whole-Q plan with up to 28 programs and balanced
KV-block plans with each program count from 1 through 28. The whole-Q plan
assigns consecutive complete Q tiles and writes final O directly without merge.
No device queue, benchmark-time tuning, shape lookup or default-v7 change.

The initial fixed cost score is the maximum program's
`0.72 * KV_blocks + 2.0 * segments`, plus, for split plans only,
`12.0 + 0.20 * partial_count + 0.15 * merge_Q_tiles`.
Units are approximate microseconds. These are initial stage estimates motivated
by prior local/merge measurements, not regression-fitted or validated latency
predictions. Common launch/allocation costs cancel in this simplified score;
occupancy, cache effects and compiler changes are not modeled. All 80 shapes use
identical coefficients, recorded in the raw report. NPU validation is required.

Raw cases include the selected plan, its estimated cost and the previous
28-program balanced plan. Compact results append `strategy` and `programs`.
Timing remains one compile round, three rotated timing pairs, five immediate
warmups, two accuracy seeds. Existing five-case smoke runs the selected policy
before the grid. CPU checks cover all 80 shapes plus tails, no overlap/gaps,
partial ownership and the independent attention/state-merge oracle.

## Direct-output ABI cleanup

No-fragment plans now use a five-buffer entry (Q/K/V/page table/final O).
The host launches it without allocating Parts/LSE or uploading Counts/Tasks/
Merges. A program computes its consecutive whole-Q-tile range from its index:
`first = pid * (tiles // programs) + min(pid, tiles % programs)`, with
`count = tiles // programs + (pid < tiles % programs)`.
The compiled split entry retains its existing nine-buffer ABI. Both entries
inline one shared Cube/Vector computation macro; the plan cost coefficients
and grid test policy are unchanged. The benchmark adapter represents absent
partial outputs as Python None values, not device tensors.

Local Ascend lowering checks cover Q1, Q129 and Q3712 with KV512 under the cost
policy, and Q129/KV512 under balanced policy. They verify five/nine kernel
arguments and absent direct metadata upload/merge. These checks use local
TileLang e84ce3de, not the server's a7cb753d; they do not establish hardware
accuracy or speed. The five-case NPU smoke and 80-shape grid remain required.
