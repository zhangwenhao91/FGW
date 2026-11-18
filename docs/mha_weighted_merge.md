# Row-times-pieces weighted Vector merge candidate

```bash
git pull --ff-only
python fgw.py mha run run_mha_weighted_grid
cat reports/mha/latest/mha_weighted_grid_paste.txt
```

The same 80 shapes and stream-policy local task assignment are used. Only
merge assignment/implementation changes. Compare weighted candidate vs installed
AscendC FIA, one compilation round, three alternating timing pairs, five
immediate warmups per control, two correctness seeds. Compilation and one-time
host task-table upload are excluded, ordinary output/workspace allocations and
all kernels included. No default v7 or fixed-merge policy change.

Inspired by SplitFD in cann/ops-transformer split_core_v2.cpp at commit
155c93cc90541443d210e0eb51ff681633670595. Work for merge group i is valid_Q_rows_i
multiplied by KV_piece_count_i. Use a 56-vector program budget for this fixed
28-Cube platform, reserve one program per group, compute average work, and give
heavier groups more programs by subdividing their row ranges. Each program
owns (Q tile, first partial slot, pieces, row start, valid row count). All
parts of a particular output row are reduced by its one owner, so no atomics
or second merge is required. No static eight-row grouping or per-shape lookup.

For Q32768/KV4096, four groups each contain 128 rows and seven pieces. The
planner produces 52 programs with 8..10 valid rows each (rather than 64 R8
programs). UB capacity rounds the maximum row count up to a power of two, but
all GM reads/writes are guarded by each task's actual valid rows. The new
SIMT merge reads GM directly and writes final output in place, avoiding
fixed-width T.copy overreads at group/Q tails. This also changes the GM access
implementation, so hardware performance cannot be attributed solely to task
count. More balanced work is not a guarantee of lower latency.

Host tests validate exact valid-row coverage, no duplicate writers, correct
partial references and <=56 programs for all grid plans and Q tails. The
independent attention oracle exercises both fixed and weighted row ownership.
Local Ascend lowering covers split short Q, direct tails and selective long-Q
merge. Official Ascend950PR_9579 CAModel checks use synthetic Partial/LSE states
with unequal group sizes/split counts, extreme LSE differences, Q tails and
untouched direct-output rows, plus the four-group/seven-piece pattern. Local
compiler version differs from the NPU server; full NPU smoke/grid remain
necessary. Compact reports include actual merge program count and maximum
valid rows; raw JSON records the complete merge task table.
