# FIA-inspired remaining-work host scheduler

```bash
git pull --ff-only
python fgw.py mha run run_mha_stream_grid
cat reports/mha/latest/mha_stream_grid_paste.txt
```

Original 80 shapes, B1H1 BF16 D128 page128 noncausal. Compare only this static
candidate and installed AscendC FIA. One compilation round, three genuinely
alternating timing pairs, five immediate warmups per sample, two correctness
seeds. Compilation and one-time plan metadata upload are excluded. No v7 or
diagnostic timing. Independent guards, incremental reports and progress remain.

Reference inspected: cann/ops-transformer commit
155c93cc90541443d210e0eb51ff681633670595, common/op_host/split_core_v2.cpp and
fused_infer_attention_score/op_host/arch35/fia_tiling_nonquant_gqa.cpp.
This is an independent simplified scheduler inspired by their assignment order,
not a port or a claim of reproducing the installed package's actual tiling.

For each candidate core count 1..28, recompute remaining KV-block work divided
by remaining cores. Prefer whole remaining Q tiles; fill remaining capacity
with KV blocks, allowing half a block of rounding tolerance. If a core would
receive nothing, assign at least one block. Record only cross-core Q tiles as
merge groups, with per-group split counts/workspace indices. Short Q can now
use split-KV even without any full wave of Q work.

Also evaluate our existing final-wave planner with adaptive (not fixed seven)
piece counts at each core count. This extra candidate family is our extension,
not a documented special tail-only rule in AscendC. Whole-Q direct output is
always an explicit candidate. The existing frozen COST_MODEL (including merge
launch/partial/merge-tile costs) scores all plans. Require split to save at
least one further block_us over direct output; otherwise choose direct. This
threshold is our initial conservative margin. It is not the source's
fdTolerance=9: source cost units/model differ from our kernel's stage estimates.
No shape table, online device tuning or coefficient fitting from this run.

Same Cube/Vector math and merge R8 are used for all selected plans. No change
to production v7 or old policy reproduction. Plan details, source hashes,
model coefficients, raw samples and per-case correctness remain in JSON.

CPU tests cover original 80 shapes plus tails across core counts, exact
coverage/no overlaps, unique partial slots, matching merge groups, and dense
attention vs independent split-state merge oracle. Local Ascend lowering
checks cover short-Q split, direct-Q tail and long-Q selective tail merge.
NPU accuracy/performance must still be validated by the command above. In
particular Q32768/KV4096 still selects the same tail plan as before, so this
scheduler change alone is not expected to guarantee a win for that target.
