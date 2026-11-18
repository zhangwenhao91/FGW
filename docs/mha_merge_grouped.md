# Group original R8 DMA chunks: clean task-assignment comparison

```bash
git pull --ff-only
python fgw.py mha run run_mha_merge_grouped
cat reports/mha/latest/mha_merge_grouped_paste.txt
```

Only Q32768/KV4096 and Q32768/KV8192, B1H1 BF16 D128 noncausal page128.
Compare grouped R8, fixed R8 and installed AscendC FIA in the same measurement
round. Local computation is one shared compiled kernel for grouped/fixed calls.
Each configuration compiles once: one local kernel and two merge kernels. The
three timings rotate through all three order positions across three triples;
each sample has five immediate warmups of that exact call. Two accuracy seeds
are checked before timing. Full-call timing includes ordinary output/workspace
allocations and all kernels, excludes compilation and one-time metadata upload.

The two merge entries inline exactly the same per-R8-block macro: identical UB
shapes, T.copy of Parts and LSE, softmax state reduction, BF16 conversion and
T.copy to final O. Grouped entry only changes ownership: sort R8 chunks by
valid_rows * pieces, greedily put them on the least-loaded static worker, then
loop over the worker's chunks. Padded task slots are guarded. Test budgets 28
and 56, four configurations total. For target plans there are 64 original R8
chunks: budget56 assigns up to two chunks per worker, budget28 up to three.
Grouping may worsen the critical worker's time; no speedup is assumed.

Both controls share identical local output contracts and host launch wrappers;
fixed metadata is prepared before timing. The fixed baseline's generated C++
for Q32768/KV4096 is byte-identical to the pre-change fixed merge in local
TileLang. Host tests cover all grid plans plus tails, unique chunk/row ownership
and independent dense-attention/state merge. Local lowering checks both worker
budgets. Official Ascend950PR_9579 CAModel checks cover Q3713 with untouched
direct rows and a one-row tail, and four groups of seven parts with multiple
chunks per worker. Local compiler differs from the server; hardware checks
remain necessary. Compact report includes all three raw samples to expose
large timing fluctuations. Production/default v7 and other policies unchanged.
