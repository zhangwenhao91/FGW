# Two-buffer prefetch for the original online R8 merge

```bash
git pull --ff-only
python fgw.py mha run run_mha_merge_prefetch
cat reports/mha/latest/mha_merge_prefetch_paste.txt
```

Only Q32768/KV4096 and Q32768/KV8192, B1H1 BF16 D128 noncausal page128.
Three same-round controls: prefetch candidate, original fixed R8, installed
AscendC FIA. One shared compiled local kernel, same plan/Partial/LSE layout,
R8 grid, online softmax state update and final division/BF16 conversion.
This experiment does not use precomputed weights, SIMD rewriting, grouped
merge tasks, or kernel fusion.

Partial and LSE input UB buffers each have two versions. A two-stage
T.Pipelined loop assigns GM-to-UB T.copy to stage0 and the original merge VF
to stage1. The intended overlap is next-piece input copy with current-piece
vector work. Generated local C++ confirms alternating parity addresses and
two pairs of MTE2/V events, including prologue/drain synchronization; source
annotations alone are not claimed as proof of hardware performance overlap.
Loop-carried accumulator/max/sum buffers remain single-version and ordered.

Local compiler emits non-affine versioning warnings because actual split
count guards the pipeline; generated indexing/events were inspected and
conditional behavior tested. These differ from the unsafe thread-sync-hoisting
warning seen and fixed in the earlier preweighted prototype. Three official
Ascend950PR_9579 CAModel fixtures pass: Q3713 with direct rows and a one-row
tail, four groups of seven parts (buffer reuse), and Q385 with two/seven-piece
groups in the same pipeline plus untouched direct rows. Extreme finite LSEs
are included. Local compiler differs from server a7cb753d, so NPU correctness
checks still precede timing.

AST checks confirm initialization, online-update VF and final-conversion VF
are identical to fixed R8. Fixed baseline generated C++ is unchanged. One
build round per shape compiles one local and two merge kernels. Three rotated
triples, five immediate warmups per control, two correctness seeds. Full-call
Event timing includes ordinary allocations and both kernels, excludes
compilation and one-time metadata upload. Progress and raw timing samples are
printed/preserved. No production/default change or speedup claim before NPU
measurement.
