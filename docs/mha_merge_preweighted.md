# Precompute all split weights before R8 Partial reduction

```bash
git pull --ff-only
python fgw.py mha run run_mha_merge_preweighted
cat reports/mha/latest/mha_merge_preweighted_paste.txt
```

Only Q32768/KV4096 and Q32768/KV8192, B1H1 BF16 D128 noncausal page128.
Same-round controls: preweighted merge, original online fixed R8 merge, native
AscendC FIA. Same one compiled local kernel, task plan and Partial/LSE layout.
Same fixed R8 merge grid and T.copy access shapes for LSE, FP32 Partial and
BF16 output. No SIMD rewrite, double buffering, task regrouping or fusion in
this experiment. Default and prior reproduction policies remain unchanged.

For each R8 group, copy all valid split LSEs into a small aligned UB matrix;
find the row maximum, calculate exp(LSE-max), sum and normalize these weights
once. Then copy each Partial as before and accumulate Partial * weight.
The output accumulator no longer rescales after every new piece, and the
final vector no longer needs elementwise division. Invalid/sentinel LSEs
produce zero weight; all-empty rows produce zero output. Per-thread local
scalars handle row max/sum; separate VF stages establish producer/consumer
boundaries. Final local lowering has no thread-sync-hoisting warnings.

The math differs in floating-point reduction order. Existing independent
attention/state-merge oracle checks and official Ascend950PR_9579 CAModel
fixtures cover unequal split groups, extreme LSEs, Q tails, untouched direct
rows, and four groups of seven parts. Local compiler differs from the server;
NPU accuracy checks precede timing. Baseline fixed R8 generated C++ is unchanged.

One build round per shape: one local kernel and two merge kernels. Three
rotated timing triples place each control in each ordering position once;
five immediate warmups per sample, two accuracy seeds. Full-call timing
includes allocations and all kernels, excludes compilation and one-time
metadata upload. Terminal shows 1/2 and 2/2 stages and each sample. Compact
report includes all raw samples, original/new/native latencies and ratios.
Passing the smoke/accuracy contract does not establish a speed improvement.
