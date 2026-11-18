# Explicit SIMD online R8 merge: warmed same-round comparison

```bash
git pull --ff-only
python fgw.py mha run run_mha_merge_simd
cat reports/mha/latest/mha_merge_simd_paste.txt
```

Only Q32768/KV4096 and Q32768/KV8192, B1H1 BF16 D128 noncausal page128.
Candidate explicit SIMD, original fixed SIMT R8, and installed AscendC FIA.
Shared single compiled local kernel; fixed R8 task grid, same online softmax
merge equations and Partial/LSE/output formats, same GM-to-UB and UB-to-GM
T.copy shapes. No precomputed weights, prefetch or task regrouping.

SIMD row-stat vectors compute max, old/new exp weights and denominator. Store
row weights, use VST_VLD barrier then broadcast each row's weights across two
64-FP32 chunks of D128, performing the original two multiplies and add.
Final SIMD divide and BF16 packed stores precede the original output DMA.
Stats UB arrays are padded to 64 elements so unmasked full-vector loads stay
inside allocation; masks restrict live stats to R8. Invalid LSEs receive zero
new weight, retaining the baseline all-empty-row behavior. FP32 rounding and
instruction implementations may differ from SIMT, so hardware accuracy must
still be checked. SIMD VF latency annotations are estimates, not measured costs.

The generated candidate uses __simd_vf__, not __simt_vf__. Local lowering is
checked; official Ascend950PR_9579 CAModel covers Q3713 with untouched direct
rows and a one-row tail, four groups of seven parts, and mixed two/seven-piece
groups with finite extreme LSEs, sentinel-invalid fragments and an all-empty
row. Fixed SIMT baseline generated C++ remains byte-identical. Local TileLang
version differs from server a7cb753d; the NPU run checks two accuracy seeds
before timing, including the exact server SIMD API/toolchain behavior.

One compilation round per shape (one local, two merge kernels), six measurement
repeats using that build. Each repeat warms all three controls 100 times,
primes timing Events outside measurement, then runs nine rotated triples with
ten immediate warmups per measured call. Each control occupies every order
position three times per repeat. No slow sample exclusion. Report all 54 raw
samples/control, aggregate medians, per-repeat ratios/ranges/win counts.
Same full-call warm-cache Event contract as confirmation runs: ordinary
allocations/all kernels included, compilation/one-time metadata upload excluded.
Terminal progress and independent process supervision retained. No default or
production policy change, no claimed performance benefit before NPU results.
