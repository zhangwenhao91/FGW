# Unnormalized Partial state with SIMD merge

```bash
git pull --ff-only
python fgw.py mha run run_mha_merge_rawsimd
cat reports/mha/latest/mha_merge_rawsimd_paste.txt
```

Only Q32768/KV4096 and Q32768/KV8192, B1H1 BF16 D128 noncausal page128.
Controls: raw Partial + SIMD merge, prior normalized Partial + SIMD merge,
installed native AscendC FIA. Task plan/Cube pipeline/R8 grid stay the same.
Raw vs normalized have different local code/output contracts, so each variant
compiles its own local and merge once. No original SIMT control is compiled.
Six measurement repeats x nine rotated triples reuse those builds. Per repeat
100 initial warmups/control + timing Event priming; per sample ten immediate
warmups. Same full-call warm-cache Event contract; no samples excluded.

For split segments only, local stores A=sum(exp(score-m)*V), m, and
l=sum(exp(score-m)), skipping per-element division and log(l). Direct complete
Q tiles still normalize/cast/write final O normally. FP32 Partial shape is
unchanged. The flat stats buffer has two planes of slots*128 FP32 values:
max plane followed by sum plane. Extra stats footprint for target plans is
14336 bytes relative to the original single LSE plane. Buffer sizes derive
from the selected plan, not hardcoded target shapes.

SIMD merge updates online states with:
Mnew=max(Mold,mi); alpha=exp(Mold-Mnew); beta=exp(mi-Mnew).
Numerator=oldNumerator*alpha + Ai*beta.
Denominator=oldDenominator*alpha + li*beta.
Final output=numerator/denominator, converted to BF16. Empty local sums give
zero new weight. Both control variants have separate correctly-sized
workspaces; never feed raw data into the normalized/LSE merge. This retains
the online reduction form and changes state format plus its normalization
location, not prefetching or task assignment.

CPU independent oracle compares unequal/empty raw partitions to dense
attention, including extreme scores. Local Ascend lowering covers mixed
full/partial tasks, both stats planes, long target shapes and direct Q tails.
Official Ascend950PR_9579 CAModel verifies Q257/KV512 with two programs:
direct tile, two split fragments and a one-row direct tail. Full output and
individual raw m/l/A states are compared to CPU reference; max BF16 output
error is 0.001788. A second merge-only fixture covers mixed two/seven pieces,
finite extreme maxima, sentinel/zero-sum states, Q tails and untouched direct
rows. Prior normalized local and SIMD merge generated C++ are unchanged in
local equivalence checks. Server compiler differs; two-seed hardware accuracy
runs before timing. No default/production path change or promised speedup.

Raw JSON includes both source sets, plan, build counts, workspace contract,
per-repeat samples and accuracy errors. Compact report checks one build per
variant and all 54 samples/control, prints overall/per-repeat ratios against
normalized SIMD and native AscendC, and retains every raw sample.
