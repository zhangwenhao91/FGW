# Two remaining AscendC targets

Run `python fgw.py mha run run_mha_static_targets` after pulling main.
Paste `reports/mha/latest/mha_static_targets_paste.txt`.

Scope: B1H1 BF16 D128 page128 noncausal, Q32768 and KV4096/8192 only.
Each shape compares current cost-policy R8 with explicit balanced P28/R8,
P28/R16, P28/R32 and P27/R32. Ten configurations total. At KV8192 the cost
control currently equals P28/R8; retaining both provides an independent build
control, not another algorithm. No default model or production v7 changes.

The aim is to offset the final partial Q-tile wave with selective split-KV,
while reducing merge launch work through larger row groups. R16/R32 reduce
merge programs but increase each program's state and work, so benefit is not
assumed. Each configuration checks two correctness seeds, compiles once, and
uses three rotated paired samples with five immediate warmups per control,
including AscendC FIA. Full-call latency determines speedup; standalone
local/merge times are diagnostic and must not simply be summed. Generated
sources, plans, accuracy checks and raw samples remain in the unique run folder.
Failures stop the scan and preserve completed reports. Terminal shows x/10,
compilation, accuracy and timing progress. Near-1x wins require later repeat
measurements, rather than selecting a noisy minimum as a confirmed win.

Local checks: argument/plan coverage, incomplete report handling, shell syntax,
and Ascend lowering for merge P28/R16, P28/R32, P27/R32. Local TileLang differs
from the server; NPU correctness and speed are not yet established.
