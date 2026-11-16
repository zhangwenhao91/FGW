# NSA-inspired v7 pipeline experiment

Reference: [pinned NSA compressed attention](https://github.com/yyttt6/TileOPs/blob/2ffc69b29d1e456f28ade01d50fc3a6f715d1392/src/tileops/_kernels/ascend/attention/nsa_compressed_varlen.py).
Adapts its two-stage offset schedule and streamed-buffer lifetime organization;
does not import the NSA algorithm, compressed K/V or the upstream Gather kernel.

The opt-in `schedule_mode="nsa"` groups K/V loading, QK and score transfer in
stage0, and softmax, PV and online output accumulation in stage1. It double
buffers L1 operands and streamed results while keeping L0A/L0B operands
single-version (the 128x128 tile already fills 64 KiB per operand memory across
Q/P or K/V). It keeps online maximum/denominator/output
single-version. Existing v7 manual arithmetic, shapes, layout handling and default
schedule are retained. The candidate retains v7's GM-to-L1 V transpose; the NSA
L1-to-L0 transpose arrangement is not part of this first experiment.

For each shape, build the ordinary current manual default first, then reuse its
exact split count for the candidate. The frozen default split model selects
split1 for B1H1/Q4096 and Q8192/KV8192: neither target has merge overhead.
The candidate does not use the model to select a different split after changing
schedule. Both retain tile128, BF16 D128, page128 and persistent cap28.

`python fgw.py mha run run_mha_nsa_pipeline` first gates compilation/accuracy on
Q129/KV512 (Q-tail, split path) and Q4096/KV8192 (direct path), then checks eight
short/long-Q configurations against a chunked CPU reference with seeds7/19.
Native AscendC FIA v2 uses identical original paged inputs. All variants must pass
before timing. Six rotating/reversing paired rounds measure full device callables;
separate compute-component rounds exclude Q preparation and merge. Components
are not additive and cannot be subtracted from full-call timing. The two direct
targets have no merge, so compute timing isolates the pipeline most directly.

Full reports and generated manual/candidate C++ sources are in a fresh directory
under reports/mha/latest/mha_nsa_pipeline.*. Compact copies are published as
mha_nsa_pipeline_smoke_paste.txt and mha_nsa_pipeline_full_paste.txt, with the
current run combined in mha_nsa_pipeline_paste.txt. On failure,
read the smoke file and log; full measurement is gated. Default v7 is not promoted
or changed by this experiment. Local syntax/structural checks do not establish
NPU compiler support, scheduling correctness or speedup.
