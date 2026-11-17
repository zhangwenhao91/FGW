# Default v7/FIA on balanced program workloads

No kernel or split coefficient changes. Run:

```bash
python fgw.py mha run run_mha_v7_balanced_ascendc
cat reports/mha/latest/mha_v7_balanced_ascendc_paste.txt
```

Twelve B1H1 BF16/D128/page128 noncausal full-cache cases:

| Q | Q tiles | Default split | Programs | Q tiles per program |
|---:|---:|---:|---:|---:|
|3584|28|1|28|1|
|7168|56|1|28|2|
|17920|140|1|28|5|
|28672|224|1|28|8|

Each Q is crossed with KV8192/16384/32768. Q3584/7168 use the unchanged frozen
model; Q17920/28672 use the existing legacy selector outside its Q range.
All select split1. The benchmark checks actual split/grid/work counts on the
NPU and fails if split1 with equal work across all28 programs is not selected.
All Q tiles and KV blocks are full, every request has the same full live cache.
This balances task counts; it does not assert identical measured core times.

The existing protocol remains: seeds7/19 vs chunked CPU reference, two independent
TL builds per shape with cache disabled, two rounds of seven paired samples,
five immediate warmups for each side before each Event sample, alternating
order, installed AscendC FIA. Compile and warmups are excluded from speedup.
There are24 TL builds total. Per-case task counts, full timings, round ranges and
useful TFLOP/s are reported, along with a4x3 speedup matrix. Useful FLOPs are
4*Q*KV*D for B1H1 noncausal attention; this is throughput, not hardware MFU.

These Q values differ from the previous grid. A change in raw latency alone
does not isolate a balancing benefit. Compare TL/FIA speedup and useful throughput,
and treat the result as evidence about the scheduling hypothesis, not a matched
same-shape causal experiment. The source and default compute path are unchanged.
