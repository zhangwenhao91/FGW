# v7 与参考 PR3896 的长短 Q MFU

运行 `python fgw.py mha run run_mha_v7_pr3896_mfu`。
固定参考 PR head 为 `49a51226eb4068d5c0bf205cee886d2a7dcf0b05`，
原样保存 core.py/example_mha.py，运行时检查 SHA256。测试机器不需要联网下载或改装 TileLang。

## 两组结果

- `v7_production`：B2/H8，重测当前 v7 manual 的长短 Q。
- `matched_single_head`：B1/H1，同一份逻辑输入比较 v7、PR 原示例 FP32 输出、PR 核心支持的 BF16 输出。

Q=1/4/16/32/64/127/128/129/256/512/1024/2048/4096/8192/16384/32768；
KV2048、BF16输入、D128、非因果、seed7，v7使用page128随机物理页。
参考接口只有二维单head数据，不能直接表达B2/H8；不能将多个独立head拼成一个Q序列共用K/V。
B1/H1组的参考K/V在计时外按页表还原为稠密序列，数值与v7一致。

PR使用原始核心的动态Q/KV（支持Q尾块）、默认auto调度和stages3、FAST_MATH；
v7使用已验证的manual/stages2、Q驻留和BF16输出。没有调优或改写PR kernel。
FP32输出保持原示例行为，BF16输出通过核心已有参数指定。两种PR输出都明确标注。

## 精度与计时

先用独立FP32 CPU分块attention参考检查每个候选，输出参考按BF16舍入，
比较时转FP32，统一rtol=0.01/atol=0.004。错误保留在报告中，不把失败项计为性能结果。
参考编译失败会缓存错误，其他v7形状继续；没有声称本地已完成NPU编译或精度验证。

同一进程、同形状各候选先预热20次，每轮交替执行，每项各51次独立NPU Event，2轮反转候选顺序。
统计中位数，warm-cache，不扣Event开销。计时只包含各自原生callable：
v7包括Q布局转换、partial/merge或直写；PR直接调用已准备的二维Q/K/V，输出分配包含在内。
不包含CPU参考、H2D、PR稠密KV准备、编译。生产组与单head组不能直接比较耗时或速度比。
PR BF16与v7输出dtype相同，但分页能力、split策略、编译动态性不同，结果是各自原生调用的性能对照。

有效计算量 `4*B*H*Q*KV*128`，仅QK/PV，排除尾块填充及softmax；
峰值沿用docs/MFU.md的28-AIC BF16 dense Cube 378 TFLOPS。
PR无因果参数，因此本轮不加入因果规格。短Q仍主要看延迟，MFU低不能直接推断Cube执行效率。

## 报告

均在reports/mha/latest：

- `mha_v7_pr3896_mfu.json`：完整原始Event样本。
- `mha_v7_pr3896_mfu_summary.json`：同样的精度状态、每轮中位耗时、TFLOPS和MFU，去除原始样本。
- `mha_v7_pr3896_mfu.log`：进度、MFU及异常。

优先上传summary.json及log；完整JSON保留本地供进一步排查。
可用 `python fgw.py mha compare-pr-mfu --list-only` 查看形状，
或用 `--q-lengths 1 128 4096 --rounds 1 --repeat 11 --warmup 5` 先运行小范围。
