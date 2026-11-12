# v7 split=1/2/4/8/16扫描

运行 `python fgw.py mha run run_mha_v7_split_scan`。
当前自动split策略保持不变；不重测AscendC，本轮专门寻找最佳split。

## 范围

KV512/2048/8192/32768；B2/H8和B1/H1；每组Q=1/16/128/129/512/1024/4096/8192。
共64个形状。KV512只有4个KV128计算块，split8/16记为skipped而非创建空split。
其余测试全部五个split，总共288个候选精度检查，4轮1152条性能记录。
Q8192上界让最大compute grid为16384，避免为了扫描高split引入过大grid。

v7参数num_splits现支持None（原自动策略）和1/2/4/8/16。
显式数值不得超过页表容量对应的KV128块数；覆盖仅适用于v7 Cube路径。
单split采用既有direct计算，多split采用既有partial+merge，三个T.prim_func计算体均未修改。

## 控制变量

BF16输入输出、D128、page128、非因果、seed7、manual/stages2/Q驻留和同一缓冲配置。
相同Q/K/V、随机物理页和块表、实际KV长度。每候选先独立CPU分块参考精度检查。
CPU参考FP32计算并按BF16舍入，rtol0.01/atol0.004；错误shape/dtype也视为失败。

自动策略所选split直接复用默认build_kernel返回的callable，并作为该split的计时项，
不额外重复测同一个split来制造独立基线。报告记录自动选择和实际几何。
四轮各候选预热20次、各51个独立Event样本，候选顺序旋转并反转，轮间反转初始顺序。
warm-cache完整调用，包括Q转换、compute、分配及merge，排除编译/H2D/CPU参考。
不将整个策略差值视为merge独立耗时。

## 汇总

best_by_case记录：

- 自动split、每候选四轮中位耗时的中位数、最小/最大轮耗时、MFU。
- 按中位耗时选择的最佳split、相对自动策略的提速。
- 每轮胜者winner_per_round、距最佳耗时不超过2%的候选within_2pct_of_best。
- 是否所有适用候选精度通过；有错误时最佳项仅代表已成功候选，不能当完整扫描结论。

comparisons为每轮相对自动split的速度比（自动耗时/候选耗时，>1为更快）。

## 运行及上传

先KV2048、Q1/129两组B/H冒烟，通过后按KV依次执行完整扫描。
首次编译候选较多，进度及错误持续写入报告，完整KV失败后继续后续KV。
reports/mha/latest下上传四组：
`mha_v7_split_scan_kv*_summary.json` 和 `mha_v7_split_scan_kv*.log`。
原始Event样本完整JSON保留本地。冒烟失败上传mha_v7_split_scan_smoke_summary.json和.log。

单独重跑示例：
`python fgw.py mha split-scan --kv 32768 --rounds 4 --warmup 20 --repeat 51 --output mha_v7_split_scan_kv32768.json`。
