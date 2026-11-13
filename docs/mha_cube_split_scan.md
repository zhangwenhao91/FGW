# 通用 Cube split=1/2/4/8/16/32 扫描

上一轮 split-KV 全部1524项精度通过；启用分片的38项中28项更快，几何
平均较同场split=1加速1.83倍，但page16尾块、部分Q128/Q129及全遮罩场景
因分片额外开销退步。本轮收集不同split的同场耗时，作为统一选择策略的依据。
通用kernel和现有fill28策略保持原样，不修改专用v7冻结系数。

## 运行与传回

```bash
git pull --ff-only
python fgw.py mha run run_mha_cube_split_scan
```

正常上传：

- `reports/mha/latest/mha_cube_split_scan_summary.json`
- `reports/mha/latest/mha_cube_split_scan.log`

若Git推送失败，直接执行并粘贴终端输出：

```bash
cat reports/mha/latest/mha_cube_split_scan_paste.txt
```

文本包含状态/异常、每个split的耗时、自动与最佳split、2%近优集合，以及
重点场景局部计算/merge的独立计时。每项结果都及时保存；不必重跑来导出文本。
默认不输出大型原始样本JSON，可用`--save-samples`保留本地JSONL。

## 覆盖

复用Cube/Vector对照的全部21个形状×FP16/BF16，共42项输入配置。
每项扫描1/2/4/8/16/32，容量不足的split记录明确跳过原因，不改变页表或
长度来凑数。总共212组合法配置，自动选择和split=1始终包含在候选中。

- Q1/D64/D128/KV2048、B1/H1/Q1/KV8192：确认短Q最佳分片数。
- Q16/D16～256、page48、因果D64/Q16：保留已有收益参照。
- D128/Q128、Q129尾块、因果Q129、page16、因果空行：重点分析退步项。
- D64/D128的Q512：query任务数已足够的控制组，也扫描强制分片。

每个合法split都先对照CPU FP32 reference，沿用原FP16/BF16阈值，检查输出
形状、类型、连续布局、有限值及空行精确为零。输入为相同有限随机Q/K/V，
seed7、原始乱序/重复/共享分页，和上一轮性能对照相同。完成全部精度检查
之后才开始测量。编译、执行或精度异常会保存当前配置和错误并停止。

## 完整调用与组件计时

完整预构建设备callable使用warm-cache NPU Event：包含输出/临时分配对
执行的影响、全部kernel和流可见启动间隙；排除编译、H2D、CPU参考、主机
长度校验。每项6轮，每轮预热20次、采31组覆盖所有候选的样本，轮换并
反转候选顺序，轮间反转输入配置顺序。报告每轮中位数、p90及相对auto收益。

针对8个重点形状、两个类型，额外用2轮、每轮预热5次/11组样本独立计时：

- `partial`：QK/softmax/PV及局部输出/LSE，包含本调用的中间输出分配。
- `merge`：消费预先算好的固定FP32局部输出/LSE，计其分配最终输出和合并。

只对split>1测这两个组件，split=1直接使用完整调用计时。组件是独立会话，
并非完整调用中嵌套打点，不能把两者相加或相减解释为精确的开销分解。
它们可辅助判断缩短局部计算的收益是否被merge延迟抵消。

最终按各轮中位数的中位数比较：

- `auto_splits`：当前fill28选择。
- `best_split`：本轮扫描的经验最佳值。
- `near_best_2pct`：耗时不超过经验最佳值1.02倍的候选。
- `smallest_near_best_2pct`：上述集合中最小分片数，供后续判断是否值得增加分片。
- `auto_regret_percent`：当前自动耗时相对经验最佳值增加的百分比。
- `variants`：每个split的耗时、MFU、较auto胜出轮数及组件耗时。

本次没有重新测AscendC；主要目标是收集split选择数据。扫描中的最佳值
不是独立验证过的默认策略。传回结果后再拟定统一选择规则，并用额外形状
验证，避免只记住这42项的最佳答案。本地无NPU，实机耗时待测。

可筛选运行，例如：

```bash
python fgw.py mha cube-split-scan --list-only
python fgw.py mha cube-split-scan --cases page16_tail d128_q128_kv2048 --dtypes bfloat16
```
