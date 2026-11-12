# 常驻28短Q控制组复测

运行 `python fgw.py mha run run_mha_persistent28_short`。
仅测试，不切换默认调度；kernel和冻结split模型均不修改。

## 形状

17个同grid控制组：

- B2/H8、KV2048、Q1/16/128。
- B1/H1、KV2048及8192，各Q1/16/128/129/512/1024。
- B1/H1、KV512、Q1，分别非因果和因果。

增加2个收益参照组：B2/H8/Q4096/KV2048、B1/H1/Q8192/KV2048。
两个参照组实际grid会从512/64变成28；其余控制组的compute_grid必须相同，否则报错。

所有组使用BF16、D128、page128、固定满缓存、相同输入及冻结split选择。
候选为persistent_programs=None和28，调度manual，其余配置相同。
记录实际split_policy、num_splits、compute_work、compute_grid、执行路径。

## 精度及计时

同一进程先完成所有编译、CPU分块参考精度检查，之后才进入测量。
12轮，每形状每轮预热各30次、61对单次Event测量，AB/BA交替并反转每轮起始顺序，
形状顺序逐轮反转。warm-cache，不扣计时开销，不丢弃离群值。

保留每轮中位数、最小/P90、逐对速度比中位数、原始样本及获胜轮数。
控制组如显示生成代码哈希完全相同，只能说明代码字符串一致，性能仍可能受时钟、缓存和调用开销影响。
不同源码哈希也不自动等于数学计算不同，可能有编译命名差异，需结合设备源码检查。

## 上传

reports/mha/latest下：

- mha_persistent28_short_paired_summary.json
- mha_persistent28_short_paired.log

完整mha_persistent28_short_paired.json含所有原始样本，保留本地。
结果确认后再统一默认28上限（min(任务数,28)），不按Q长度增加版本分支。
