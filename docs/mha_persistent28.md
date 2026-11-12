# 28-program常驻调度候选

运行 `python fgw.py mha run run_mha_persistent28`。

## 实现

v7新增persistent_programs=None/28，默认None，保持原逻辑program数量。
候选compute_grid=min(tasks*splits,28)，每program用串行循环处理
work_id=program_id+iteration*compute_grid，末轮越界任务跳过。
Q/K/V等固定形状缓冲区分配在外层，每任务重置output、maximum和total，并重新加载Q。
partial和direct都采用同一映射规则；merge本身不变，模型系数和split选择不变。
默认模式外层循环只有一次，期望编译器常量化简；因此也先跑默认模式精度，并保留性能对照。

这仅是program内多任务循环，不等于AscendC的跨核边界拆分/FD，也没有把流水贯穿到下一个query任务。
每任务仍有原KV流水启动与排空；收益要实测，不预先归因或保证。
任务交错映射还可能改变L2缓存局部性，计时必须包含短Q、尾块和不同KV。
显式候选只用于v7 Cube路径，不支持的规格拒绝覆盖。

## 运行顺序

- 当前模式30项Cube精度，候选30项非因果、9项单query因果精度。
- 同输入v7当前、persistent28、AscendC FIA v2小范围冒烟。
- KV2048/8192，B2/H8和B1/H1，Q1/16/128/129/512/1024/4096/8192/16384/32768：
  共40形状、120候选精度检查和4轮480条性能记录。
- 全部通过后，B2/H8/Q4096/KV2048做3轮Event及msprof_detail，
  同进程比较当前full/direct、persistent full/direct和FIA full，共5个调用对象。

计算tile128x128、manual、缓冲配置、冻结split模型、BF16输入输出和分页数据不变。
独立CPU参考检查，Event warm-cache和msprof清缓存分别比较，计数器为上游聚合口径。
报告geometry包含compute_work、compute_grid、persistent_programs和实际split_policy/num_splits。
persistent_comparisons中速度比=当前耗时/persistent耗时，大于1表示候选更快。

## 本地检查

检查program映射完整且无重叠，缓冲初始化保留，原两处KV流水计算体及merge计算体不变。
本地没有Ascend编译器/NPU，语法检查不代表外层循环及同步已在设备上通过。
因此精度失败会停止性能测量，并保存错误，不将候选设为默认。

## 上传

reports/mha/latest下：

- mha_persistent28_control_accuracy.json、mha_persistent28_accuracy.json、mha_persistent28_causal.json及对应.log。
- mha_persistent28_kv2048_summary.json/.log、mha_persistent28_kv8192_summary.json/.log。
- mha_persistent28_kv2048_persistent_device.cpp（生成的候选设备源码）。
- mha_persistent28_msprof_summary.json/.log及mha_persistent28_msprof_msprof_source.py。

冒烟失败上传mha_persistent28_smoke_summary.json和.log。完整Event样本JSON留本地。
如果当前控制组精度或候选编译失败，先上传相应accuracy报告及.log即可。
