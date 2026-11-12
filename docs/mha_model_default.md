# 已验证范围默认采用冻结split模型，并重测AscendC

运行 `python fgw.py mha run run_mha_model_default_ascendc`。

## 当前行为

build_kernel和公开mha_paged_decode默认split_policy="auto"。
auto只有同时满足以下条件才采用cost_model：

- 28 Cube，BF16，D128，page128，非因果；manual/stages2/buffers2/KV版本2/Q驻留。
- B在1到4，H在1到16，B*H<=32，Q在1到8192，KV容量在512到32768且128对齐。
- 确认每请求KV长度恰等于页表容量（固定满缓存）。

这些是验证过的配置覆盖范围，范围内的所有整数形状并未逐个测试，不按单个B/H/Q查表。
公开入口复用原有长度检查得到满缓存信息，没有新增一次CPU同步。
变长、等长但短于容量、不满足配置/设备条件的输入保留legacy；原有非Cube回退不变。
Q16384/32768暂在默认模型范围外，用legacy（本轮对应形状本来就选择split1）。

低层build_kernel不读取设备长度，默认full_kv_cache=False，因此不能证明满缓存时保守使用legacy。
已知固定满缓存的调用方应传full_kv_cache=True，声明必须真实；相应MFU/msprof/AscendC基准已同步。
公开mha_paged_decode自动检查，无需调用方声明。报告记录resolved split_policy和实际num_splits。
显式split_policy="legacy"可复现旧策略；显式num_splits优先于策略。
显式cost_model仍是原实验入口，调用方需自行保证语义及校准适用性。

冻结系数JSON不变，不重新拟合、不使用本轮AscendC结果更新模型。
所有kernel计算体不变。历史split扫描和旧Q驻留实验已显式固定legacy，防止旧基线含义漂移。

## 重新对照

先公开入口30项非因果Cube精度及9项单query因果精度，再做小范围新默认/legacy/FIA v2冒烟。
随后三方使用同输入、分页布局、BF16输出，在同一进程交替测量4轮。

- KV512/8192/32768：B2/H8、B1/H1，Q1/16/128/129/512/1024/4096/8192。
- KV2048：两组B/H，完整Q1到32768的16形状。
- KV4096/16384：B1/H1、B1/H8、B4/H4、B2/H16，Q1/64/129/256/512/2048/8192。

共136形状、408个候选独立精度检查、4轮1632条性能记录（另有前置回归及冒烟）。
每轮预热各20次、各51个单次warm-cache Event样本，旋转/反转候选顺序，计算有效QK/PV MFU。
计时包括原生调用、分配、Q转换和merge；不包括CPU参考/H2D/元数据准备或编译。
不调用公开入口的逐次长度校验来计时，保持此前compiled callable测量口径。

v7_manual_bf16为当前默认，v7_legacy_bf16为旧策略。
comparisons中分别给FIA耗时/当前v7耗时、FIA耗时/旧v7耗时，>1表示相应v7更快。
policy_comparisons给旧v7耗时/当前v7耗时，>1表示新默认更快。
报告保存冻结模型hash/权重、实际策略、split数、误差和原始Event样本。

## 上传

reports/mha/latest下优先上传：

- mha_model_default_kv*_summary.json和对应mha_model_default_kv*.log（六组）。
- mha_model_default_accuracy.json/.log、mha_model_default_causal.json/.log。

若冒烟失败上传mha_model_default_smoke_summary.json和.log。
完整计时样本JSON保留本地。单KV手动重跑可使用：
`python fgw.py mha compare-ascendc --compare-legacy --fia-api v2 --kv 2048 --rounds 4 --warmup 20 --repeat 51 --output mha_model_default_kv2048.json`。
