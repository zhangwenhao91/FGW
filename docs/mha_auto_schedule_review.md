# Q4096自动调度审阅与候选

审阅实际导出manifest、direct_device.cpp、028_tl.AutoSchedule.py、031_tl.InsertSync.py；原始文件位于reports/mha/latest/mha_auto_schedule_20261111_162626_650071/q4096。配置为Q驻留、两级流水、各流式缓冲两版本，single split。

## 观察

- 028中schedule_unit的stage均为0；循环仅标记num_stages=2，没有enable_offset。双缓冲版本存在，但未见跨迭代非零stage安排。
- Q从GM到L1，再到L0A，在KV循环外只执行一次。
- 自动调度已重排V加载：设备源码中K加载、K→L0B、QK发射后才加载V，再执行QK结果→UB。V加载可以与已发射的Cube计算交叠；不能把stage0解读成所有硬件完全串行。
- P_l1、kb、pa、vb实际为单版本；K/V、qk/pv、scores/probs/alpha/current为两版本。P从Vector写L1后，Cube等待两个AIV的通知再取P。
- 每次循环仍依赖QK→softmax→P→PV→输出累加；跨迭代交叠是待验证的候选，不直接据此声称存在可消除的全部等待。
- 编译器还生成了分页物理地址相关OOB guard/fill。此次不删这些保护，也不改变页表有效性的前置条件。

## 三组对照

1. auto：完整保留当前默认调度。
2. auto-offset：只为T.Pipelined开启enable_offset，让编译器自行求跨迭代安排，缓冲策略保持原值。
3. manual：开启offset，并使用T.Stage显式指定阶段；同步、多缓冲重写仍由编译器完成，不手写硬件flag。

手工候选：K加载0；K→L0B及QK为1；QK→UB和V加载为3；softmax为5；P→L1、PV操作数准备及PV为6；PV→UB与输出累加为7。Q驻留仍在循环外。这个方案借鉴上游Stage组织，但针对本地导出中已驻留的Q移除重复搬运，并单独设置内存策略，不能视为已验证的最优调度。

手工内存策略：K/V L1三版本、scores三版本、alpha三版本；current只在同阶段生产/消费，候选设为一版本；probs、qk/pv两版本。用减少current版本补偿scores额外UB占用。真实容量、WAR依赖及调度可行性由编译器检查，失败候选不解除内存限制强行运行。

候选实验阶段默认auto。只有编译、30项非因果及9项单query因果精度通过，才采集长Q的Event和msprof。两个计时口径分别比较；手工候选同时改变阶段和必要缓冲策略，auto-offset用于帮助区分启用offset本身的收益。


## 短 Q 交替计时

运行 `python fgw.py mha run run_mha_schedule_short_q`。输出为
`reports/mha/latest/mha_schedule_short_q_paired.json` 和同名 `.log`。

比较 auto、auto-offset、manual，同一进程使用相同输入，全部编译及 CPU 参考精度检查完成后再计时。
固定 BF16、D128、page128、Q resident、stages2；缓冲区配置遵循各调度候选。
覆盖 B2/H8 的 Q16、32、127、129（KV2048），B1/H16/Q1/KV512 的因果及非因果，
B1/H8/Q1/KV8192，以及 B2/H64/Q1/KV2048 的 split=1 直写路径。
每项 8 轮，每轮预热各30次、各61次完整调用 Event 测量，轮换并反转调度顺序，轮间反转形状顺序。
报告保留原始样本、每轮中位数、对 auto 的速度比、获胜轮数及执行路径；采用 warm cache，不扣除计时开销。
本脚本不修改默认调度。短 Q 主要比较延迟，不以其 MFU 判断 Cube 算力上限。


## 验证后采用的默认值

长 Q 两轮精度及性能、短 Q 八轮交替测量通过后，v7 统一默认 manual。
API 的 build_kernel/mha_paged_decode，以及 precision/bench/msprof/export 入口均一致。
保留显式 auto 和 auto-offset；历史实现默认仍解析为 auto。
Q驻留历史 paired 实验显式固定 auto，避免旧对照随当前默认值改变。

长 Q warm-cache Event：manual 的 Q4096 330.64–331.07 us、MFU54.91–54.98%；
Q32768 2658.21–2659.53 us、MFU54.69–54.71%。msprof独立计时确认同方向收益。
短 Q 精度通过，Q16/32/127/129、Q1长KV及Q1直写六项均8/8轮快于auto。
B1/H16/Q1/KV512非因果和因果分别为manual30.31/30.14 us、auto30.12/30.12 us；
非因果存在小幅回退，统一默认接受该取舍，不增加形状分派。
短Q统计为各轮中位耗时的中位数，完整JSON尚未成功上传；摘要由用户粘贴确认，log已入库。
