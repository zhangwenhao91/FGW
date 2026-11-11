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

默认仍auto。只有编译、30项非因果及9项单query因果精度通过，才采集长Q的Event和msprof。两个计时口径分别比较；手工候选同时改变阶段和必要缓冲策略，auto-offset用于帮助区分启用offset本身的收益。
