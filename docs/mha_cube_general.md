# 通用 Cube 候选：先扩大功能覆盖

新增 `cube_mode="general"`，通过同一个通用 kernel 对所有支持规格执行
Cube QK 与 PV，不调用 v2，不因短 Q、因果或尾块改用 Vector 点积。
这是尚待 NPU 验证的候选；不改变已经验证的 v7 默认路径。

| 项目 | 通用候选 |
| --- | --- |
| 数据类型 | FP16 / BF16，保持输入输出类型，FP32 累加 |
| D | 16～256，16 的倍数，包括非二次幂 |
| page_size | 正的 16 倍数，缓存池包含完整物理页 |
| Q | 任意正长度，32 行 tile，支持尾块 |
| KV | 0～页表容量，允许非 128 对齐、部分末页 |
| 因果 | 支持，右对齐：第 q 行最多读取 max(KV长度-Q长度+q+1,0) 个 key |
| 空行 | 精确输出零 |
| 输入布局 | Q[B,Q,H,D]、K/V[pool,H,D]，与当前 API 相同 |

保留其他输入契约：同 NPU、连续布局、Q/K/V 类型一致，页表和长度 int32，
有效页号必须指向缓存池。此候选不新增 GQA、量化、滑窗或 KV 更新功能。

实现位于 `operators/mha/kernels/mha_paged_decode_ascend_cube_general.py`。
BR=32、BC=16，两 AIV 各处理 16 个 query；BC 整除所有支持的页大小，
因此一次 KV tile 不会跨物理页。D 补到至少 32 的二次幂，补零由 kernel
完成，不在主机侧扩展缓存，也不改变 softmax 的原始 D 缩放。
QK 和 PV 是两次 L0 输入 `T.gemm`；Vector 仍负责搬运、掩码、softmax、
归一化及类型转换，这些属于 Cube attention 的正常组成部分。

读取 K/V 前检查有效长度，补齐的特征和 KV token 清零，防止无效位置的
NaN 经 PV 传播。有限负数哨兵与显式概率清零共同处理全遮罩行。
UB ND→compact NZ→L1 的转换使用当前安装提交支持的 layout-driven copy，
不直接套用 D128/BC128 的专用 SIMD 打包。

首版固定 split=1，串行遍历 KV；暂不移植冻结 split 模型、常驻 grid、
手工流水优化，也不接受这些配置的覆盖。D128 专用优化不一定适用于
其他 D 与 BC。预期小 tile、SIMT 搬运和填充可能产生开销；必须实测，
不能据此宣称所有形状会加速。

## NPU 上运行

激活现有 zwh 环境并 source 已验证的 CANN set_env.sh 后：

```bash
git pull --ff-only
python fgw.py mha run run_mha_cube_general
```

依次执行 48 项冒烟、360 项边界集、原 extended 集 540 项回归。
覆盖两个数据类型、因果/非因果、满/混合/空缓存、三个随机种子；
边界集另覆盖 Q=31/32/33/129、D=16/32/48/64/80/96/128/192/256。
每项沿用原精度阈值，包含乱序/重复/共享物理页与无效缓存 NaN。
报告逐项保存，明确标记 `cube_general`，记录生成代码哈希，并检查设备源码
包含至少两处 `asc_mmad` 调用；编译/运行错误停止后续阶段。

上传 `reports/mha/latest/mha_cube_general_*.json` 和对应 `.log`。
本地只能做语法、入口与数学模型检查；不代表通过 NPU 编译或精度验证。
收到实机回归后再决定自动替代原回退分支，并补测相同输入的性能。

单独运行例子：

```bash
python fgw.py mha accuracy --cube-mode general --suite general --seeds 7
```

API：`operators.mha.current.mha_paged_decode(..., cube_mode="general")`。
候选不捕获编译错误去调用 Vector，因此不会用回退掩盖新路径问题。
