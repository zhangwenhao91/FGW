# 通用 Cube 与 v2 Vector 耗时对照

通用 Cube 已通过 48/48 冒烟、360/360 边界、540/540 原扩展回归。
本轮比较它与 v2 Vector 在相同输入上的设备调用耗时，暂不改变默认路径。

## 运行

在已有 zwh/CANN 环境中：

```bash
git pull --ff-only
python fgw.py mha run run_mha_cube_vector
```

上传这三个文件：

- `reports/mha/latest/mha_cube_vector_paired_summary.json`：紧凑汇总及每轮中位数。
- `reports/mha/latest/mha_cube_vector_paired.json`：包含所有原始样本。
- `reports/mha/latest/mha_cube_vector_paired.log`：运行日志。

## 范围

21 个形状，每个测试 FP16、BF16，总共 42 项对照。固定 seed=7。

| 维度 | 覆盖 |
| --- | --- |
| D 扫描 | Q16、KV512，D=16/48/64/80/96/128/192/256 |
| Q 扫描 | D64/D128，Q1/128/512，KV2048 |
| 单请求长缓存 | B1/H1/Q1/D64/KV8192 |
| 小页与跨页 | page16、page48，非整块 KV |
| Q 与 KV 尾块 | Q129、D128、KV511/257 |
| 因果 | D64/Q16、D128/Q129、D256/Q33，含无可见 key 的行和空请求 |

其他形状为 B2/H2。页表包含乱序、共享及重复物理页，无效缓存填 NaN。
两个实现均先对照独立 CPU FP32 reference，沿用原精度阈值，检查输出布局、
有限值和空行精确为零。所有选中形状编译及精度检查完成后才开始计时。

## 口径

通用 Cube 固定 split=1；v2 使用原有自动 split/merge，split=1 时会选择原 v1
Vector kernel。此处比较两套完整设备计算策略，不隔离矩阵乘指令耗时。

两个预构建 callable 接收相同 NPU tensor，每个 Event 样本仅执行一次。
包含输出/临时张量分配对设备执行的影响、全部 kernel、流可见的启动间隙；
排除编译、H2D、CPU reference，以及公共 API 的主机侧校验与长度同步。
因此不是公共 API 的 CPU 墙钟延迟，也不包含 `cache_seqlens.cpu()` 的代价。

Warm-cache：每个形状每轮先交替预热 20 次；随后采集 31 对样本，AB/BA
交替，6 轮之间反转形状顺序。报告每轮中位数、p90、配对加速比，汇总各轮
中位数的中位数及胜出轮数。加速比为 **v2 Vector 耗时 / 通用 Cube 耗时**，
大于 1 表示通用 Cube 更快；不扣除 Event 基线开销。

汇总分组区分：原 `vector_fallback`、原 `stable_rejects_lengths`、
`cube_control`，另按数据类型汇总。BF16/D128 满块控制组也与 v2 比较，
**没有与优化过的默认 v7 Cube 比较**；不能据此替换现有 D128 快速路径。
NPU 测量结果传回后再判断哪些原回退规格适合默认转向通用 Cube。

可独立筛选，例如：

```bash
python fgw.py mha cube-vector --cases page16_tail page48_crossing --rounds 2 --pairs 7 --warmup 5
python fgw.py mha cube-vector --list-only
```
