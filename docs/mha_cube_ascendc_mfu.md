# 当前通用 Cube：MFU 与 AscendC 加速比

保持当前通用 Cube kernel 不变（BR32/BC32、split=1），与安装的 CANN FIA v2
比较，测两者 MFU 与耗时。这里的“当前”指新验证的通用 Cube，**不是默认
v7 的 BF16/D128 专用快速路径**；没有加入 split-KV。

## 运行与上传

激活现有 zwh 环境并 source CANN set_env.sh 后：

```bash
git pull --ff-only
python fgw.py mha run run_mha_cube_ascendc_mfu
```

先跑 4 项接口/精度/计时冒烟（D64 非因果、D128 右对齐因果，两个类型），
成功后跑全部 50 项。编译或精度失败会保存当前样例和异常并停止。

默认直接输出紧凑报告，无需上传大型原始样本 JSON：

- `reports/mha/latest/mha_cube_ascendc_mfu_summary.json`
- `reports/mha/latest/mha_cube_ascendc_mfu.log`
- `reports/mha/latest/mha_cube_ascendc_mfu_smoke_summary.json`
- `reports/mha/latest/mha_cube_ascendc_mfu_smoke.log`

有需要可用 `--save-samples` 额外保留本地 JSONL 原始计时。

## 覆盖与原生接口

复用上一轮 21 形状 × FP16/BF16，再补 D64/D128 的 Q4096/Q32768、
B2/H2/KV2048，共 25 形状 × 两种类型。短 Q、不同 D、KV 尾块和因果
均保持同一通用 Cube 路径。所有输入 Q/K/V 为相同的有限随机值，保留原始
乱序/重复/共享分页；无效区 NaN 已在独立精度回归验证，本轮按既有 FIA
性能基准使用有限缓存池。

FIA 为 `torch_npu.npu_fused_infer_attention_score_v2`，使用原 K/V 的
BnBsH 零拷贝 view、同一页表、同一长度、BSND Q/O、原 D 的缩放。
采用安装版本，不从 ops-transformer master 重新构建。

依照[原生接口文档](https://github.com/Ascend/op-plugin/blob/master/docs/zh/custom_APIs/torch_npu/torch_npu-npu_fused_infer_attention_score_v2.md)：

- 非因果 `sparse_mode=0`，不传 mask，`inner_precise=0`。
- 因果 `sparse_mode=3`（右下角对齐），传 2048×2048 的 bool 压缩 mask，
  True 表示遮罩。普通场景 `inner_precise=0`；存在全遮罩行时显式设为 2，
  启用行无效修正，并完整计入其耗时。参数逐项写入报告。
- 分页 `block_size` 要求 128～512 且为 128 的倍数，因此 page16/page48
  两个形状、共4项仅测 Cube MFU，标记为 AscendC skipped，**不进入加速比
  汇总**。不修改页大小或通过重排缓存伪造同规格比较。

因此共 50 项 Cube 测量、46 项原生 AscendC 配对比较。文档支持范围仍需
安装的 CANN 版本实测确认；编译、执行或精度异常不会静默换接口或放宽阈值。
两边都对照 CPU FP32 reference，并检查形状、类型、连续输出、有限值与
空行精确为零，沿用原来的 FP16/BF16 精度阈值。

## MFU 和计时口径

按仓库 `MFU.md`：

```text
非因果：有效 FLOPs = 4 * H * D * Q * sum(KV长度)
右对齐因果：有效 FLOPs = 4 * H * D * sum_b sum_q max(KV长度[b] - Q + q + 1, 0)
MFU(%) = 有效 FLOPs / (耗时秒 * 378e12) * 100
加速比 = AscendC耗时 / 通用Cube耗时
```

分子只计原始 D、有效 query/key 的 QK+PV，不计补齐、softmax 或被遮罩的
位置。两个实现使用相同分子及 28 Cube 核 FP16/BF16 dense 峰值 378 TFLOPS，
设备核数会验证；其他设备必须显式指定峰值。

计时为 warm-cache NPU Event，完整预构建设备 callable：包含输出/临时分配
对执行的影响、全部 kernel 和流可见启动间隙，排除编译、H2D、CPU reference、
主机长度检查、mask/列表构建与零拷贝 view 创建，不是 CPU 墙钟或 msprof 时间。
先完成全部样例的精度检查，再对每项做6轮、每轮20次预热、31对AB/BA交替
样本，轮间反转形状顺序。

报告保存每轮中位数、p90、配对加速比；最终取各轮中位数的中位数，计算
TFLOPS/MFU/加速比，并给出每轮胜出数及几何平均加速比。
Cube-only 样例仍列 MFU，不会计入 paired_cases。

已有上一轮 Cube/Vector 报告可回算：在那42项范围内，最高通用 Cube MFU
为约0.4021%（BF16、B2/H2/Q512/D128/KV2048，1412.807 μs）。这是历史
耗时的公式回算，不是本轮新测量，也不是专用 v7 快路径的 MFU。

可筛选运行：

```bash
python fgw.py mha cube-ascendc --list-only
python fgw.py mha cube-ascendc --suite long-q --dtypes bfloat16
```
