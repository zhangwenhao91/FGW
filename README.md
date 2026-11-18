# MHA Ascend — 50fca6d clean

从 FGW 提交 `50fca6d93466f6c8eed2388b58366c3aae819299` 提取的独立精简版本。
这是历史版本，不含后续 stream、raw/SIMD merge、寄存器 softmax 等优化。

只保留：

* `operators/mha/kernels/mha_paged_decode.py`：该提交的静态 MHA 算子，仅重命名文件。
* `operators/mha/static_schedule.py`：该提交的 host 调度和冻结成本参数。
* `example.py`：最小 NPU 精度验证示例。
* 来源清单、必要的包文件和 TileLang 许可证。
* 本分支新增的单版本全规格基准、主机资源守护和对应检查。

算子与调度文件内容逐字保留；`SOURCE_MANIFEST.json` 记录原路径和 SHA256。
独立分支只包含精简快照与本次基准的新增提交，不包含旧仓库提交历史、历史测试、报告和实验脚本。

## 环境与规格

使用原 NPU 环境：Ascend950PR_9579（28 Cube）、Torch 2.13.0+cpu、
torch_npu 2.13.0、TileLang 0.1.15+cpu.gita7cb753d，以及已加载环境变量的 CANN。
规格为 B1H1、BF16、D128、非因果、page128、正的 128 对齐 KV 长度；支持 Q 尾块。
`lengths=[KV]`，即完整有效缓存。分页表索引物理 K/V 池。

## 运行

```bash
python example.py
python example.py --q 32768 --kv 8192
```

示例仅编译该算子并对照分块 FP32 CPU 参考验证精度，不含性能实验或多版本对照。
示例显式使用 `plan_policy="cost"`，与该提交的 80 规格测试一致。
原始 `build_kernel` 的函数默认仍为 `balanced`，两个选项均来自原提交，并未加入新策略。

## 调用

```python
from operators.mha.kernels.mha_paged_decode import build_kernel

program = build_kernel(Q_length, KV_length, k.shape[0], plan_policy="cost")
program.validate_inputs(q, k, v, block_table, lengths)
out = program(q, k, v, block_table, lengths)
```

`q` 为 `[1,Q,1,128]`，K/V 为 `[物理池行数,1,128]`，均为连续 NPU BF16；
`block_table` 为 `[1,KV/128]`，`lengths` 为 `[1]`，均为连续 NPU int32。
编译后的 callable 会缓存 host 任务表；无 split 时直接输出，无 partial/LSE 工作区和 merge。
有 split 时使用原提交的 normalized Partial/LSE 和 SIMT merge。

本地整理验证涵盖来源字节校验、80 个规格的 host 任务表合法性和 Python 语法。
没有在本次整理中重新运行 NPU 精度或性能；运行示例可在目标环境验证。

## 80 规格原生 AscendC 对照

```bash
git pull --ff-only
bash run_grid.sh
cat reports/mha_50fca6d_grid_paste.txt
```

仅测试本分支冻结算子（原 `cost` 调度）与安装环境的原生
`torch_npu.npu_fused_infer_attention_score_v2`，不导入其他算子版本。
Q 为 64/128/256/512/1024/2048/4096/8192/16384/32768，
KV 为 512/1024/2048/4096/8192/16384/24576/32768，共 80 项。
固定 B1H1、BF16、D128、非因果、page128，分页输入与上一轮主线 80 项一致。

每个规格使用独立受监督进程、编译一次 local/必要的 merge，两个种子 7/19
对照分块 FP32 CPU 参考检查精度（rtol=0.01、atol=0.004），再复用编译结果做三轮计时。
每轮双方各预热 100 次、预先初始化 Event，然后四组交替测量；每次 Event
采样前，被测实现立即预热十次。每轮双方各占先后位置两次。
AscendC 内部缓存及派发保持安装环境原生行为。

完整调用的 warm-cache Event 耗时包括 Q 准备、常规输出/工作区分配和全部 kernel；
不含编译及一次性的 host 任务表构建/上传。加速比为 AscendC/TL，使用全部十二个
样本的各自中位数；同时保留三轮比值与原始样本，不剔除慢样本。
所有源码保持与 `SOURCE_MANIFEST.json` 一致，否则拒绝测试。

终端显示 1/80 到 80/80 及每组采样进度。每个规格完成即落盘，后续失败不会删除
已完成结果；全部结果存于打印出的 `reports/grid.*` 私有目录。提供原始 JSON、
日志、生成源码与哈希、汇总 `summary.json`/`summary.csv` 和可粘贴矩阵。
`resource_guard.py` 仅提供主机资源监督，默认进程树 RSS 4 GiB、超时 1800 秒、
CPU 线程上限 4，不采集 NPU profiling，也不能保证避免驱动故障。
此次仅新增基准工具，未修改冻结的算子与调度代码。实际 NPU 性能需上机运行确认。
