# MHA Ascend — 50fca6d clean

从 FGW 提交 `50fca6d93466f6c8eed2388b58366c3aae819299` 提取的独立精简版本。
这是历史版本，不含后续 stream、raw/SIMD merge、寄存器 softmax 等优化。

只保留：

* `operators/mha/kernels/mha_paged_decode.py`：该提交的静态 MHA 算子，仅重命名文件。
* `operators/mha/static_schedule.py`：该提交的 host 调度和冻结成本参数。
* `example.py`：最小 NPU 精度验证示例。
* 来源清单、必要的包文件和 TileLang 许可证。

算子与调度文件内容逐字保留；`SOURCE_MANIFEST.json` 记录原路径和 SHA256。
独立分支只有一次初始提交，不包含旧仓库提交历史、历史测试、报告和实验脚本。

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
