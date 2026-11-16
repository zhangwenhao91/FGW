# 分页MHA

公开入口为 `operators.mha.current`，当前指向v7。`kernels/`保留v0～v7，历史版本用于复现，不作为多套长短Q分流。

Q/O为[B,Sq,H,D]，K/V物理池为[N,H,D]，页表[B,W]和长度[B]均int32。输入连续、同设备，页表和长度值由调用方保证有效。只读缓存，不追加KV。

当前Cube路径：BF16、D128、page_size为128的倍数、正数且128对齐的缓存长度。非因果支持Q尾块，单Q因果支持；多Q因果和其他既有规格回退v2。不要用v7混合长度测试来宣称任意KV尾块受支持。

默认Q驻留L0A、流水深度2、K/V L1和Vector UB两版本。可显式 `q_resident=False` 对照；CLI使用 `--no-q-resident`。split=1无需Partial/LSE及merge，多split按行重缩放后合并。

相邻目录：`tests/mha/`精度、`benchmarks/mha/`计时、`scripts/mha/`批量实验、`reports/mha/`结果。新报告统一通过根目录fgw.py入口或批量脚本生成。

受限规格v7 Cube默认采用参考PR #3896的连续Q tile分配思路：每个program处理一段连续任务，余数均衡，保持与原跨步分配相同的每program任务数量。长短Q统一使用，persistent上限仍为28，split冻结模型、manual流水和Fast Math设置保持原样。原PR在显式block数量下要求整除；这里保留已经验证的非整除边界处理。

`build_kernel(..., task_assignment=None)`按路由选择默认方式：受限Cube为`contiguous`，通用Cube和Vector保持原有任务组织。可显式`task_assignment="strided"`复现旧受限Cube分配，或`"contiguous"`指定新方式；后者仅用于受限Cube。对照脚本`run_mha_q_assignment`显式比较两种分配，不随默认值变化。

采用依据：B1H1/BF16/D128/page128/非因果，56项完整矩阵及9项边界测试、种子7/19精度通过。连续分配相对跨步的几何平均加速比分别为0.9946和0.9934；此次默认切换按用户偏好统一参考PR组织方式，不宣称性能提升。其他B/H及因果组合尚未做本次分配方式的NPU回归。
