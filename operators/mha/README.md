# 分页MHA

公开入口为 `operators.mha.current`，当前指向v7。`kernels/`保留v0～v7，历史版本用于复现，不作为多套长短Q分流。

Q/O为[B,Sq,H,D]，K/V物理池为[N,H,D]，页表[B,W]和长度[B]均int32。输入连续、同设备，页表和长度值由调用方保证有效。只读缓存，不追加KV。

当前Cube路径：BF16、D128、page_size为128的倍数、正数且128对齐的缓存长度。非因果支持Q尾块，单Q因果支持；多Q因果和其他既有规格回退v2。不要用v7混合长度测试来宣称任意KV尾块受支持。

默认Q驻留L0A、流水深度2、K/V L1和Vector UB两版本。可显式 `q_resident=False` 对照；CLI使用 `--no-q-resident`。split=1无需Partial/LSE及merge，多split按行重缩放后合并。

相邻目录：`tests/mha/`精度、`benchmarks/mha/`计时、`scripts/mha/`批量实验、`reports/mha/`结果。新报告统一通过根目录fgw.py入口或批量脚本生成。
