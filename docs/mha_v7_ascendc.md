# v7 与官方 AscendC 分页 attention 加速比

运行：`python fgw.py mha run run_mha_v7_ascendc`。
先跑 Q1/128/4096 的接口、精度和短计时冒烟，成功后执行完整基准。
不安装新软件，不修改CANN，不编译当前ops-transformer源码。
基线是机器现有CANN中的FusedInferAttentionScore，通过torch_npu调用；
记录API schema、版本、CANN环境目录及v7源码哈希，不将已安装二进制声称为当前master源码。

## 接口

默认 `--fia-api auto`：优先选择可调用的 `npu_fused_infer_attention_score_v2`，
只有Python未暴露v2时选择v1。出现参数/运行/精度错误不自动切换接口，也不改变形状或dtype。
可显式 `--fia-api v1`、`v2` 或 `both`，后一选项分别报告，不能混为同一个基线。
参考官方op-plugin的schema：
https://github.com/Ascend/op-plugin/blob/master/op_plugin/config/op_plugin_functions.yaml
v2在950的官方分发路径调用aclnnFusedInferAttentionScoreV5；实际安装版本以机器环境为准：
https://github.com/Ascend/op-plugin/blob/master/op_plugin/ops/opapi/FusedInferAttentionScoreV2KernelNpuOpApi.cpp

## 相同输入和语义

B2/H8、B1/H1各一组；Q=1/4/16/32/64/127/128/129/256/512/1024/2048/4096/8192/16384/32768。
KV2048、BF16输入输出、D128、page128、非因果、scale=1/sqrt(128)、seed7。
两者使用同一个Q、同一个随机物理页顺序、同一个int32页表以及相同的每请求KV长度。
v7 Q/O是BSHD，FIA对应BSND；v7的K/V `[pool_rows,H,D]` 仅view为
`[num_pages,128,H*D]`（BnBsH），不gather、不转置、不额外复制缓存。
FIA sparse_mode=0、不传mask、不量化、不输出LSE、inner_precise=0。
v7保持当前manual/stages2/Q驻留、自动split策略。

## 检查和计时

使用独立FP32 CPU分块reference（参考输出按BF16舍入），统一rtol0.01/atol0.004，
还检查输出shape和dtype，并记录最大绝对误差。
形状和接口错误保留在报告中，失败项不参与加速比。

完整基准每项各预热20次；每轮各51个单次调用Event样本，轮换/反转候选顺序，
2轮反转初始顺序。warm-cache，不扣计时开销；包含原生调用输出分配和stream可见调度间隙。
v7包括Q布局转换和merge，FIA包括该原生调用发射的所有kernel；
不包含编译、H2D、CPU参考、元数据列表构造和K/V的零拷贝view创建。
这是原生接口完整调用的Event延迟，不是纯Cube执行时间，也不是CPU wall time。

`speedup_v7_over_ascendc = AscendC中位耗时 / v7中位耗时`。
大于1表示v7更快，小于1表示AscendC更快。每个形状、每轮、每个API分别配对。
MFU仅计 `4*B*H*Q*KV*128`，峰值沿用28-AIC BF16 Cube 378TFLOPS。

## 报告

reports/mha/latest 下生成：

- mha_v7_ascendc_speedup_summary.json：精度、耗时、MFU、配对加速比及环境信息。
- mha_v7_ascendc_speedup.log：进度及错误。
- mha_v7_ascendc_speedup.json：额外包含原始Event样本，可保留本地。

冒烟采用相同命名前缀但将speedup替换为smoke；冒烟失败则不启动完整基准。
优先上传summary.json和log。手动指定接口示例：
`python fgw.py mha compare-ascendc --fia-api v1 --q-lengths 1 128 4096 --rounds 1 --repeat 11 --warmup 5 --output mha_v7_ascendc_v1_smoke.json`。
本地没有NPU，静态检查不代表实机通过。
