# 实机AscendC kernel与tiling证据采集

运行 `python fgw.py mha run run_mha_kernel_inspect`。
固定B2/H8/Q4096/KV2048、BF16、D128、page128、非因果及相同seed7分页数据。
不改kernel、模型系数或全局机器环境，不执行性能调优候选。

## 三个独立子进程

1. FIA DEBUG：只在此子进程设置ASCEND_GLOBAL_LOG_LEVEL=0和ASCEND_SLOG_PRINT_TO_STDOUT=1，
   让首次FIA v2执行打印安装版本的host tiling/dispatch日志。调用结果与CPU参考检查。
2. FIA profiler：正常ERROR日志级别，预热后只采集一次FIA调用，使用torch_npu原生profiler，
   record_shapes和Level1，输出kernel_details/op_summary等CSV及trace。
   输入准备和精度检查在profiler范围之外，便于将kernel记录对应到此形状。
3. v7 export：使用当前默认策略和相同形状，导出设备源码和实际pass IR（AutoSchedule/InsertSync）。

DEBUG环境仅对子进程生效，不污染调用终端；采集时日志和profiler都会影响性能，结果不是延迟基准。
每个阶段失败都会保存错误，其他阶段继续，最后给出完整/部分成功状态。
原始日志及trace放在唯一时间戳目录，脚本不自动删除或覆盖原始目录。

## 证据解释

summary中runtime_kernel_rows保留profiling CSV原始列，通常可含kernel名称、任务类型、Block Dim、
输入shape等，具体字段以当前安装版本为准。CSV不可用则尝试trace的kernel事件作为后备。
任何字段未返回都保持unknown/not_observed，不根据公开仓最新源码补值。

runtime_numeric_fields只是相关运行日志中的原始键和值，不自动当成已解码tiling结构。
特别是blockSize是分页大小，s1Size/s2Size通常为输入长度，不能误当Q/KV compute tile。
必须结合真实kernel名/tiling key/安装版本再解释其tile、分核及缓冲策略。
不使用最新master的数据结构强行解码已安装weekly包的二进制tiling。

## 上传提取结果

reports/mha/latest下优先上传以下小文件，原始trace和全量DEBUG日志保留本地：

- mha_kernel_inspect_summary.json
- mha_kernel_inspect_runtime_tiling.log
- mha_kernel_inspect_kernel_rows.json
- mha_kernel_inspect_v7_device.cpp
- mha_kernel_inspect_v7_AutoSchedule.py（生成时）
- mha_kernel_inspect_v7_InsertSync.py（生成时）
- mha_kernel_inspect.log

如果tiling字段没有输出，summary会明确显示not_observed；下一步再根据实际kernel名选择匹配的采集/解析入口。

## 官方依据

- DEBUG级别： https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/910/maintenref/envvar/envref_07_0122.html
- 标准输出日志： https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/910/maintenref/envvar/envref_07_0121.html
- 官方FIA host源码的PrintAllTilingData/OP_LOGD： https://gitcode.com/cann/ops-transformer/blob/master/attention/fused_infer_attention_score/op_host/arch35/fia_tiling_nonquant_gqa.cpp

公开源码只用于确定采集入口，不当作本机已经采用这些tiling值的证据。
