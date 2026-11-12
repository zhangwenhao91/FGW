# v7 与 AscendC 的同口径流水计数器对照

运行 `python fgw.py mha run run_mha_v7_ascendc_msprof`。
固定B2/H8/Q4096/KV2048、BF16、D128、page128、非因果、seed7。
当前冻结模型默认，预计split1；若不是direct路径则报错而非替换规格。

同一份Q/K/V和随机物理页、块表，三个候选：

- v7_full：原生compiled callable完整调用，包括Q布局转换和direct输出。
- v7_direct：预先转换Q，仅调用同一个编译好的direct kernel。
- ascendc_full：已验证torch_npu FIA v2，BSND及BnBsH缓存view，不gather。

全部先通过独立CPU FP32分块attention参考精度检查（参考BF16舍入，rtol0.01/atol0.004）。
保持FIA inner_precise0/sparse_mode0，不传mask，不量化、不输出LSE。kernel和模型均不修改。

## 两种计时

Event：warm-cache，每轮预热20次、每候选51个单次样本，交替轮换/反转执行顺序，3轮。
记录每轮中位耗时、有效QK/PV MFU，用于确认此前Event差距能否复现。

msprof_detail：直接使用同一个tilelang.profiler.do_bench，三个候选相同参数，
每次调用前由上游插入256MiB清缓存；每会话预热20次、正式30次，3轮轮换会话顺序。
保存全部KernelProfile字段：dur_ns、AIC/AIV total cycles、mad、vec、MTE、fixpipe、scalar等。
保存实际安装的tilelang.profiler.msprof源码、SHA256及do_bench签名，便于核对聚合方式。

上游dur_ns是每次调用的profiler记录聚合，并非Event elapsed median或CPU wall time。
多kernel完整调用的profile是上游聚合值，不等同于单个计算kernel的计数器。
比例使用各自AIC/AIV周期分母，可重叠，不能相加为独占耗时分解，也不能当成MFU。
full和direct在独立会话测量，差值不能精确视作Q转换耗时。
FIA原生调用如有辅助kernel，这些kernel都包含在完整调用聚合中。
本轮不改写解析器，也不将Event与msprof绝对时间混用。

## 报告

reports/mha/latest下上传：

- mha_v7_ascendc_msprof_summary.json
- mha_v7_ascendc_msprof.log
- mha_v7_ascendc_msprof_msprof_source.py（实际解析源码，建议一起上传）

完整mha_v7_ascendc_msprof.json额外包含Event原始样本，可留本地。
错误同样保存报告；没有可用计数器时不会用零值伪装为有效结果。
