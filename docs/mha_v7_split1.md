# 自动 split 与强制 split=1 对照

运行 `python fgw.py mha run run_mha_v7_split1`。
先跑 Q1/129/4096 冒烟（B2/H8和B1/H1），全部精度通过后运行完整集。

## 参数及范围

当前 v7 build_kernel/mha_paged_decode 新增可选 `num_splits`：

- None：原有自动选择，默认行为不变。
- 1：强制走已有 single-split direct kernel，内部完成归一化、BF16转换和输出。

该覆盖只用于v7 Cube范围，回退规格拒绝显式覆盖，后续split扫描扩展支持2/4/8/16，详见mha_v7_split_scan.md。
没有新增按长短Q分派的版本，也没有改写Cube/softmax/merge内部计算。
返回callable暴露num_splits、query_tile_tasks和execution_mode，报告直接读取实际配置。

完整基准B2/H8及B1/H1，Q=1/4/16/32/64/127/128/129/256/512/1024/2048/4096/8192/16384/32768，
KV2048、BF16输入输出、D128、page128、非因果、seed7；相同Q/K/V、随机物理页、块表及长度。
两个候选都用manual、stages2、默认缓冲配置和Q驻留，只改变num_splits。
每个候选先独立CPU FP32分块参考精度检查（参考输出BF16舍入，rtol0.01/atol0.004），
检查shape/dtype并记录误差。失败项保存，不计算配对速度比。

## 计时及解释

warm-cache Event完整调用，每轮各预热20次、各51个单次样本，交替旋转/反转候选顺序，4轮反转初始顺序。
包括Q转换、分配、计算和原本需要的merge，不包括编译/H2D/参考/元数据准备。

速度比=自动split耗时/强制split1耗时，>1表示强制split1更快。
保存的latency_saved_us=自动耗时-强制耗时，正值表示节省。

差值是整体策略代价：减少了计算并行度，同时移除了Partial/LSE写回、merge和额外launch，
不能将全部差值解释为独立merge耗时。
原本自动split=1的形状作为相同算法控制组，帮助观察计时波动。
MFU沿用有效QK/PV FLOPs及378TFLOPS峰值。

## 结果

reports/mha/latest：

- mha_v7_split1_paired_summary.json：实际split数/执行模式、误差、耗时、MFU、速度比和节省微秒。
- mha_v7_split1_paired.log：进度和错误。
- mha_v7_split1_paired.json：额外含全部原始样本，留本地供排查。

冒烟文件使用mha_v7_split1_smoke前缀。优先上传summary和log；默认策略需分析结果后再决定。
