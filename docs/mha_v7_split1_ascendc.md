# 自动split、强制split1、AscendC三方对照

运行 `python fgw.py mha run run_mha_v7_split1_ascendc`。
默认split策略保持不变，kernel计算没有修改。
复用已跑通的torch_npu FIA v2调用和同一分页输入，在每个进程中交替比较三个候选。

## 范围

- KV2048：完整16项Q=1/4/16/32/64/127/128/129/256/512/1024/2048/4096/8192/16384/32768。
- KV512/8192/32768：Q=1/16/128/129/512/1024/4096/8192，覆盖短Q、尾块、长Q。
- 每项均B2/H8、B1/H1；共80个形状、240个候选精度检查、4轮960条性能记录。
- 统一BF16输入输出、D128、page128、非因果、seed7、相同随机物理页/块表/有效长度。
- v7自动及split1都为manual/stages2/Q驻留，其余参数相同。
- AscendC显式固定已验证的FIA v2、BSND及BnBsH view，inner_precise=0，不做稠密gather。

KV8192、Q1/129/4096先做两组B/H冒烟，失败不进入完整集。
完整集每形状4轮，各候选预热20次、各51个单次Event，轮换/反转三个候选顺序。
编译和CPU独立精度参考（FP32计算，BF16舍入）都在计时外。
内含完整调用的Q转换、分配、compute和merge；不含H2D和元数据准备，warm-cache。
精度失败保存错误，不生成该候选加速比；非致命的完整集失败会继续其他KV，最后以非零退出。

## 结果解释

`comparisons` 分别保存FIA/自动v7、FIA/强制split1，两者>1均表示相应v7更快。
`split_comparisons` 保存自动v7/split1，>1表示强制split1更快，另给节省微秒及自动split数。
每个结果包括geometry、执行路径、精度误差、延迟和MFU。
不能把split对照的全部差值当成merge耗时，因为计算并行度、临时结果写回和launch也发生改变。

## 上传

reports/mha/latest下优先上传：

- mha_v7_split1_ascendc_kv512_summary.json 和同前缀.log
- mha_v7_split1_ascendc_kv2048_summary.json 和同前缀.log
- mha_v7_split1_ascendc_kv8192_summary.json 和同前缀.log
- mha_v7_split1_ascendc_kv32768_summary.json 和同前缀.log

完整原始样本.json保留本地。冒烟失败上传mha_v7_split1_ascendc_smoke_summary.json和.log。
已有完整KV的结果可保留，单独重跑指定KV，例如：
`python fgw.py mha compare-ascendc --split-comparison --fia-api v2 --kv 32768 --q-lengths 1 16 128 129 512 1024 4096 8192 --rounds 4 --warmup 20 --repeat 51 --output mha_v7_split1_ascendc_kv32768.json`。
