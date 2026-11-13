# 同一通用 Cube 计算体的 split-KV 并行

在既有通用 Cube 内加入 KV 分片。FP16/BF16、D16～256、16倍数分页、
Q/KV 尾块及右对齐因果都保留，QK/PV 计算只维护一个 `compute_tile` macro。
长短 Q 不切换不同数学实现或回退 Vector。

## 分片与合并

一个任务对应 (B,H,32行Q)。静态任务数为 `B*H*ceil(Q/32)`。
当任务数不足28时，按缓存容量逐次增大 split（1/2/4/8/16/32），直到任务数
乘 split 至少28、分片已达32，或容量不够进一步分片。这个首版启发式只补
并行度，不使用专用 D128 的冻结系数，也未宣称对所有形状是最佳选择。

设备读取每个请求的实际长度，计算 `N=ceil(length/32)` 个有效 KV tile。
第 s 份负责 `[floor(N*s/S), floor(N*(s+1)/S))`，覆盖恰好一次，不按物理
页号切分。每个 token 仍经页表寻址，支持 page16/page48 的跨页 tile。
容量大、实际长度短时，允许空分片。

例如满缓存时，B2/H2/Q1/KV2048 自动选8份（32个计算program），
B1/H1/Q1/KV8192 自动选32份；B2/H2/Q512 已有64个query tile任务，
自动选1份。QK/PV代码相同，仅工作区间与输出方式不同。

每份执行同一 QK→softmax→PV 循环，FP32 累加；多份时输出 FP32 局部
归一化结果与 log-sum-exp，另一个 merge kernel 按行用稳定权重合并，
最后转换为原输入类型。空分片记为零权重；全部为空或因果全遮罩时输出
精确零。Partial/LSE 是每次调用独有的临时张量，两个 kernel 使用当前流顺序。

split=1 复用相同 macro，直接归一化/转换/写最终输出，不分配 Partial/LSE、
不启动 merge；多份只增加分片索引和输出/合并步骤。不是另一套 attention。
merge 使用独立 Vector kernel 属于分片 softmax 的正常收尾；两次矩阵乘仍在 Cube。

## API 与兼容

```python
from operators.mha.current import mha_paged_decode

# 新通用候选默认选择 split；已验证的专用 v7 默认路径保持原状。
out = mha_paged_decode(q, k, v, table, lengths, page_size,
                       cube_mode="general")
out1 = mha_paged_decode(q, k, v, table, lengths, page_size,
                        cube_mode="general", num_splits=1)
```

通用 `build_kernel(..., num_splits=None)` 为自动，显式支持1/2/4/8/16/32，
不能超过页表容量可容纳的 KV tile 数。`num_splits`、`compute_grid`、
`execution_mode` 等实际配置随 callable 暴露。

历史 Cube/Vector 性能脚本显式固定1份；原 Cube/AscendC 脚本默认也保持1份。
新增 `--cube-splits auto --compare-split1` 才同时测自动与1份控制组。
对照使用本次共享计算体的 split=1，未假设其生成二进制与重构前完全相同。

## NPU 测试

```bash
git pull --ff-only
python fgw.py mha run run_mha_cube_splitkv
```

依次执行：

1. 48项自动分片冒烟。
2. 6个显式分片数，每个96项精度检查，共576项。包括仅1个有效tile却
   有32份分片、空缓存、乱序/重复/共享页、page16/page48、因果、Q尾块。
3. 自动策略360项边界与540项原扩展回归，阈值保持不变。
4. 原50项MFU集，比较自动Cube、同体split=1与原生AscendC，6轮交替计时。
   page16/page48的4项仍仅比较两个Cube策略；AscendC46项。

优先上传 `reports/mha/latest/mha_cube_splitkv_mfu_summary.json` 和
`mha_cube_splitkv_mfu.log`。精度报告均为 `mha_cube_splitkv_*accuracy.json`，
另有 `mha_cube_splitkv_smoke.json` 与 `mha_cube_splitkv_boundaries.json`。
如编译/执行失败，脚本停止并保留对应JSON和日志；不用继续后续步骤。
本地没有 NPU，语法/策略/分片覆盖及数学检查不代替实机验证。
本地已通过4356组分片覆盖性质检查，以及1476项NumPy密集参考数学检查；
包括全部6种分片数，最差误差/允许误差约0.796。它们不是NPU精度结果。

MFU与AscendC基线继续使用[同口径](mha_cube_ascendc_mfu.md)。另外报告
`speedup_split_over_split1=split1耗时/自动分片耗时`、每轮胜出数与实际分片数。
长Q任务足够时自动选1份，作为同计算体控制组，不要求其因分片而提速。
新策略主要针对并行度不足；搬运、NZ转换和流水交叠仍是后续优化项。
