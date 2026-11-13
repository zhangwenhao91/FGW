# 通用 Cube 独立成本模型候选

用同一通用Cube的split扫描数据校准模型，保持同一QK/PV/merge计算体。
模型文件与专用v7 D128冻结模型完全独立。当前默认仍是fill28；候选仅在
显式`split_policy="cost_model"`时使用，待独立实机验证后再决定是否替换。

## 数据与离线结果

数据来自用户传回的`mha_cube_split_scan_paste.txt`，保存在
`reports/mha/calibration/general_split_scan_20261113.json`：21种形状、42项
FP16/BF16配置、212组合法split，每组为6轮中位数的中位数，文本精度0.001μs。
未补造原始样本、每轮分布或置信区间；独立组件计时不可加，因此不参与拟合。

拟合代码为`tools/mha/fit_general_split_model.py`，模型为
`operators/mha/general_split_cost_model.json`，使用非负最小二乘的坐标下降，
最小化相对耗时平方误差。模型预测只使用Python标准库，不在NPU机上重新拟合。

| 验证方式 | 距经验最佳≤2% | 最大耗时差距 | 几何平均差距 |
| --- | ---: | ---: | ---: |
| 校准数据回放 | 42/42 | 0.699% | 0.0322% |
| 完整形状留一 | 40/42 | 2.402% | 0.1416% |

留一验证会同时移除该形状的两个dtype和全部split，再拟合权重与延迟下限；
不是把同一形状的另一个dtype泄漏进训练。两项未落在2%内的均为D64/Q1/
KV2048：选32而实测16更快，差距约2.24～2.40%。以上仍是离线结果，
不是独立NPU测试，更不是已经验证的新默认。

## 统一预测规则

令`T=B*H*ceil(Q/32)`，`W=ceil(T*S/28)`，`K`为各请求实际KV tile数
`ceil(length/32)`的平均值，`d=max(32,next_power_of_2(D))/64`。
merge行分组沿用kernel的`min(8,next_power_of_2(Q))`，总组数记为`G`。
八个非负特征是：

```text
1, W*K/S*d, W*K/S, W*d, W,
I(S>1), I(S>1)*G*S*d/56, I(S>1)*T*S*d/28
```

相同规则覆盖所有Q、D和分页，没有按样例名称查表。分片数超过页表容量时
排除。先预测各合法split耗时，再在预测最小耗时的2%以内选择最小split。
单kernel/多kernel预测分别使用校准数据的最小观测耗时22.921/30.715μs
作为经验下限；它们仅适用于该warm-cache Event环境，不是硬件物理常数。
独立验证会检验这些下限与系数能否泛化。

## API和长度提示

```python
out = mha_paged_decode(q, k, v, table, lengths, page_size,
                        cube_mode="general", split_policy="cost_model")
```

公共接口复用已有的长度校验所取得的主机长度，不额外增加一次D2H同步。
直接`build_kernel`需要`cache_lengths=tuple(...)`作为选择提示；该提示仅
用于选split，设备计算仍读取传入的动态Lengths tensor，不把长度编译为常量。
显式`num_splits`优先于模型。模型仅支持已校准的28 Cube核配置。
性能脚本仍排除主机元数据校验/模型选择时间，比较完整预构建设备callable。

## 独立NPU验证

```bash
git pull --ff-only
python fgw.py mha run run_mha_cube_model_validate
```

依次执行：

1. 使用公共API模型选择，360项边界集+540项原扩展精度，原阈值不变。
2. 校准集同场回放，42项配置、212组split，记录模型/旧策略/经验最佳耗时。
3. 独立holdout，24种新形状×两个dtype，共48项、262组split。

holdout改变B/H（含B3/B4、H3/H8）、Q2/7/31/32/33/64/127/255/256/513/
1024、D32/48/64/80/96/128/192/256、page16/48/64/128/256及不同缓存长度。
包含容量明显大于有效长度、请求间长度不均、零缓存和因果全遮罩行；
这些形状均未进入拟合。系数在实测前冻结，脚本不会根据新结果再训练。

每个split先检查精度，随后6轮、每轮20次预热、31组轮换顺序的完整调用
Event计时，结果逐项保存。本轮不重新测AscendC，先确认选择策略质量。
实机报告会给出模型距经验最佳差距、相对旧策略加速比及近优覆盖率。

优先上传：

- `reports/mha/latest/mha_cube_model_replay_summary.json`
- `reports/mha/latest/mha_cube_model_holdout_summary.json`
- 对应`.log`，以及精度报告。

Git上传失败时，直接粘贴：

```bash
cat reports/mha/latest/mha_cube_model_status.txt
```

该文本在成功或失败退出时自动导出，只包含本次运行的报告，避免混入上次
残留的成功结果；包含精度计数、异常、分组及每项模型选择/耗时。

## 复现拟合（本地CPU）

```bash
python -m tools.mha.fit_general_split_model \
  --input reports/mha/calibration/general_split_scan_20261113.json \
  --output operators/mha/general_split_cost_model.json
```

此命令会重写候选模型，供开发复现使用；NPU验证脚本不调用它。源数据哈希、
非负系数、经验下限、特征顺序和两种离线验证结果均保存在模型JSON中。
