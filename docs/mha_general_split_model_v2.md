# 通用 Cube 模型 v2：最长分片与长度不均

本次只修改主机端分片选择。通用Cube的QK/PV、局部输出和merge计算体不变，
长短Q仍共用同一套kernel逻辑。默认fill28不变；原通用模型v1和专用v7
D128冻结系数均保留，v2仅显式启用。

## 扩展校准数据与隔离

用户已提供上一轮独立集的全部262组split耗时，记录于
`reports/mha/calibration/general_split_holdout_20261113.txt`，并与原212组
合并为`general_split_expanded_20261113.json`：45种形状、90项dtype配置、
474组合法split。来源为文本中0.001μs精度的6轮中位数，未补造原始样本。
数据来自不同测量会话，模型需容忍调用延迟偏移。

上一轮holdout现在明确转为v2训练数据，不能再称为v2独立验证。
原v1 JSON保持原样，另存冻结候选`general_split_cost_model_v2.json`。

## 新特征与拟合

保留v1八项特征，增加：

```text
N_b = ceil(实际KV长度[b]/32)
longest = ceil(max_b(N_b)/split)
excess = longest - mean_b(N_b)/split
新增特征 = excess * 补齐D/64，以及 excess
```

原平均工作量描述总量，新项近似描述最长分片相对平均分片的尾部工作量，
避免把长短请求混合时的关键路径低估。它不是硬件调度器的精确模拟。

拟合仍约束所有系数非负；v2增加同一形状各split之间的相对耗时差损失，
固定权重0.25，以更直接拟合候选排序，降低不同会话常数偏移的影响。
选择时在预测最优的`max(2%, 1μs)`范围内选最小split。1μs是候选的保守
决策超参数，不是计时置信区间或硬件常数。经验单/多kernel延迟下限继续
来自训练集。每条规则适用于所有规格，不按形状名称查表。

| 离线开发检查 | 距最佳≤2% | 最大差距 | 几何平均差距 |
| --- | ---: | ---: | ---: |
| 扩展校准集 | 88/90 | 2.974% | 0.157% |
| 完整形状留一 | 88/90 | 2.974% | 0.157% |

留一同时移除两个dtype及全部split。交叉验证参与模型开发，不作为独立
性能结论；剩余两项为旧`hold_causal_q127_d128`，v2选8而扫描最佳为1。
上一轮主要误选（不均长度Q1、混合因果空行等）已在离线数据上修正，仍需
新实机形状验证。所有系数与超参数在新测试前冻结。

## API与版本对照

```python
out = mha_paged_decode(q, k, v, table, lengths, page_size,
                        cube_mode="general", split_policy="cost_model",
                        general_model_version=2)
```

省略版本时仍使用v1；不指定`cost_model`仍是fill28。直接通用builder使用
`model_version=2, split_policy="cost_model", cache_lengths=tuple(...)`。
显式num_splits依旧优先。长度提示只选split，设备仍读取动态Lengths。
公共API复用已有的主机长度校验，不新增一次D2H。

## 新一轮NPU验证

```bash
git pull --ff-only
python fgw.py mha run run_mha_cube_model_v2_validate
```

依次执行：

1. 120项公共API精度检查（1个seed，含两个dtype、因果、尾块、空缓存和
   无效位置NaN）；kernel数学未变，随后所有扫描候选还会逐项检查精度。
2. 扩展校准集90项配置、474组split的同场回放。
3. **另一批全新24种形状×两个dtype**，48项配置、282组split，未进入任何拟合。

新集增加B4/B5、H3/H5/H6，Q1/2/3/5/15/16/17/19/31/32/33/35/63/65/
125/126/129/253/257/511/1023，覆盖长度高度不均、部分请求零缓存、不同
分页和因果空行，并保留均匀长度参照。所有输入张量静态规格都与45种训练
形状不同；新结果不会自动回写系数。

每项扫描所有容量允许的split，分别记录v2选择、v1选择、旧fill28和经验
最佳，6轮、每轮20次预热和31组轮换顺序的完整调用Event计时。相同选择
复用同一测量项，不重复计时制造虚假的策略差异。设备为已校准28 Cube核。

优先传回：

- `mha_cube_model_v2_replay_summary.json`、对应log。
- `mha_cube_model_v2_holdout_summary.json`、对应log。
- 公共API精度报告`mha_cube_model_v2_accuracy.json`。

传不上去时，无论成功或报错退出都可以直接粘贴：

```bash
cat reports/mha/latest/mha_cube_model_v2_status.txt
```

文本会同时给出v2相对旧策略、相对v1、相对扫描最佳的差距；只包含本次
运行结果，排除上次残留文件。本轮不重新测AscendC，默认切换仍待独立结果。

## 复现校准

```bash
python -m tools.mha.fit_general_split_model --version 2 \
  --input reports/mha/calibration/general_split_expanded_20261113.json \
  --output operators/mha/general_split_cost_model_v2.json
```

训练仅在本地CPU完成；NPU测试脚本不拟合。v1仍可用原命令复现，v2不改
它的特征和参数含义。
