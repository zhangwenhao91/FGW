# FGW

## 当前默认：常驻28上限

v7 Cube长短Q统一使用 `compute_grid=min(tasks*splits,28)`，冻结split模型保持不变。
显式 `persistent_programs=None` 可恢复原grid；非Cube保持既有回退。
[验证结果与对照入口](docs/mha_persistent28_default.md)。

## 常驻28短Q控制组复测

```bash
git pull --ff-only
python fgw.py mha run run_mha_persistent28_short
```

17个同grid控制组、2个长Q参照组，同进程12轮AB/BA交替计时，比较生成源码哈希。
历史默认切换前复测。[说明及上传文件](docs/mha_persistent28_short.md)。

## 28-program常驻调度实验

```bash
git pull --ff-only
python fgw.py mha run run_mha_persistent28
```

复用现有tile计算和冻结split模型，比较当前调度、28-program候选及AscendC。
先精度再Event及msprof；默认调度不变。[范围与报告](docs/mha_persistent28.md)。

## 实机kernel名称与tiling采集

```bash
git pull --ff-only
python fgw.py mha run run_mha_kernel_inspect
```

固定B2/H8/Q4096/KV2048，采集FIA运行日志、实际kernel记录及v7设备源码/流水IR。
[范围、限制与上传文件](docs/mha_kernel_inspect.md)。

## v7 / AscendC 同口径msprof对照

```bash
git pull --ff-only
python fgw.py mha run run_mha_v7_ascendc_msprof
```

固定B2/H8/Q4096/KV2048，对照v7完整调用、v7直写kernel和FIA v2完整调用。
先精度，后Event及msprof_detail各3轮，保存流水计数器和解析源码。
[计时口径及上传文件](docs/mha_v7_ascendc_msprof.md)。

## 当前默认：验证范围内启用冻结split模型

```bash
git pull --ff-only
python fgw.py mha run run_mha_model_default_ascendc
```

固定满缓存且配置在验证范围内时自动使用cost_model，其他情况保留legacy。
冻结系数不变；同输入重新对照当前v7、旧策略v7和AscendC FIA v2。
[范围、低层调用约定和上传文件](docs/mha_model_default.md)。

## split代价模型独立验证

```bash
git pull --ff-only
python fgw.py mha run run_mha_split_model_validation
```

模型已离线拟合并冻结，验证新B/H、KV4096/16384及原边界形状，逐项扫描全部split检查预测损失。
该实验当时默认legacy；当前默认见上文。[模型、离线误差与上传文件](docs/mha_split_cost_model.md)。

## v7 split扫描

```bash
git pull --ff-only
python fgw.py mha run run_mha_v7_split_scan
```

扫描split=1/2/4/8/16，覆盖四种KV长度、两组B/H和8种Q长度。
先精度后4轮交替计时，汇总最佳split及稳定性；当前自动策略不变。
[规格与报告说明](docs/mha_v7_split_scan.md)。

## 多KV长度下split1与AscendC三方对照

```bash
git pull --ff-only
python fgw.py mha run run_mha_v7_split1_ascendc
```

KV512/2048/8192/32768，交替测v7自动split、强制split1和官方FIA v2。
先冒烟及独立精度检查，再进行四轮Event计时；默认策略不变。
[形状、口径与上传文件](docs/mha_v7_split1_ascendc.md)。

## v7 强制 split=1 对照

```bash
git pull --ff-only
python fgw.py mha run run_mha_v7_split1
```

在相同长短Q输入上交替测自动split与强制split=1，先精度、后4轮Event计时。
报告实际split数、耗时、MFU和节省的延迟；当前默认split策略不变。
[基准说明](docs/mha_v7_split1.md)。

## v7 与官方 AscendC 加速比

```bash
git pull --ff-only
python fgw.py mha run run_mha_v7_ascendc
```

使用已安装CANN的FusedInferAttentionScore：先冒烟，再比较B2/H8和B1/H1的长短Q。
同一份分页数据、BF16输入输出，先精度后交替计时；加速比=AscendC耗时/v7耗时。
[配置与报告说明](docs/mha_v7_ascendc.md)。

## v7 与 PR3896 长短 Q MFU 对照

```bash
git pull --ff-only
python fgw.py mha run run_mha_v7_pr3896_mfu
```

重测v7 B2/H8长短Q，并另做B1/H1同输入对照（参考PR原FP32输出及BF16输出）。
参考源码固定版本原样保存，不增加batch/head适配kernel。计时范围和差异见
[基准说明](docs/mha_v7_pr3896_mfu.md)。优先上传 `reports/mha/latest/mha_v7_pr3896_mfu_summary.json` 与同前缀 `.log`。

## 当前默认调度：manual

v7 的长短 Q 统一使用 manual：启用跨迭代 offset、显式 Stage 和配套缓冲策略。
单 split 直写与多 split 合并使用同一调度选择，不按 Q 长度切换。
公开 API、精度、基准、msprof 和源码导出入口均采用这一默认值。
对照原调度可显式传 `--schedule-mode auto` 或 API 参数 `schedule_mode="auto"`。
历史实现 v0～v6 不受此默认值影响。

目标 NPU 已完成三种调度各两轮的30项非因果及9项单query因果检查，全部通过；
manual 长 Q MFU 约55%，Q4096/Q32768相对auto提速约1.74/1.71倍。
短 Q 的8轮交替计时精度通过，六个场景8/8轮提速；B1/Q1/KV512两项约持平，
manual 各轮中位耗时的中位数分别增加约0.19和0.02微秒。保留auto入口供复现。
计时口径及配置见 [审阅记录](docs/mha_auto_schedule_review.md)。

## 自动流水审阅后的Stage候选对照（历史实验）

已根据实际Q4096导出审阅自动阶段、同步和生成源码，见 [审阅记录](docs/mha_auto_schedule_review.md)。实验最初默认auto；对比auto-offset和manual编译选项，直接输出与多split采用同一选择规则。

```bash
git pull --ff-only
python fgw.py mha run run_mha_schedule_sweep
```

三组依次对比原自动调度、仅开启跨迭代offset、显式Stage加选择性缓冲策略。先做30项非因果和9项单query因果精度，成功后测Q4096/Q32768的msprof和完整调用MFU；正反顺序各一轮。编译/精度失败会保存报告并跳过性能测试，继续其他候选。

上传reports/mha/latest/里 `mha_schedule_*_run*` 的JSON和log。若需要查看候选实际调度，可运行 `python fgw.py mha export-schedule --cases q4096 --schedule-mode manual` 或auto-offset；建议只上传q4096设备源码和AutoSchedule/InsertSync三项，避免重复提交所有IR。目标NPU已完成精度及长短Q性能验证，结果见上文；本地仅进行静态及CLI检查。

## 导出原自动流水（复现实验）

```bash
git pull --ff-only
python fgw.py mha export-schedule --schedule-mode auto
```

在NPU开发环境编译并导出v7原auto配置的AscendC设备源码、host源码、输入IR及各pass IR，覆盖Q4多split、Q129尾块、Q4096和Q32768直接输出。开启编译器DumpIR，其他计算配置不变，不运行性能测试。使用唯一输出路径避免直接复用旧dump。

结束后将 `reports/mha/latest/mha_auto_schedule_*.zip` 上传。压缩包内含manifest、源码哈希及调度/同步/多缓冲pass文件。失败也生成包含错误的压缩包。先审阅真实AutoSchedule、同步和缓冲安排，再据此确定手工Stage候选；不直接照搬参考PR阶段编号。普通测试不传dump_ir_dir，因此不产生这些额外文件。

Ascend算子开发与验证工作区。当前包含分页MHA；新算子使用并列目录，不与MHA实验文件混放。

```text
operators/<op>/        算子实现与当前公开入口
tests/<op>/            精度回归
benchmarks/<op>/       性能、MFU及profiling
scripts/<op>/          批量实验入口
tools/env/             环境采集、最小Ascend kernel验证
docs/                  方法、开发约定及历史记录
reports/<op>/latest/   新生成的报告和日志
reports/<op>/archive/  已上传的历史实验结果
third_party/licenses/  复用代码的许可证
fgw.py                 统一命令入口
```

## 环境

在NPU机上激活原开发环境并加载CANN，保持原可见卡设置，再从仓库根目录运行：

```bash
git pull --ff-only
python fgw.py env collect
python fgw.py env smoke
```

结果写入 `reports/environment/latest/`。本仓库不自动更改机器的代理、证书或全局Git配置。

## 当前MHA

当前版本：**v7，默认manual调度、Q驻留L0A、两级流水**。单split直接输出，多split批量合并；长短Q使用同一套实现。Cube范围为BF16/D128、页与KV有效长度128对齐；支持Q尾块。多query因果和其他既有规格仍走原回退路径。

```python
from operators.mha.current import mha_paged_decode

o = mha_paged_decode(q, k_cache, v_cache, block_table, cache_seqlens,
                     page_size=128, is_causal=False)
```

精度、性能与profiling入口默认选择v7；历史版本用 `--impl v0`～`--impl v6` 显式指定。

```bash
# 当前Cube精度，涵盖Q尾块；KV保持整块
python fgw.py mha accuracy --suite cube --patterns full --dtypes bfloat16 --causal noncausal
# 当前标准性能基准
python fgw.py mha bench
# 长Q MFU
python fgw.py mha bench --suite large-q --dtypes bfloat16 --causal noncausal --reference-chunk-size 128 --warmup 20 --repeat 51
# 长Q设备侧计数器
python fgw.py mha msprof --cases query4096 query32768 --rounds 3 --repeat 30 --warmup 10
```

新报告自动写入 `reports/mha/latest/`。相对 `--output`、`--baseline-report` 也相对于这个目录；要引用归档报告请传绝对路径。每项命令的参数用 `python fgw.py mha bench --help` 等查看。Event热缓存与msprof缓存驱逐口径应分开比较。

批量脚本已更新路径，可从根目录运行：

```bash
python fgw.py mha run run_mha_v7
# 或直接运行
bash scripts/mha/run_mha_v7.sh
```

脚本日志同样写入 `reports/mha/latest/`。请上传该目录下本轮的JSON及对应log；msprof解析器源码可一并保留。latest是固定工作目录，重跑同名输出会覆盖，保留实验时请用不同输出前缀或归档。

## 已验证结果与资料

- 当前环境：Ascend950PR_9579，28 AIC；torch 2.13.0+cpu、torch_npu 2.13.0、TileLang 0.1.15+cpu.gita7cb753d。
- Q驻留的长Q最高实测MFU约32.09%，复测32.06%；短Q四场景同进程配对均8/8轮更快。只代表已记录的形状与计时条件。
- [MFU计算口径](docs/MFU.md)
- [当前MHA说明](operators/mha/README.md)
- [新算子约定](docs/DEVELOPMENT.md)
- [历史实验记录](docs/history/mha_experiments.md)（含旧路径命令，仅供追溯）

历史版本v0～v6保留在 `operators/mha/kernels/`，不会由当前入口按Q长度切换。已上传报告保留在archive，原根目录文件的移动关系记录于 `docs/history/file_moves.json`。
