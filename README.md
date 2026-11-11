# FGW

## 导出当前自动流水，准备手工调度对照

```bash
git pull --ff-only
python fgw.py mha export-schedule
```

在NPU开发环境编译并导出当前v7默认配置的AscendC设备源码、host源码、输入IR及各pass IR，覆盖Q4多split、Q129尾块、Q4096和Q32768直接输出。开启编译器DumpIR，其他计算配置不变，不运行性能测试。使用唯一输出路径避免直接复用旧dump。

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

当前版本：**v7，默认Q驻留L0A、两级流水**。单split直接输出，多split批量合并；长短Q使用同一套实现。Cube范围为BF16/D128、页与KV有效长度128对齐；支持Q尾块。多query因果和其他既有规格仍走原回退路径。

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
