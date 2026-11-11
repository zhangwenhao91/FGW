# FGW

## v3 Cube QK/PV实验路径

当前修订为避免GM→L1动态列补齐的编译限制，增加Vector预处理kernel：按页表生成每请求逻辑顺序的V，将有效长度外的行清零。Cube阶段只搬完整二维V块。完整计时包括V预处理、局部注意力、split合并三个kernel及其临时张量开销；该预处理会增加内存占用和数据搬运，仍属于Cube链路验证版本。

参考tile-ai/tilelang的Ascend FlashAttention core，在安装版本对应的固定提交a7cb753d上复用SIMD softmax/NZ打包宏（上游许可证保存在 `TILELANG_LICENSE`）。新增 `mha_paged_decode_ascend_v3.py`，BF16、D=128、页大小为128的倍数时使用Cube QK与PV、SIMD softmax、NZ权重打包、两级自动流水及split-KV合并。其他规格明确回退v2，每条结果记录 `execution_path`。

这是Cube链路验证版本：每个query复制为128行，以保持已有64行/AIV的SIMD打包布局，只有首行结果参与输出；大量冗余计算可能使它比v2慢，不能据此宣称优化成功。MFU分子只计模型有效计算量，不把重复行计为有用计算。分页尾块清零无效V；分数掩码、空split和空缓存均有处理，仍需NPU验证。

```bash
git pull --ff-only
bash run_mha_v3.sh
```

脚本先做含Cube路径的12项编译/精度冒烟检查，再跑540项回归、v2的20项重测、v3的20项比较。当前540项中仅36项真正走Cube，504项回退v2；20项性能用例中4项走Cube，其余16项回退v2。quick精度集不含Cube规格，因此不能用quick通过代表Cube通过。

请上传 `mha_cube_v3_smoke.json`、`mha_paged_accuracy_v3_report.json`、`mha_mfu_v2_recheck_report.json`、`mha_mfu_v3_report.json` 和对应日志。如果冒烟阶段失败，脚本会停止，上传其日志即可。

打通后再针对decode缩小/组合Q tile，进一步减少重复计算。现有v0/v1/v2实现均保留。

## v2 split-KV并行

新增 `mha_paged_decode_ascend_v2.py`。分块阶段将每个输出的有效KV tiles分配给多个program，各自写FP32局部最大值m、权重和l、未归一化加权V。第二个kernel按统一最大值重缩放并合并，再除以合并后的l。空split完整写零权重/零累加值，完全空输出保持零。两个kernel在同一当前流按序启动，临时缓冲区每次调用独立分配。

split数从1起倍增，最多16，受页表容量tiles数量约束，输出数×split达到128后停止；仅需一个split时复用v1。该规则尚未经调优，实际短缓存仍可能付出额外合并开销。仍使用FP32 SIMT，不包含Cube计算。

```bash
git pull --ff-only
bash run_mha_v2.sh
```

依次执行v2的540项精度回归、v1的20项重新计时、v2的20项计时，报告 `speedup_vs_v1` 和MFU。计时包含分块与合并两个kernel的完整调用。任何阶段失败即停止。

上传三组JSON及对应log：`mha_paged_accuracy_v2_report.json`、`mha_mfu_v1_recheck_report.json`、`mha_mfu_v2_report.json`。可先用 `python -u test_mha_paged_accuracy.py --impl v2 --suite quick --seeds 7 --output mha_v2_quick.json` 做编译冒烟检查。实际NPU编译、精度及加速仍需验证。

## v1 分块KV版本：精度回归及性能对比

新增 `mha_paged_decode_ascend_v1.py`，v0保持不变。v1每次搬运16或32行K/V，每块一次SIMT VF，完成块内QK、softmax与加权V归约，再更新online softmax。tile整除页大小，不跨物理页；尾块可能加载缓存中的无效行，但计算在读取这些行前做条件判断，避免NaN传播。仍使用FP32 Vector计算，未使用Cube或split-KV。

在先前成功的NPU环境中运行：

```bash
git pull --ff-only
bash run_mha_v1.sh
```

脚本依次执行：v1全部540项精度回归 → 重新测量v0的20项MFU基准 → 测量v1相同20项并计算 `speedup_vs_v0`。保持原有误差容限，任一步失败即停止。不会覆盖此前v0报告；运行前应确认逻辑设备0对应的物理卡空闲。

请上传以下六个文件：

- `mha_paged_accuracy_v1_report.json`、`mha_paged_accuracy_v1.log`
- `mha_mfu_v0_recheck_report.json`、`mha_mfu_v0_recheck.log`
- `mha_mfu_v1_report.json`、`mha_mfu_v1.log`

可先做较短编译/精度验证：

```bash
python -u test_mha_paged_accuracy.py --impl v1 --suite quick --seeds 7 --output mha_paged_accuracy_v1_quick.json
```

v1尚未在作者当前机器上进行NPU编译运行；数学分块更新式与Python语法已本地检查，实际兼容性、精度和加速以NPU报告为准。

## 当前 MHA 的 MFU 基准

读取 `MFU.md` 后新增 `bench_mha_mfu.py`，沿用已通过精度检查的v0 kernel。默认5种形状 × FP16/BF16 × 因果/非因果，共20项，每项先校验精度，再预热5次、用NPU Event测31次，取中位数。编译、输入传输、CPU参考计算不计时；直接调用已编译kernel，不包括Python参数校验。计时包含流内可见的启动间隙及输出分配影响，不是host wall time。

```bash
git pull --ff-only
set -o pipefail
python -u bench_mha_mfu.py 2>&1 | tee mha_mfu.log
```

有效FLOPs按 `4 * H * D * sum(每个请求每个query可见的KV数量)` 计算，QK和PV各贡献一半，不统计softmax或online rescale。decode因果对齐缓存末尾，不能一律按非因果计算量除以2。

分母采用文档给出的28 AIC、FP16/BF16 dense Cube峰值378 TFLOPS；当前kernel实际是FP32 SIMT Vector运算，报告中的 `cube_normalized_mfu_percent` 是相对该模型计算目标的比率，不能解释为Vector或Cube实际占用率。峰值来自用户文档，没有在脚本中独立测定。

默认重复读取同一份输入（warm cache），也可另测每次清理256MiB缓存后的结果：

```bash
python -u bench_mha_mfu.py --cache flush --output mha_mfu_flush_report.json 2>&1 | tee mha_mfu_flush.log
# 先只跑最小规格，共4项
python -u bench_mha_mfu.py --cases decode_b1_k512 --output mha_mfu_small_report.json
```

清理发生在计时区间之外；该模式是软件驱逐缓存的近似，不保证完全冷缓存。请在目标卡空闲时测试。上传 `mha_mfu_report.json` 和 `mha_mfu.log` 后可分析每种形状的MFU。

## 扩大分页 MHA 精度覆盖

D=192 的原始版本曾在 SIMT 布局检查失败；现在将所有按 head 维度并行的循环统一补齐到2的幂，并保护真实UB边界。NPU重测命令（108次，包含此前未完成的三组规格）：

```bash
git pull --ff-only
set -o pipefail
python -u test_mha_paged_accuracy.py --suite extended --cases dim192 cache4096 cache8192 --output mha_paged_accuracy_remaining.json 2>&1 | tee mha_paged_accuracy_remaining.log
```

剩余用例通过后，建议用更新后的kernel完整重跑extended，输出新文件名以保留旧报告。

沿用已成功运行原型的 conda/CANN 环境：

```bash
git pull --ff-only
set -o pipefail
python -u test_mha_paged_accuracy.py --suite standard 2>&1 | tee mha_paged_accuracy.log
```

默认使用随机种子7、42、2026，10种规格 × 3种子 × 2类型 × 2掩码模式 × 3长度模式，共360次检查。覆盖 B=1/2/3/4/8、Sq=1/2/4/8、H=2/4/8、D=16/32/64/80/128/256、页大小16/32/64/128/256、最长2048 token。

每个规格切换混合长度、满缓存、全空缓存；随机页表包含跨请求共享和请求内重复物理页，物理池大于单请求容量。未使用物理行填NaN；零可见长度的输出要求严格为零。使用与原型相同的FP16/BF16容限，同时记录最大/平均绝对误差以及误差与容限之比，不因长缓存放宽容限。

```bash
# 较短检查：108次
python -u test_mha_paged_accuracy.py --suite quick
# 扩展：540次，包括D=48/96/192、页大小48和4096/8192 token缓存
python -u test_mha_paged_accuracy.py --suite extended --output mha_paged_accuracy_extended.json
# 自定义种子；仅列出计划不导入NPU依赖
python test_mha_paged_accuracy.py --list-only --seeds 1 2 3
```

原型逐token计算，长缓存回归可能较慢；输出中的wall_seconds包含编译、数据传输及校验，不是性能基准。报告每次检查后更新，精度不匹配会记录后继续，其他运行错误会停止，避免NPU上下文异常造成连续误报。

完成后请上传 `mha_paged_accuracy_report.json` 和 `mha_paged_accuracy.log`。如果进程被原生运行库终止，报告中的最后一个running用例可辅助定位，并请一并保留日志。

## 分页 MHA Ascend 原型精度检查

`mha_paged_decode_ascend_v0.py` 是独立 SIMT 原型：每个 program 处理一个请求、query 位置和 head，逐 token 更新 FP32 online softmax。尚未完成 NPU 验证，也没有性能优化或 TileOPs 集成。

在原来成功运行向量加法的环境中执行：

```bash
conda activate zwh
source /home/CANN/9.3.0.weekly.20260930/ascend-toolkit/set_env.sh
export ASCEND_RT_VISIBLE_DEVICES=4
git pull --ff-only
set -o pipefail
python -u mha_paged_decode_ascend_v0.py 2>&1 | tee mha_paged_decode_v0.log
```

直接运行文件会用 CPU FP32 参考结果检查 FP16/BF16 × 因果/非因果四种组合，包括分页乱序、共享页、缓存尾部 NaN、零长度及完全被掩码的 query。四条 `PASS` 且进程退出码为 0 表示这些小用例通过，不代表完整算子精度或性能验收。若编译失败，请保留完整日志。

可将 `mha_paged_decode_v0.log` 上传到此仓库或发回聊天，以继续定位。逻辑 `npu:0` 对应可见物理卡4。

## TileLang Ascend 最小编译运行验证

在此前成功采集环境的同一终端、同一 conda/CANN 环境执行：

```bash
git pull --ff-only
python collect_env.py
python verify_tilelang_ascend.py
```

第二个命令更新环境报告，第三个命令显式以 `target="ascend"` 编译 FP32 向量加法，随后执行随机输入、更换输入、全零输入三组 CPU 参考比对。不运行性能基准，不使用 CPU 回退。初次编译可能较慢。

看到 `PASS: compiled and executed TileLang Ascend kernel` 且进程退出码为 0，才表示这条编译和执行链路通过。脚本保存 `tilelang_ascend_smoke.json`；失败时也会保存阶段、版本及异常堆栈。若 Python 进程被原生运行库终止，报告可能来不及生成，此时请保留终端日志。

当前 `ASCEND_RT_VISIBLE_DEVICES=4` 时，默认逻辑设备 `--device 0` 即物理卡 4，无需改成 `--device 4`。

请将更新的 `ascend_env_report.json` 和 `tilelang_ascend_smoke.json` 发回聊天；也可将报告手动提交到仓库（smoke 报告默认被 gitignore 忽略，可用 `git add -f tilelang_ascend_smoke.json`）。

API 和缓冲区配置依据安装版本对应的[官方 Ascend 向量加法示例](https://github.com/tile-ai/tilelang/blob/a7cb753dafc6b2514cbbd7c8431260c9d05e7df0/examples/ascend/example_vecadd.py)及[后端说明](https://github.com/tile-ai/tilelang/blob/a7cb753dafc6b2514cbbd7c8431260c9d05e7df0/tilelang/ascend/README.md)。

环境探测中的 TVM 构建信息、目标类型、注册函数和 Ascend 语言模块已拆成独立项，某项 API 不可用不会跳过后续探测。

用于 TileOPs Ascend 适配的环境诊断工具。脚本仅依赖 Python 标准库，不安装软件、不需要 sudo、不访问网络。

## 运行

在 Ascend 机器上，先激活实际开发使用的 Python 环境，并按机器原有方式加载 CANN 环境，然后运行：

```bash
git clone https://github.com/zhangwenhao91/FGW.git
cd FGW
python3 collect_env.py
```

已克隆则先 `git pull --ff-only`。务必使用安装了 torch、torch_npu、tilelang 的那个 Python；也可以用其绝对路径执行脚本。

生成 `ascend_env_report.json` 和 `ascend_env_report.txt`，将其中任意一个文件发回聊天即可。报告保存在本机，不会自动上传，已被 gitignore 排除。

## 采集内容

- 操作系统、CPU、内存、磁盘、Python 解释器及已安装包版本。
- NPU 管理工具信息、设备名称、数量、属性和显存信息。
- CANN / 驱动 / 固件版本文件，以及编译工具版本；无法访问的项目保留错误。
- PyTorch、torch_npu 的实际导入版本及路径、PyTorch 构建配置。
- TileLang 的实际导入版本及路径、TVM 构建信息、目标类型、Ascend 相关注册函数和文件、语言接口符号。
- 对用户提供的 `Ascend950PR_9579`、`torch 2.13.0+cpu`、`torch_npu 2.13.0`、`tilelang 0.1.15+cpu.gita7cb753d` 做精确比对。
- 默认在逻辑设备 `npu:0` 上运行很小的 FP32 / FP16 / BF16 逐元素、RMSNorm 计算和矩阵乘法探测，记录误差或异常。

TileLang 的符号和目标探测只能提供后端支持线索，不代表已经成功编译或运行 TileLang NPU kernel。精度探测记录数值，不作为正式算子验收测试。

```bash
# 仅采集环境，跳过 NPU 计算
python3 collect_env.py --skip-smoke

# 慢速初始化环境可延长每个 Python 子进程的超时
python3 collect_env.py --timeout 300 --output reports/ascend_env_report
```

各探测项独立执行；缺失命令、导入失败、超时不会阻止其他信息采集。输出可能包含软件安装路径、包名及设备标识，分享前可检查。脚本不读取令牌、SSH 密钥或完整环境变量。
