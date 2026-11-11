# FGW

## 默认配置选择前的短Q成对交替测量

```bash
git pull --ff-only
set -o pipefail
python -u profile_q_resident_paired.py 2>&1 | tee mha_q_resident_paired.log
```

覆盖Q16/Q32（B2/H8/KV2048）和B1单query因果/非因果（H16/KV512），固定BF16/D128。所有候选在同一进程先完成编译与CPU参考检查，再运行8轮；每轮每场景预热30对、测61对，逐对反转base/resident顺序，轮间反转场景顺序。保存原始Event样本、每轮耗时及加速比和胜出轮数，不自动修改默认配置。

上传 `mha_q_resident_paired.json` 和 `mha_q_resident_paired.log`。结合此前长Q结果，确认无稳定短Q回退后再统一更改默认值。

## Q驻留候选的短Q性能确认

```bash
git pull --ff-only
bash run_mha_short_q_resident.sh
```

对比当前v7默认base和仅开启q_resident的候选，保持两级流水及缓冲区版本。固定BF16/D128、B2/H8/KV2048，扫描Q1/4/16/32/64/128，并加入原B1/H16/KV512单query用例。7项非因果，另2项单query因果，每项先校验CPU参考精度，再warm-cache Event预热20次、采样51次。两轮反转候选与形状顺序，共36项计时结果；多query因果不在此次Cube测试范围。

上传 `mha_short_q_*_run*_noncausal.json`、`mha_short_q_*_run*_causal.json` 和对应log。默认配置仍未修改，待短Q结果确认后决定是否统一启用Q驻留。

## v7 Cube搬运交叠候选扫描

保持同一v7计算/合并实现和原默认值，新增编译参数q_resident与kv_versions。对比base（原2/2）、resident（Q在KV循环外搬到L0A）、prefetch（三级流水、K/V L1三版本、其余缓冲区两版本）、combined（两者组合）。Q驻留会延长L0A生命周期，需要实际编译检查，不预先假定可行或加速。

```bash
git pull --ff-only
bash run_mha_overlap_sweep.sh
```

每个候选先做30项非因果Cube精度（含Q尾块及单/多split）与6项单query因果精度，再测Q4096/Q32768的msprof及warm-cache Event MFU。两轮反转候选顺序，每份报告独立进程运行。失败时保存报告日志并继续下一候选，不会自动选取配置或改变默认值。

请上传 `mha_overlap_*_run*_accuracy.json`、`*_causal.json`、`*_msprof.json`、`*_mfu.json` 和对应log（均以mha_overlap开头）。精度/编译失败后没有对应性能报告是预期行为；实际错误以日志为准。msprof与Event两种缓存/计时口径继续分开比较。

## v7：单split直接归一化、转换并输出

Cube路径在编译期split=1时，直接在原计算kernel内归一化并转BF16，写最终连续[B,Q,H,D]，不生成Partial/LSE、不编译或启动merge。输出采用[B,Q,H*D]的等价视图，保留Q轴边界以正确处理Q尾块。类型转换复用FP32输出UB，沿用参考实现的顺序SIMD转换形式。split>1保留v6的批量合并，不按长短Q切换历史版本。支持范围仍为BF16/D128/整块KV，其他规格保持原回退。

```bash
git pull --ff-only
bash run_mha_v7.sh
```

先跑2项直接输出冒烟，覆盖单Q和Q129尾块；再跑30项非因果及9项单query因果精度（包含直接与多split两条路径）。随后采集Q4096/Q32768的msprof并进行长Q正反向MFU扫描。新报告用 `kernel_mode` 标记直接输出或split合并；msprof的单split记录full/direct，不再有虚构的merge时间。

请上传上述脚本生成的v7 JSON与对应log，尤其 `mha_v7_direct_smoke.json`、`mha_cube_v7_accuracy.json`、`mha_cube_v7_causal.json`、`mha_v7_large_q_msprof.json`、`mha_mfu_v7_large_q_report.json` 和 `mha_mfu_v7_large_q_reverse_report.json`。任一步失败停止。NPU编译、精度与实际收益尚待验证。

## Q4096/Q32768的局部计算、合并和流水计数器

```bash
git pull --ff-only
bash run_mha_large_q_msprof.sh
```

固定v6、BF16、B2/H8/D128/KV2048/page128、非因果和默认流水2/2，分别采集full/partial/merge，每项3轮、预热10次、重复30次。CPU参考分块128行验证，不计入profile。报告记录kernel源码SHA256和实际规则推导的启动规格，Q32768使用已修复的有界merge网格。两个长Q场景的split数均为1，但当前实现仍经过partial和merge，可据结果评估这部分中间处理成本。

上传 `mha_large_q_msprof_report.json`、`mha_large_q_msprof.log`、`mha_large_q_msprof_report_msprof_source.py`。该后端有缓存驱逐，dur_ns按上游profiler记录聚合口径，不能直接替代此前warm-cache Event延迟；各pipe比率可重叠，不可相加作为耗时占比。kernel没有改动。

## 增大Q，观察MFU平台期

```bash
git pull --ff-only
bash run_mha_large_q.sh
```

固定v6/BF16/B2/H8/D128/KV2048/page128/非因果，扫描Q=512/1024/2048/4096/8192/16384/32768，正反各测一次。与此前长Q测试一样预热20次、Event采样51次，warm-cache，378 TFLOPS归一化，先校验精度再计时。没有修改kernel。

CPU参考按128行Q分块，避免构建完整长Q注意力矩阵；Q512先与原逐query参考交叉校验。较大Q的参考计算与NPU测试可能较慢，报告会在每个用例前后保存，可从日志判断进度。Q大于KV长度是合法的非因果矩形注意力场景，但不是普通逐token decode。

上传 `mha_mfu_v6_large_q_report.json`、`mha_mfu_v6_large_q_reverse_report.json` 及对应log。结果只代表该固定B/H/D/KV、软件版本及缓存口径下的趋势，不等于设备或算子的理论MFU上限。

## 补测长Q的MFU

```bash
git pull --ff-only
bash run_mha_long_q.sh
```

固定v6、BF16、B2/H8/D128/KV2048、page128及非因果，扫描Q=4/16/32/64/128/129/256/512。包含Q4对照和Q129尾块，全部走Cube路径。每项先校验CPU参考精度，再预热20次、NPU Event测51次取中位数；保持warm-cache和378 TFLOPS峰值口径，完整调用包括Q布局准备、Cube局部计算及合并。正向/反向各测一次，不修改kernel。

上传 `mha_mfu_v6_long_q_report.json`、`mha_mfu_v6_long_q_reverse_report.json` 和对应log。按实际有效Q计算FLOPs，不将补齐行计入有用计算；较长Q的CPU参考验证可能稍慢。

## v6：直接输出及按行计算合并权重

v6保持v5统一的批量合并算法，输出直接写入最终连续[B,Sq,H,D]张量的各head区域，去掉额外permute/contiguous。每个split先按query行计算peak、旧结果缩放及新权重，再广播给D维累加；行状态在累加后更新。Cube局部计算与默认2/2配置不变，没有按长短Q选择历史版本。

```bash
git pull --ff-only
bash run_mha_v6.sh
```

依次检查21项非因果（含Q尾块）及6项单query因果精度，再重测v5/v6的msprof与20项MFU。精度检查新增输出连续性断言。直接跨head步长的UB→GM写出与Q尾块裁剪仍需NPU编译/精度验证。

上传 `mha_cube_v6_accuracy.json`、`mha_cube_v6_causal.json`、`mha_merge_v5_recheck.json`、`mha_merge_v6_msprof.json`、`mha_mfu_v5_recheck.json`、`mha_mfu_v6_report.json` 及对应log。v5只是测量基线，v6内部没有长短Q版本分流；BF16/D128/整块KV支持范围仍保持不变，其他规格的已有v2回退未扩展。

## v5批量合并query行

新增独立v5版本，Cube局部计算保持v4默认2/2配置不变。合并每个program处理至多8个真实query，Sq小于8时使用1/2/4行，连续读取Parts和LSE，减少逐行小搬运。8整除Q tile128，合并块不会跨partial tile。内部按[B,H,Sq,D]连续存储，再转换为对外连续[B,Sq,H,D]；该输出转换包含在full和merge计时中，不能漏算。需要NPU验证尾块clamp和自动布局。

```bash
git pull --ff-only
bash run_mha_v5.sh
```

先跑21项非因果Cube边界精度与6项单query因果精度，再重新采集v4/v5的四场景msprof，最后执行20项MFU基准比较。请上传 `mha_cube_v5_accuracy.json`、`mha_cube_v5_causal.json`、`mha_merge_v4_msprof.json`、`mha_merge_v5_msprof.json`、`mha_mfu_v4_merge_baseline.json`、`mha_mfu_v5_report.json` 和对应日志。msprof解析器源码文件可一并保留。默认实现仍为v4，没有预先假定批量合并能加速。

## 根据msprof结果对照流水候选

已保留v4默认两级配置，build_kernel新增pipeline_stages和buffer_versions参数。先对比(1,1)、(2,2)、(3,3)三个组合，每个在原有四种场景校验精度再采集msprof；两次独立进程扫描反转候选顺序。计时仍按上游msprof_detail口径，包含局部计算和合并分别采样，不能将pipe比率相加或直接当作端到端时间占比。

```bash
git pull --ff-only
bash run_mha_pipeline_sweep.sh
```

上传六组 `mha_pipeline_s*_v*_run*.json` 和对应log。失败候选会保留错误并继续其他组合；全部通过后脚本才返回0。每份报告另有同名前缀的msprof解析器源码。默认配置不自动替换，等对比结果后选取候选，并针对新候选扩大Q尾块精度覆盖。

本轮msprof已显示Q128下merge聚合时长显著高于partial；流水扫描只影响partial，因此也可能不足以改善该场景，合并优化仍需单独评估。

## 使用msprof_detail采集Cube流水计数器

```bash
git pull --ff-only
set -o pipefail
python -u profile_mha_msprof.py 2>&1 | tee mha_msprof.log
```

覆盖单query因果/非因果、Q4、Q128四种场景，分别采集full/partial/merge，每项3轮、预热20次、重复100次。保持当前v4的两级流水和缓冲区版本不变，先看计数器再选调优候选。若需要短运行，可加 `--cases decode_noncausal --rounds 1 --repeat 30`。

使用与安装版本对应的上游 `do_bench(backend="msprof_detail")`。记录AIC MAD、MTE、FIXPIPE以及AIV Vector/Scalar等字段。该后端每次调用前驱逐缓存，dur_ns是上游对profiler记录聚合后除以重复次数的结果，不是Event中位耗时；不直接将它与此前warm-cache Event读数比较，也不把各pipe占比相加。脚本额外保存本机msprof解析器源码与SHA256，方便核对聚合口径。

上传 `mha_msprof_report.json`、`mha_msprof.log`、`mha_msprof_report_msprof_source.py`。若profiling不被当前环境支持，会保留原始错误，不回退Event冒充msprof结果。上游采集器会清理它自己的临时原始profile，本脚本保留其全部返回计数器。

## v4单query因果/非因果重复测量与分阶段诊断

```bash
git pull --ff-only
set -o pipefail
python -u profile_mha_v4.py 2>&1 | tee mha_v4_profile.log
```

在之前成功的NPU环境运行。固定同一组BF16输入 B1/Sq1/H16/D128/KV512，分别编译因果和非因果版本，先对照CPU参考并检查两版本输出一致。默认6轮，每阶段预热20次、采样51次，交替因果/非因果顺序，并轮换阶段顺序。

报告包含完整调用、局部Cube计算、split合并、Q布局转换的NPU Event原始样本与每轮中位数。Sq=1时布局转换可能是零拷贝视图，另记录是否共享存储、CPU转换调用耗时和空Event开销，不把空Event时间当作实际数据搬运。独立阶段计时不可直接相加，完整调用仍以full项为准。

本次仅暴露v4已编译的partial/merge函数供计时，没有更改其计算逻辑。请上传 `mha_v4_profile_report.json` 和 `mha_v4_profile.log`，完成后再判断差异来自哪个阶段。

## v4：真实Q分块及Q尾块，KV仍128对齐

`mha_paged_decode_ascend_v4.py`将每请求、每head的真实query按128行分块，去掉v3的同一query复制。Q输入先转换到连续的[B,H,Sq,D]布局，让GM→L1尾块补齐沿最后两个轴工作；该转换的开销包含在完整计时中。最后一个Q块用参考PR的自动clamp/zero-pad路径，merge仅输出真实query行。

仍仅BF16/D128/128对齐页和KV长度走Cube；不支持KV尾块。多query因果暂回退v2，单query因果支持。保留v3以便对比。

```bash
git pull --ff-only
bash run_mha_v4.sh
```

先用Sq=4和129做2项冒烟，再对Sq=1/4/127/128/129/257做21项非因果回归和6项单query因果回归，均走Cube，包含3个随机种子。随后重新测v3并跑v4的20项性能比较（其中3项Cube、17项回退）。上传 `mha_cube_v4_smoke.json`、`mha_cube_v4_accuracy.json`、`mha_cube_v4_causal.json`、`mha_mfu_v3_recheck_report.json`、`mha_mfu_v4_report.json` 及对应日志。

这是独立Q尾块实验，不能视作全形状540项覆盖；NPU编译与精度仍需验证。

## 当前v3仅验证完整KV块

按最新约定，v3 Cube路径仅接受正数且128对齐的有效KV长度，已去掉尾块分支及V预处理，K/V直接按页表进行完整二维GM→L1搬运。BF16/D128、页大小128对齐的范围不变。多query因果模式会使各行可见长度出现非128对齐，因此暂回退v2；单query因果与非因果都可走Cube。直接调用build_kernel所得程序时，调用方也必须保证长度满足约束。

不要再运行旧的混合/空缓存v3冒烟命令，改用：

```bash
git pull --ff-only
bash run_mha_v3.sh
```

脚本先跑3项整块Cube冒烟，再跑9项非因果和6项单query因果Cube精度用例（均为BF16、3个随机种子），然后执行v2/v3性能对比。当前20项性能中3项走Cube，其余回退v2。此前540项完整覆盖仍是v2的基线，不能将新的整块Cube验证视作540项通过。

仍保留单query复制到128行的bring-up布局，尚未缩小Q tile，实际性能以新报告为准。

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
