# FGW

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
