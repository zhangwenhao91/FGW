# 多算子目录约定

新算子使用稳定的小写名称，如rmsnorm。建立 `operators/rmsnorm/`、`tests/rmsnorm/`、`benchmarks/rmsnorm/`、`scripts/rmsnorm/`。实现放kernels或算子子目录，current.py导出当前统一接口；必要的历史实验可保留，但不让业务调用方按版本选择。

公共测试和基准接口应显式记录算子、实现、dtype、shape、语义参数、设备、编译配置、精度状态及计时口径。新增fgw.py命令映射，让结果落在reports/<op>/latest。归档只移动有用报告，不提交缓存、编译对象、令牌或机器私钥。

基准应先做独立参考精度检查，编译与数据传输不计入kernel计时。MFU分子只计模型有效计算量。明确区分Event、msprof、缓存策略及完整调用/单kernel范围。

目录调整的本机验证使用标准库编译和CLI --help/--list-only；没有Ascend硬件的机器不运行或声称通过NPU检查。算子计算改动后在目标机运行对应精度及性能检查。
