# 实验性统一split代价模型

本文记录候选阶段；当前默认范围见[mha_model_default.md](mha_model_default.md)。原候选入口：`split_policy="cost_model"`。
参数num_splits显式指定时优先于策略，Cube计算体不变。
候选仅用于28-Cube、BF16/D128、page128、非因果、manual/stages2/buffers2/Q驻留。
模型使用页表容量对应KV块数；目前校准及验证均为每请求等长且缓存长度等于容量的场景。
变长、短于容量、不同硬件/精度/tiling尚未验证，不能据此推广。

## 模型及拟合

令T=B*H*ceil(Q/128)，S为候选split，K为KV128块数，
W=ceil(T*S/28)，L=ceil(K/S)，M=B*H*ceil(Q/min(8,next_power_of_two(Q)))。
用6个非负权重组合：W*L、W、1[S>1]、1[S>1]*T*S、1[S>1]*M*S、1[S>1]*W*L。
这些项分别描述计算波次和KV循环、每波固定工作、多kernel固定成本、partial流量、merge规模以及多split计算路径差异。
成本分数用于选择split，不是绝对微秒预测，也不能解释为独立硬件计数器。

离线用64形状的各候选中位耗时，拟合同形状相对split1的差值，消除形状固定成本；
按最佳耗时归一化，采用非负坐标下降。源码和系数、源报告SHA256均保存。
`python -m tools.mha.fit_split_cost_model` 可复现，离线拟合需要NumPy；NPU选择器只用标准库，无新增依赖。

完整结果见mha_split_cost_model_evaluation.json：

- 训练数据回代：55/64形状距最佳不超过2%，几何平均耗时损失1.29%，最差13.39%。
- KV留一：各组几何平均损失0–2.68%，最差13.39%。
- B/H留一：留出1x1时几何平均损失5.37%、最差30.72%；留出2x8时平均1.61%、最差18.33%。

这些是探索性评估，不等于独立验证，更不能声称模型已优于现有策略所有场景。

## NPU独立验证

`python fgw.py mha run run_mha_split_model_validation`

先KV4096、B1/H1及B1/H8、Q1/129冒烟，再跑：

- 新规格：KV4096/16384，B1/H1、B1/H8、B4/H4、B2/H16，Q1/64/129/256/512/2048/8192，共56形状。
- 原边界复测：KV2048/32768，B1/H1及B2/H8，Q1/128/129/1024，共16形状。

共72形状、360候选精度检查、4轮1440条性能记录；保持原BF16独立CPU参考及Event计时口径。
每形状扫描1/2/4/8/16，模型选中的split和legacy选中的split直接复用对应候选callable，避免重复计时。
模型权重冻结：本轮脚本不会拟合或修改模型，不向模型泄露新结果。

报告best_by_case.cost_model包括所选split、实测耗时/最佳耗时、相对legacy的加速比及精度状态。
完整ranking和每轮胜者保留，以区分选择错误与计时波动。若候选失败，只能比较已通过候选。

## 上传

reports/mha/latest下：

- mha_split_model_holdout_kv4096_summary.json 和.log
- mha_split_model_holdout_kv16384_summary.json 和.log
- mha_split_model_boundary_kv2048_summary.json 和.log
- mha_split_model_boundary_kv32768_summary.json 和.log

冒烟失败上传mha_split_model_smoke_summary.json和.log。原始样本完整JSON留本地。
测试完成后再决定是否修正模型及设为默认；不按B/H或Q查表记忆扫描结果。
