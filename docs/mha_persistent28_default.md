# 常驻28正式默认

v7 Cube路径的build_kernel和mha_paged_decode现在默认persistent_programs=28，
compute_grid=min(tasks*splits,28)。所有Q统一采用这一上限，任务数不足28时仍按实际数量启动。
没有按Q长度切换版本。merge计算保持原样，冻结split模型系数保持原样。
非Cube规格继续既有v2回退，不因默认28参数被拒绝。
显式persistent_programs=None恢复无上限grid，用于对照。
精度CLI的--persistent-programs 0对应无上限；默认v7为28。

## 已有实机验证

候选30项非因果、9项单query因果精度通过，40性能形状均通过。
实际改变grid的22形状连续4轮提速；Q4096同进程msprof确认完整调用约330.62->282.76us。
短Q补测17个同grid控制组和2个收益参照组，每项12轮/61对：精度通过。
17控制组normal/persistent28生成设备源码哈希全部相同，原B1/H1/Q16/KV2048异常不再复现。
但相同源码的短任务仍有小幅、部分稳定的Event差异：
B2/H8/Q16/KV2048速度比中位0.9876（0/12轮更快），B1/H1/Q512/KV2048为0.9825（0/12）。
不能声称所有短Q都更快，也不能把差异直接归因于kernel算法改变。
两收益参照组Q4096/B2H8和Q8192/B1H1均12/12轮提速，中位分别1.1524x、1.1055x。

## 入口一致性

公共API、普通MFU、msprof和源码导出采用当前默认。
--compare-persistent明确保留无上限control对照28候选，不随默认值改变。
--compare-legacy明确保留旧split选择+无上限grid。
历史split扫描、split1对照、Q驻留/调度实验均显式固定无上限，以便复现旧报告。
模型配置范围、固定满缓存判定及范围外legacy split回退不变。

更新后无需指定参数即可使用28默认：`git pull --ff-only`。
本次只是将已经上机验证的候选设为默认，没有再次修改计算体。
