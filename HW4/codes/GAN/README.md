# 提交说明

主要修改如下

- `GAN.py` 增加 MLP GAN 实现
- `dataset.py` 增加 worker 的随机数种子固定
- `main.py` 增加
  - 对 MLP GAN 支持
  - 随机采样和隐空间插值代码
  - 一些工具函数
- `grouping.py` 根据 log 统计各实验组的均值方差等信息
- `run.sh` 实验过程中执行的所有命令