# 提交说明

- `run.sh`：训练过程中使用的命令
- `inference.sh`：测试时使用的命令
- `select_output.py`：从生成结果中随机选取句子的脚本
- `model_tfmr.py`
  - 新增了一个按 batch 计算 loss 的函数，避免 `main.py` 和 `model_tfmr.py` 中的代码重复
- `main.py` 做了许多修改，包括
  - 增加更多可选参数
  - 实现加载预训练模型的不同 layer
  - 一些工具函数，如 loss 记录，early_stop 策略调整，数据集的重复和混洗等等，便于进行探索实验
  - 优化模型输出的格式
- `outputs.txt`：提交的模型生成结果