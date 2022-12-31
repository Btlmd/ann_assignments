# 代码修改说明
- `main.py` 
  - ArgParse 中引入了更多参数，便于调参
  - 使用 wandb 记录训练过程
  - 进行了随机数种子设置等初始化工作
- `models.py` 
  - 给模型增加了隐藏层大小等更多参数，便于调参
- `basic.sh` 
  - 运行环境，默认参数， wandb 模式等初始设定
- `commands.sh` 
  - 实验过程中参数搜索的脚本

以默认参数运行模型，可以在 `basic.sh` 中配置好 conda，数据集位置等，然后执行

```bash
bash
. ./basic.sh && train
```

注意脚本只能在 bash 中正常执行而可能不能在 zsh 中执行。