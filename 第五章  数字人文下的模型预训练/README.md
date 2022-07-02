## 语言模型预训练

本部分包含预训练BERT类模型和将bin模型转换为ckpt模型的代码

1.pytorch_chinese_lm_pretrain文件夹内包含bert类模型预训练的基础代码，修改sh文件夹中的文件即可使用。此处实现参照了transformers库的预训练预训练代码和中文模型预训练的github项目(https://github.com/zhusleep/pytorch_chinese_lm_pretrain)

2.transfer.py用于将预训练完成的bin文件转为ckpt格式，可供tensorflow框架加载。

## 建议运行环境
```
torch==1.6.0

transformers==3.4.0

1.15.0<= tensorflow <2.0
```
