
## 命名实体识别代码

该部分为本书第四章节对应的源代码，包含数据预处理和使用LSTM-CRF与BERT实现命名实体识别的代码实现


## 数据预处理模块

1.将预处理的数据放入data文件夹下，其格式需与filename.txt保持一致。
2.运行pro_ner.py将数据转为BIOES标注格式
3.运行train_test_divide.py划分训练集与测试集


## 命名实体识别模块

1.BILSTM-CRF代码见ChinsesNER-pytorch-master文件夹
2.BERT代码见BERT-NER-pytorch_sample文件夹
