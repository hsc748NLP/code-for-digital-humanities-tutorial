
## word2vec计算古籍文本语义相似度

使用的数据集为data文件夹下的pairs_source.json文件，若想在自己的数据集上训练，应将您的语料分词后以列表形式存入json文件中。

运行如下代码:

```
python word2vec.py
```
修改代码最后一行的参数，可打印出语料中任意两个词语的相似度。

## 孪生神经网络计算古籍文本语义相似度
主要数据集为pairs_source.json，pairs_target.json，以及古汉语与译文的词汇对应关系文件forward.json。

双语语料以及双语词汇的对齐由fast align工具生成，读者可通过github的fast align项目生成自己的平行语料。

如需重新训练自己的数据集，需取消train.py的138行的训练注释。

运行如下代码:
```
python train.py
```

当不注释训练语句时，仅返回模型的预测结果。
