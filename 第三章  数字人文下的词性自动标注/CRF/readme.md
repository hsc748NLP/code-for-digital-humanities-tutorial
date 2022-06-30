#### 使用CRF实现词性标注的步骤
以Window10为例：
- 1.打开终端，并进入CRF项目文件夹
   - 同时按<kbd>Win</kbd>+<kbd>R</kbd>打开“运行”窗口。
   - 输入`cmd`并按`确定`。
   - 在终端窗口中，输入`cd`+`空格`+`CRF文件夹绝对路径`，如`cd code-for-digital-humanities-tutorial\第三章  数字人文下的词性自动标注\CRF`
- 2.依次在终端中输入下述的CRF运行指令，即可实现基于CRF的词性标注模型的训练、测试、性能评估。

#### CRF运行指令

1.训练模型
>crf_learn template train.txt model 

2.测试模型
>crf_test -m model test.txt >output.txt

3.评估模型在测试集上的效果
>conlleval.pl < output.txt