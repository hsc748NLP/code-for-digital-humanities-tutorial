# 基于pytorch的中文语言模型预训练

提供三种中文语言模型预训练的方法。预训练bert类模型对硬件的要求较高，建议在16G以上显存的设备上运行代码。

## bert-base-chinese

(https://huggingface.co/bert-base-chinese)
​

基于官方案例实现bert模型训练。

https://github.com/huggingface/transformers/tree/main/examples/pytorch/language-modeling
(本文使用的transformers版本为3.4.0)
```
python run_language_model_bert.py     --output_dir=output     --model_type=bert     --model_name_or_path=bert-base-chinese     --do_train     --train_data_file=data/train.txt     --do_eval     --eval_data_file=data/eval.txt     --mlm --per_device_train_batch_size=4  --save_total_limit=1  --num_train_epochs=5

```
会自动从官网上下载bert-base-chinese模型来继续训练。

## roberta-wwm-ext

(https://github.com/ymcui/Chinese-BERT-wwm)


要基于上面的代码run_language_model_roberta.py继续预训练roberta。还需要做两个改动。
* 下载roberta-wwm-ext到本地目录hflroberta，在config.json中修改“model_type”:"roberta"为"model_type":"bert"。
* 对上面的run_language_modeling.py中的AutoModel和AutoTokenizer都进行替换为BertModel和BertTokenizer。

假设config.json已经改好，可以运行如下命令。
```
python run_language_model_roberta.py     --output_dir=output     --model_type=bert     --model_name_or_path=hflroberta     --do_train     --train_data_file=data/train.txt     --do_eval     --eval_data_file=data/eval.txt     --mlm --per_device_train_batch_size=4  --save_total_limit=1  --num_train_epochs=5
```

### ernie
https://github.com/nghuyong/ERNIE-Pytorch）

ernie是百度发布的基于百度知道贴吧等中文语料结合实体预测等任务生成的预训练模型。这个模型的准确率在某些任务上要优于bert-base-chinese和roberta。如果基于ernie1.0模型做领域数据预训练的话只需要一步修改。

* 下载ernie1.0到本地目录ernie，在config.json中增加字段"model_type":"bert"。
运行
```
python run_language_model_ernie.py     --output_dir=output     --model_type=bert     --model_name_or_path=ernie     --do_train     --train_data_file=train.txt     --do_eval     --eval_data_file=eval.txt     --mlm --per_device_train_batch_size=4  --save_total_limit=1  --num_train_epochs=5

```
