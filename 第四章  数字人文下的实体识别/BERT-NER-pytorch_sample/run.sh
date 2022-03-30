###
 # @Author: your name
 # @Date: 2021-04-15 10:54:17
 # @LastEditTime: 2021-04-15 11:00:54
 # @LastEditors: Please set LastEditors
 # @Description: In User Settings Edit
 # @FilePath: /BERT-NER-Pytorch/run.sh
### 
CUDA_VISIBLE_DEVICES=1  python run_ner.py --data_dir=train_data_cixing/ \
--bert_model=pretrain_models/sikuroberta_vocabtxt/ \
--task_name=ner \
--output_dir=output/train_data_cixing_out/ \
--max_seq_length=128 \
--do_train --do_eval  --eval_batch_size=64  --train_batch_size=64 --num_train_epochs 10 \
--warmup_proportion=0.4 > logsikubert0.log 2>&1 & echo $! > run.pid