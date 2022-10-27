###
 # @Author: your name
 # @Date: 2021-05-14 20:57:35
 # @LastEditTime: 2021-05-30 14:55:08
 # @LastEditors: your name
 # @Description: In User Settings Edit
 # @FilePath: /pytorch_chinese_lm_pretrain-master/run_roberta.sh
### 
TRAIN_FILE='train.txt'
TEST_FILE='eval.txt'
PreTrain_Model='roberta-base'
From_Scratch='/home/admin/zihe.zhu/pytorch_chinese_lm_pretrain-master/train_tokenizer/pretrained_models/'
mkdir -p log
CUDA_VISIBLE_DEVICES=0,1 nohup python full_copy.py \
    --output_dir=output/$PreTrain_Model \
    --model_type=bert \
    --overwrite_output_dir \
    --save_total_limit=3 \
    --num_train_epochs=10 \
    --learning_rate=5e-4 \
    --local_rank=-1 \
    --cache_dir=$From_Scratch \
    --config_name=$From_Scratch \
    --tokenizer_name=$From_Scratch \
    --do_train \
    --train_data_file=$TRAIN_FILE \
    --do_eval \
    --eval_data_file=$TEST_FILE \
    --mlm \
    --per_device_train_batch_size=32  \
    > log/log_$PreTrain_Model.log 2>&1 & echo $! > log/run_$PreTrain_Model.pid