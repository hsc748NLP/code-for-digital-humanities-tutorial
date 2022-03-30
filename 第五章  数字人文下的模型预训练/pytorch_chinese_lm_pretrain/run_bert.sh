###
 # @Author: your name
 # @Date: 2021-05-15 20:18:45
 # @LastEditTime: 2021-06-08 20:27:08
 # @LastEditors: Please set LastEditors
 # @Description: In User Settings Edit
 # @FilePath: /pytorch_chinese_lm_pretrain-master/run_bert.sh
### 
TRAIN_FILE='train.txt'
TEST_FILE='eval.txt'
PreTrain_Model='bert-base-chinese'
mkdir -p log
CUDA_VISIBLE_DEVICES=0,1 python run_language_model_bert.py   \
    --output_dir=output/$PreTrain_Model   \
    --model_type=bert     \
    --overwrite_output_dir \
    --save_total_limit=3 \
    --num_train_epochs=10 \
    --learning_rate=5e-4 \
    --local_rank=-1 \
    --model_name_or_path=$PreTrain_Model  \
    --do_train     \
    --train_data_file=$TRAIN_FILE     \
    --do_eval     \
    --eval_data_file=$TEST_FILE     \
    --mlm \
    --per_device_train_batch_size=32  \
    > log/log_$PreTrain_Model.log 2>&1 & echo $! > log/run_$PreTrain_Model.pid
