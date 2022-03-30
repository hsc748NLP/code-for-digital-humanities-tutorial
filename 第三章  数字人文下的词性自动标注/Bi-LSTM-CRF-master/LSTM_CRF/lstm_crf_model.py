#encoding=utf-8
import tensorflow as tf
from tensorflow.contrib.rnn import LSTMCell
from tensorflow.contrib.crf import crf_log_likelihood
class BiLSTM_CRF(object):
    def __init__(self,hidden_dim,num_tags,input_x,sequence_lengths,dropout_pl,labels):
        self.hidden_dim=hidden_dim
        self.num_tags=num_tags
        self.input_x=input_x
        self.sequence_lengths=sequence_lengths
        self.dropout_pl=dropout_pl
        self.labels=labels
    # 建立模型，执行正向传播，返回正向传播得到的值
    def positive_propagation(self):
        with tf.variable_scope('lstm-crf'):
            cell_fw=LSTMCell(self.hidden_dim)
            cell_bw=LSTMCell(self.hidden_dim)
            #inputs(self.input_x)的shape通常是[batch_size, sequence_length, dim_embedding]
            #output_fw_seq和output_bw_seq的shape都是[batch_size, sequence_length, num_units]
            (output_fw_seq, output_bw_seq), _=tf.nn.bidirectional_dynamic_rnn(cell_fw,cell_bw,self.input_x,self.sequence_lengths,dtype=tf.float32)
            out_put=tf.concat([output_fw_seq,output_bw_seq],axis=-1) #对正反向的输出进行合并
            out_put=tf.nn.dropout(out_put,self.dropout_pl) #防止过拟合
        #循环神经网络之后进行一次线性变换，用于把输出转换为crf_log_likelihood的接收格式，主要
        #是把最后一维的维度转换为num_tags，以便于随后进行优化
        with tf.variable_scope('proj'):
            W=tf.get_variable(name='W',
                              shape=[2*self.hidden_dim,self.num_tags],
                              initializer=tf.contrib.layers.xavier_initializer(),
                              dtype=tf.float32
                              )
            b=tf.get_variable(name='b',
                              shape=[self.num_tags],
                              initializer=tf.zeros_initializer,
                              dtype=tf.float32
                              )
            s=tf.shape(out_put)
            #正向传播的结果计算
            out_put=tf.reshape(out_put,[-1,2*self.hidden_dim]) #就是一个维度变换
            pred=tf.matmul(out_put,W)+b #进行线性变换
            #s[1]是所选取的最大句子长度
            logits = tf.reshape(pred, [-1, s[1], self.num_tags])

            #CRF损失值的计算
            #transition_params是CRF的转换矩阵，会被自动计算出来
            #tag_indices：填入维度为[batch_size, max_seq_len]的矩阵，也就是Golden标签，注意这里的标签都是以索引方式表示的这个就是真实的标签序列了
            #sequence_lengths：维度为[batch_size]的向量，记录了每个序列的长度
            #inputs：unary potentials，也就是每个标签的预测概率值，这个值根据实际情况选择计算方法，CNN,RNN...都可以
            #crf_log_likelihood求的是CRF的损失值，牵扯到前向后向算法，会获得概率转移矩阵
            log_likelihood,transition_params=crf_log_likelihood(inputs=logits,tag_indices=self.labels,sequence_lengths=self.sequence_lengths)
            loss=-tf.reduce_mean(log_likelihood)
            return loss,transition_params,logits
