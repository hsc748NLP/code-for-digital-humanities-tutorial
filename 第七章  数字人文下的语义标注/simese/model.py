# 引入所需要的包
import torch
import torch.nn as nn
from torch.nn.modules import dropout
import torch.nn.functional as F

'''
基于孪生网络的语义相似度计算模型
'''
# 定义网络结构类
class SiameseNetwork(nn.Module):
    # 初始化
    def __init__(self, vocab_size=100, embedding_dim=100, gru_hidden_size=100, gru_layer_num=2, gru_num_directions=1, dropout_prob=0.5):
        super(SiameseNetwork, self).__init__()
        # 定义将词语向量化的Embedding
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # 定义编码层，选择GPU模型
        self.gru = nn.GRU(embedding_dim, gru_hidden_size, gru_layer_num) # , bidirectional=True)
        # 定义全连接层，将拼接后的两个向量转变为维度为1的向量
        self.linear = nn.Linear(2 * gru_num_directions * gru_hidden_size, 1)
        # 定义sigmoid层
        self.sigmoid = nn.Sigmoid()
        # 定义dropout层，用于防止过拟合
        self.dp = nn.Dropout(dropout_prob)

    # 因为是孪生网络，两次输入用同一个编码层编码，这是一次编码过程。
    def forward_once(self, sentence, dx):
        # 将输入的句子中的每个词语向量化
        embedding = self.embedding(sentence) # embedding.shape=(batch, seq_len, embedding_dim)
        # 将向量维度转换，方便后续运算 (batch, seq_len, embedding_dim) =>(seq_len,batch , embedding_dim)
        embedding_permute = embedding.permute(1, 0, 2) # shape=(seq_len,batch , embedding_dim)
        # 用GRU编码句子中的每个词，结果为gru_states，h_n是整个句子的编码，用不到
        gru_states, h_n = self.gru(embedding_permute) # (seq_len, batch, gru_num_directions * gru_hidden_size)
        # 将向量维度转换，方便后续运算
        gru_states_permute = gru_states.permute(1, 2, 0) # (batch, gru_num_directions * gru_hidden_size, seq_len)
        # 取出需要判断语义关系的词汇的向量表示
        current_step_states = torch.matmul(gru_states_permute, dx).squeeze(-1) # (batch, gru_num_directions * gru_hidden_size)
        # 返回需要判断语义关系的词汇的向量表示
        return current_step_states # (batch, gru_num_directions * gru_hidden_size)
    
    # 模型的forward过程
    def forward(self, sentence_a, sentence_b, idx, jdx):
        # 用sentence_a和idx 经过一次编码，返回古汉语词汇A的编码，返回的维度在每行代码的后面有注释
        output1 = self.forward_once(sentence_a, idx.unsqueeze(-1)) # (batch, gru_num_directions * gru_hidden_size)
        # 用sentence_b和jdx 经过一次编码，返回古汉语词汇B的编码
        output2 = self.forward_once(sentence_b, jdx.unsqueeze(-1))
        # 拼接古汉语词汇A和古汉语词汇B的两个编码表示
        cat_hidden = torch.cat((output1, output2), 1) # (batch, 2 * gru_num_directions * gru_hidden_size)
        # 通过dropout随机遮掩部分参数
        cat_hidden = self.dp(cat_hidden)
        # 经过全连接层，返回一个表示两个词汇相似度的logits
        final_hidden = self.linear(cat_hidden)
        # 将这个logits归一化到0-1
        output = self.sigmoid(final_hidden) # (batch, 1)
        # 将输出的维度变为(batch,)
        output = output.squeeze(dim=-1) # shape=(batch,)
        # 返回两个词汇的语义相似度
        return output

