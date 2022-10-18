# -*- coding: utf-8 -*-
# @Time    : 2021/7/24 9:07
# @Author  : YangFan

import random

import numpy as np
import torch
import torch.nn.functional as F
from earlystopping import EarlyStopping
from gensim.models import Word2Vec
from sklearn import metrics
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
from torch import nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import Dataset, DataLoader

w2v_model = Word2Vec.load('vec_model/w2v.model')
w2v_model_np=np.load('vec_model/w2v.model.wv.vectors.npy')

labels_dict={'地区':0,'电脑网络':1,'电子数码':2,'健康生活':3,'教育科学':4}
#这个要根据非遗的数据类别改一下，类别在excel文件里面找

'''
代码思路：
1. 读取数据，找到一个合适的max_seq_length   #test 平均长度192.6953 ;train 213.929142
2. 处理数据：content(words list) to ids;label to id;pad_sequence
    注意：因为后面打算用word2vec的预训练词向量，所以这里对word取索引时，需要通过index = model.wv.key_to_index[word]来获取，而不是单纯地赋值0123...
    pad_sequence补0，网络中用pach_padded_sequence消除补充的这个数据带来的影响
    传入网络数据长度：从长到短；同时输入content_list和length_list
3. 编写网络结构：三层：embedding层、lstm层、线性层；注意参数传递
4. 设置optimizier、loss_func
5. 设置一个取批次数据的函数def get_batch_data
6. 开始训练：
    epoch(batch（train（cal loss（...
7. 测试
'''

def setup_seed(seed):
    torch.manual_seed(seed) #为CPU中设置种子，生成随机数
    #torch.cuda.manual_seed_all(seed)   #为所有GPU设置种子，生成随机数
    #torch.cuda.manual_seed(seed)   #为特定GPU设置种子，生成随机数
    np.random.seed(seed)
    random.seed(seed)

class ModelRNN(nn.Module):
    def __init__(self,emb_dim,hidden_size,num_layers,num_classes):
        super(ModelRNN, self).__init__()
        embedding=w2v_model_np
        embedding=torch.tensor(embedding,dtype=torch.float32)
        self.embedding=nn.Embedding.from_pretrained(embedding,freeze=False)
        self.embedding.requires_grad = False
        self.lstm=nn.LSTM(emb_dim,hidden_size,num_layers,batch_first=True,bidirectional=True,dropout=0.5)
        self.classifier=nn.Linear(hidden_size*2,num_classes)

    def forward(self,x,lens):
        x=x.type(torch.LongTensor)
        embed=self.embedding(x)
        packed_input=pack_padded_sequence(embed,lens,batch_first=True)
        packed_out,(hidden,cell)=self.lstm(packed_input)     #lstm_out:(batch,seq,hidden*direction) hidden(4,64,128)
        # 连接最后的正向和反向隐状态 参考：https://blog.csdn.net/qq_43238260/article/details/112425230
        #hidden is stacked [forward_1, backward_1, forward_2, backward_2, ...]
        #hidden [-2, :, : ] is the last of the forwards RNN  (64,128)
        #hidden [-1, :, : ] is the last of the backwards RNN (64,128)
        hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1) #(64,256)
        out=self.classifier(hidden)
        return out

class MyDataset(Dataset):
    def __init__(self,content_and_label):
        self.x_data=content_and_label[0]
        self.y_data=content_and_label[-1]
        self.len=len(content_and_label[0])

    def __getitem__(self, index):
        x_new=[w2v_model.wv.key_to_index[word] for word in self.x_data[index]]
        if len(x_new)<200:  #截断
            x_new=x_new
        else:
            x_new=x_new[:200]
        y_new=labels_dict[self.y_data[index]]

        #print('len(x_new)',len(x_new))
        return {'x':x_new,'y':y_new}

    def __len__(self):
        return self.len

def my_collate_fn(x):
    lengths = np.array([len(term['x']) for term in x])
    sorted_index = np.argsort(-lengths) #从大到小排列，取对应索引

    # build reverse index map to reconstruct the original order
    reverse_sorted_index = np.zeros(len(sorted_index), dtype=int)
    for i, j in enumerate(sorted_index):
        reverse_sorted_index[j] = i
    lengths = lengths[sorted_index]

    # control the maximum length of LSTM
    max_len = lengths[0]
    batch_size = len(x)
    input_tensor = torch.FloatTensor(batch_size, int(max_len)).zero_()   #(batch_size*max_len) 0矩阵
    output_tensor = torch.FloatTensor(batch_size, 1).zero_()

    for i, index in enumerate(sorted_index):
        input_tensor[i][:lengths[i]] = torch.FloatTensor(x[index]['x'])
        output_tensor[i][:lengths[i]] = x[index]['y']

    input_tensor = Variable(input_tensor,requires_grad=True)
    output_tensor = Variable(output_tensor,requires_grad=True)

    return {'x': input_tensor,'y':output_tensor,'lengths': lengths}

def read_txt(path):
    with open(path, 'r', encoding='utf-8') as f:
        lines=[line.strip() for line in f.readlines()]
    random.shuffle(lines)
    x=[_.split("\t")[1].split() for _ in lines]
    y=[_.split("\t")[0] for _ in lines]
    return x,y

def main():
    train_iter = DataLoader(MyDataset(read_txt('train.txt')),
                            batch_size=64,
                            shuffle=True,
                            collate_fn=my_collate_fn)

    val_iter = DataLoader(MyDataset(read_txt('test.txt')),
                            batch_size=64,
                            shuffle=True,
                            collate_fn=my_collate_fn)

    model=ModelRNN(emb_dim=300,hidden_size=128,num_layers=2,num_classes=5)
    loss_func=nn.CrossEntropyLoss()
    optimizer=torch.optim.Adam(model.parameters(),lr=1e-3)

    early_stopping = EarlyStopping(verbose=True)

    train_losses = []
    valid_losses = []
    valid_acces=[]
    n_epochs=100

    for epoch in range(n_epochs):
        model.train()
        for i, batch in enumerate(train_iter):
            # forward
            output=model(batch['x'],batch['lengths'])
            labels=batch['y'].squeeze(1).type(torch.LongTensor)
            loss=loss_func(output,labels)
            train_losses.append(loss.item())
            #backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        for batch in val_iter:
            output=model(batch['x'],batch['lengths'])
            labels=batch['y'].squeeze(1).type(torch.LongTensor)
            loss=loss_func(output,labels)
            valid_losses.append(loss.item())

            softmax_output = F.softmax(output, dim=1)
            y_pre = torch.argmax(softmax_output, dim=1)  # argmax对softmax的输出返回概率最大的类别,dim=1表示在行上求
            y_true = labels
            acc=metrics.accuracy_score(y_true,y_pre)
            valid_acces.append(acc)

        train_loss = np.average(train_losses)
        valid_loss = np.average(valid_losses)
        valid_acc=np.average(valid_acces)
        epoch_len = len(str(n_epochs))
        print_msg = (f'[{epoch:>{epoch_len}}/{n_epochs:>{epoch_len}}] ' +
                     f'train_loss: {train_loss:.5f} ' +
                     f'valid_loss: {valid_loss:.5f}'+
                     f'valid_acc: {valid_acc}')

        print(print_msg)

        # clear lists to track next epoch
        train_losses = []
        valid_losses = []

        # early_stopping needs the validation loss to check if it has decresed,
        # and if it has, it will make a checkpoint of the current model
        early_stopping(valid_loss, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break

    torch.save(model, 'rnn_model.pkl')
    # load the last checkpoint with the best model
    test()

def test():
    model=torch.load('rnn_model.pkl')
    test_iter = DataLoader(MyDataset(read_txt('test.txt')),
                            batch_size=1500,
                            shuffle=True,
                            collate_fn=my_collate_fn)
    for batch in test_iter:
        output = model(batch['x'], batch['lengths'])
        labels = batch['y'].squeeze(1).type(torch.LongTensor)

        softmax_output=F.softmax(output,dim=1)
        y_pre = torch.argmax(softmax_output, dim=1)     #argmax对softmax的输出返回概率最大的类别,dim=1表示在行上求
        y_true=labels
        p = precision_score(y_true, y_pre, average='macro')
        r = recall_score(y_true, y_pre, average='macro')
        f = f1_score(y_true, y_pre, average='macro')
        report = classification_report(y_true, y_pre)
        print('p=', '%.2f%%' % (p * 100), 'r=', '%.2f%%' % (r * 100), 'f=', '%.2f%%' % (f * 100))
        print(report)

if __name__ == '__main__':
    setup_seed(1)
    main()