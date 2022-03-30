#encoding=utf-8
import pickle as pickle
import numpy as np
#用于字标注向索引的转换，因为tensorflow中的crf接收的数据是数字形式的索引
def tags2id(tags_list,tag2label):
    '''
    :param tags_list:
    :param tag2label:字标注向索引映射的字典
    :return:
    '''
    tags_label_list=[]
    for tags in tags_list:
        tags_label=[]
        for tag in tags:
            tags_label.append(tag2label[tag])
        tags_label_list.append(tags_label)
    print('final tags2id')
    return tags_label_list
#用于获得数据
def get_data(file_location):
    '''
    :param file_location: 文件的存放路径，注意文件中的字和字标注是空格隔开的
    :return: 两个大的List，前面的list里面一个个小list存放的是一个个句子，后面的list里面一个个小list存放的是一个个句子对应的标签
    '''
    sentences_list=[]
    tags_list=[]
    with open(file_location,'r',encoding='utf-8') as fr:
        sentence_list=[]
        tag_list=[]
        for line in fr.readlines():
            if line!='\n':
                [word,tag]=line.strip().split()
                sentence_list.append(word)
                tag_list.append(tag)
            else:
                sentences_list.append(sentence_list)
                tags_list.append(tag_list)
                sentence_list = []
                tag_list = []
    print('final get_data')
    return sentences_list,tags_list
#用于获得训练集中每个字对应的id，返回的是键为字值为id的一个字典
#注意这个dict只用于当前训练集，换训练集需要自己生成
def get_word_id(file_location):
    with open(file_location,'rb') as fr:
        word2id_dict=pickle.load(fr)
    print('final get_word_id')
    return word2id_dict
#用于初始化字向量，这里并没有通过word2vec获得，而是通过随机正太分布获得
def random_embedding(word2id_dict,embedding_size):
    '''
    :param word2id_dict: 用于获得总的字符个数
    :param embedding: 每个字的维度
    :return: 字向量组
    '''
    embedding_mat=np.random.uniform(-0.25,0.25,(len(word2id_dict),embedding_size))
    embedding_mat=np.array(embedding_mat).astype(np.float32)
    print('final random_embedding')
    return embedding_mat
#获得一个句子中每个字对应的索引
def sentence2id(sentences,word2id_dict):
    '''
    :param sentences: 所有句子
    :param word2id_dict: 记录字和字对应索引的字典
    :return: 包含所有句子中每个字索引的List
    '''
    sentences_id_list=[]
    for sentence in sentences:
        sentence_id_list=[]
        for word in sentence:
            if str(word).isdigit():
                word='<NUM>'
            elif ('\u0041' <= word <= '\u005a') or ('\u0061' <= word <= '\u007a'):
                word = '<ENG>'
            if word not in word2id_dict.keys():
                word = '<UNK>'
            sentence_id_list.append(word2id_dict[word])
        sentences_id_list.append(sentence_id_list)
    print('final sentence2id')
    return sentences_id_list
#对句子和标签都可进行填充,同时获得每个序列长度的列表
def padding_sentences(sentences_index,pad_mark=0):
    '''
    :param sentences_index: 每个句子各个字或者字标注对应的索引
    :param pad_mark: 用什么进行填充，默认为用零进行填充
    :return: 填充后的各个句子或标注的索引和序列长度列表
    '''
    sen_max_len=max(map(lambda x:len(x),sentences_index))
    sen_index_list,sen_len_list=[],[]
    for sen_index in sentences_index:
        sen_index=list(sen_index)
        new_sentence_index=sen_index[:sen_max_len]+[pad_mark]*max(sen_max_len-len(sen_index),0)
        sen_index_list.append(new_sentence_index)
        sen_len_list.append(min(len(sen_index),sen_max_len))
    return np.array(sen_index_list),np.array(sen_len_list)
# if __name__ == '__main__':
#     word2id=get_word_id('data/word2id.pkl')
#     for key,id in word2id.items():
#         if id==0:
#             print(key)
