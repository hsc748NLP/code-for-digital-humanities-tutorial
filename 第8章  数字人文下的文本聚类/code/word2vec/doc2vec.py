import codecs

import gensim
import numpy as np
from gensim.models.doc2vec import Doc2Vec

TaggededDocument = gensim.models.doc2vec.TaggedDocument


def readtxt(path):
    data = []
    with codecs.open(path,'r',encoding='utf-8') as f:
        doc = f.readlines()
        #for line in f.readlines():
            #line = line.strip('\n')
            #data.append(line)
        return doc


def train(x_train):
    # D2V参数解释：
    # min_count：忽略所有单词中单词频率小于这个值的单词。
    # window：窗口的尺寸。（句子中当前和预测单词之间的最大距离）
    # size:特征向量的维度
    # sample：高频词汇的随机降采样的配置阈值，默认为1e-3，范围是(0,1e-5)。
    # negative: 如果>0,则会采用negativesampling，用于设置多少个noise words（一般是5-20）。默认值是5。
    # workers：用于控制训练的并行数。
    model_dm = Doc2Vec(x_train, min_count=1, window=3, vector_size=160, sample=1e-3, negative=5, workers=4)
    # total_examples：统计句子数
    # epochs：在语料库上的迭代次数(epochs)。
    model_dm.train(x_train, total_examples=model_dm.corpus_count, epochs=70)
    model_dm.save('d2v_heritage_160.model')

    return model_dm

def test():
    model_dm = Doc2Vec.load("model/model_dm_wangyi")
    test_text = ['《', '舞林', '争霸' '》', '十强' '出炉', '复活', '舞者', '澳门', '踢馆']
    inferred_vector_dm = model_dm.infer_vector(test_text)
    print(inferred_vector_dm)
    sims = model_dm.docvecs.most_similar([inferred_vector_dm], topn=10)

    return sims

    '''sims = test()
    for count, sim in sims:
        sentence = x_train[count]
        words = ''
        for word in sentence[0]:
            words = words + word + ' '
        print(words, sim, len(sentence[0]))
     '''


def get_dataset():
    with open(r'D:\我\非遗\cut_words_entity.txt', 'r', encoding='utf8') as f:
        docs = f.readlines()
        print(len(docs))

    x_train = []
    # y = np.concatenate(np.ones(len(docs)))
    for i, text in enumerate(docs):
        word_list = text.split(' ')
        l = len(word_list)
        word_list[l - 1] = word_list[l - 1].strip()
        document = TaggededDocument(word_list, tags=[i])
        x_train.append(document)

    return x_train


def main():
    path = r'D:\我\非遗\cut_words_entity.txt'
    x_train = get_dataset()
    #print(x_train)
    train(x_train)
    #test()
    data = []
    model = Doc2Vec.load('d2v_heritage_160.model')
    for i in range(2690):
        data.append(model.docvecs[i])
    X = np.array(data)
    np.savetxt("./d2v_heritage_160.txt", X)
    # print(model.docvecs[10])


main()