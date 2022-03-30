from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
import numpy as np
from gensim.models.keyedvectors import KeyedVectors
    # 训练word2vec模型 参数说明：
    # sentences: 包含句子的list，或迭代器
    # size:      词向量的维数，size越大需要越多的训练数据，同时能得到更好的模型
    # alpha:     初始学习速率，随着训练过程递减，最后降到 min_alpha
    # window:    上下文窗口大小，即预测当前这个词的时候最多使用距离为window大小的词
    # max_vocab_size: 词表大小，如果实际词的数量超过了这个值，过滤那些频率低的
    # workers:   并行度
    # iter:      训练轮数
    # sg=0 cbow，sg=1 skip-gram
    # hs=0 negative sampling, hs=1 hierarchy
    #sentences = word2vec.Text8Corpus(r'D:\我\非遗\cut_words_entity')
    #model.save('heritage.model')  保存模型
    # https://blog.csdn.net/laobai1015/article/details/86540813 参数解释

def build_vec(list_sentence, model):
    list_vec_sentence = []
    for sentence in list_sentence:     # 每个sentence为一个list

        if len(sentence) > 1000:
            arrlists = [model[word] for word in sentence[0:1000]]
            x = np.average(arrlists, axis=0)
        else:
            arrlists = [model[word] for word in sentence]
            x = np.average(arrlists, axis=0)
        list_vec_sentence.append(x)
    return list_vec_sentence


def main():
    path = r'D:\我\非遗\cut_words_entity.txt'
    sentences = LineSentence(path)
    model = Word2Vec(sentences, sg=0, size=100, min_count=0)  # sg=0 cbow，hs=0默认 negative sampling
    # model.wv.save_word2vec_format('heritage_word_100.bin', binary=True)
    model.save('heritage_ns_100.model')
    vec_sentence = build_vec(sentences, model)
    #print(vec_sentence)
    list_vec_sentence = []
    for s in sentences:
        for word in s:
            arrlists = [model[word]]
            x = np.average(arrlists, axis=0)
        list_vec_sentence.append(x)
    np.savetxt("w2v_sentence_vec_100D_ns.txt", list_vec_sentence)
    '''

    model = Word2Vec.load('./heritage.model')
    word = model.most_similar("赛龙舟")
    print(word)
    #print(model[''])
'''

if __name__ == '__main__':
    main()