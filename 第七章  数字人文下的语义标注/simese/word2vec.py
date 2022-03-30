import nlpertools
from gensim.models import Word2Vec
import logging


# 设置当前打印日志的等级
logging.getLogger().setLevel(logging.INFO)


# 一个例子
def example(corpus):
    # 训练word2vec模型
    model = Word2Vec(corpus, sg=1, size=256, min_count=5, window=10)
    # 打印“罢”和“废”的语义相似度
    print(model.similarity('罢','废'))

# 训练并保存模型
def save_model(corpus):
    model = Word2Vec(corpus, sg=1, size=256, min_count=5, window=10)
    # 保存训练好的word2vec模型到位置："output/word2vec.model"
    model.save("output/word2vec.model")


# 加载模型并预测语义相似度
def predict(w1, w2):
    # 加载位置："output/word2vec.model" 的模型
    model = Word2Vec.load("output/word2vec.model")
    print(model.similarity(w1, w2))

if __name__ == '__main__':
    print('??')
    # 读取data/pairs_source.json文件的数据
    corpus = nlpertools.load_from_json('data/pairs_source.json')
    # 打印前三条数据
    logging.info(corpus[:3])
    # example(corpus)
    save_model(corpus)
    predict('罢','废')