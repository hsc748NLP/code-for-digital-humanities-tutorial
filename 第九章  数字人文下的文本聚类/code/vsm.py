import time
import codecs
import shutil
from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
import os
import math




path = r"C:\Users\lenovo\Desktop\信息检索系统\文摘"

def readtxt(path):
    cate = [path +'\\'+ x for x in os.listdir(path)]
    print(cate)
    data = []
    for f in cate:
        with codecs.open(f, 'r', 'utf8') as f:
            line = f.readlines()
            data.append(line)
    data_final=[]
    for i in range(len(data)):
        ll = ''
        for l in data[i]:
            l = l.replace('\r\n','').replace('.','').replace(',','').replace('"','').replace('--','').replace('\'','')
            ll = ll+l
        data_final.append(ll)
    print(data_final)
    return data_final


if __name__ == "__main__":
    #corpus = []  # 文档预料 空格连接
    corpus = readtxt(path)
    # 读取预料 一行预料为一个文档
    #path1 = "D:\我\非遗\高维聚类结果\\350_类11.txt"
    #for line in open(path1, 'r', encoding="utf-8").readlines():
    #    corpus.append(line.strip())
        # print corpus
    #time.sleep(5)

    # 将文本中的词语转换为词频矩阵 矩阵元素a[i][j] 表示j词在i类文本下的词频
    vectorizer = CountVectorizer(
    min_df=0,
    token_pattern=r"\b\w+\b"
    )

    # 该类会统计每个词语的tf-idf权值
    transformer = TfidfTransformer()

    # 第一个fit_transform是计算tf-idf 第二个fit_transform是将文本转为词频矩阵
    tfidf = transformer.fit_transform(vectorizer.fit_transform(corpus))

    # 获取词袋模型中的所有词语
    word = vectorizer.get_feature_names()
    print(len(word))
    # 将tf-idf矩阵抽取出来，元素w[i][j]表示j词在i类文本中的tf-idf权重
    weight = tfidf.toarray()
    print(weight)
    #path2 = "./words_tfidf2.txt"
    #result = codecs.open(path2, 'w', 'utf-8')
    #for j in range(len(word)):
     #   result.write(word[j] + ' ')
    #result.write('\r\n\r\n')
    # 打印每类文本的tf-idf词语权重，第一个for遍历所有文本，第二个for便利某一类文本下的词语权重
    sum = 0
    sq1 = 0
    sq2 = 0
    for i in range(len(weight[0])):
        sum += weight[0][i] * weight[1][i]
        sq1 += pow(weight[0][i], 2)
        sq2 += pow(weight[1][i], 2)
    try:
        result = round(float(sum) / (math.sqrt(sq1) * math.sqrt(sq2)), 2)
    except ZeroDivisionError:
        result = 0.0
    print(result)

    #result.close()

