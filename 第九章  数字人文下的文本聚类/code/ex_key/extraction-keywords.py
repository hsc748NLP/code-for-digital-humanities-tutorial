from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import numpy
import pandas
import codecs

def extract(corpus):
    '''corpus = []  # 文档预料 空格连接

    # 读取预料 一行预料为一个文档
    path1 = "D:\我\非遗\高维聚类结果\\350_类11.txt"
    for line in open(path1, 'r', encoding="utf-8").readlines():
        corpus.append(line.strip())
    #print(corpus)'''
    corpus1 = ["我 来到 北京 清华大学",  # 第一类文本切词后的结果，词之间以空格隔开
              "他 来到 了 网易 杭研 大厦",  # 第二类文本的切词结果
              "小明 硕士 毕业 与 中国 科学院",  # 第三类文本的切词结果
              "我 爱 北京 天安门"]

    contents = [
        '我 是 中国 人。',
        '你 是 美国 人。',
        '他 叫 什么 名字？',
        '她 是 谁 啊？'
    ]
    countVectorizer = CountVectorizer(
    '''min_df=0,
    token_pattern=r"\b\w+\b"'''
    )  # 增加了min_df=0参数，保留最小长度为0的分词，和token_pattern,设置分词的正则表达式。
    textVector = countVectorizer.fit_transform(corpus)
    transformer = TfidfTransformer(sublinear_tf=True)  # 该类会统计每个词语的tf-idf权值
    tfidf = transformer.fit_transform(textVector)  # .fit_transform()方法得到tf-idf矩阵
    weight = tfidf.toarray()
    #print(weight)
    word = countVectorizer.get_feature_names()
    #print(word)
    sort = numpy.argsort(weight, axis=1)[:, -10:]  # 对tf-idf矩阵每行的值进行排序，输出对应索引，并取每行前五，得到sort,格式为numpy.ndarray
    keywords = pandas.Index(word)[sort].values
    tagDF = pandas.DataFrame({
        'tag1': keywords[:, 0],  # 提取第一行，得到包含所有文档的第1个关键词的数组
        'tag2': keywords[:, 1],  # 提取第二行，得到包含所有文档的第2个关键词的数组
        'tag3': keywords[:, 2],
        'tag4': keywords[:, 3],
        'tag5': keywords[:, 4],
        'tag6': keywords[:, 5],
        'tag7': keywords[:, 6],
        'tag8': keywords[:, 7],
        'tag9': keywords[:, 8],
        'tag10': keywords[:, 9],
    })
    tagDF.to_csv("D:\我\非遗\高维聚类结果\\400-10\\掉包keywords_10.txt",header=False,index=False)
    print(tagDF)

def read(path):
    with codecs.open(path, 'r', 'utf8') as f:
        line = f.readlines()
    return line


def corpus(data):
    final = []
    for line in data:
        l = line.split(' ')
        res = [x.strip() for x in l if x.strip() != '']
        cor = " ".join(res)
        final.append(cor)
    #print(final[1])
    return final


def main():
    path = "D:\我\非遗\高维聚类结果\\400-10\\总.txt"
    data = read(path)
    final = corpus(data)
    #print(final)
    extract(final)
    #print(data)


if __name__ == "__main__":
    main()