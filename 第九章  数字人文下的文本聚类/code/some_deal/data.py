import codecs
import xlrd
import jieba
import random
from sklearn.feature_extraction.text import TfidfVectorizer

def readxls(path, col):
    xl = xlrd.open_workbook(path)
    sheet = xl.sheets()[0]
    data = list(sheet.col_values(col))[1:]
    return data


def readtxt(path):
    with codecs.open(path, 'r', 'utf8') as f:
        line = f.readline()
        line.replace('\\u3000','')
        data = list(line)
    return data


def uni(title, info):
    uni_lis = []
    for i, j in zip(title, info):
        if i != '' and j != '':
            n = i+' '+j
            uni_lis.append(n)
    return uni_lis


def writetxt(path, txt):
    with codecs.open(path, 'a', 'utf-8') as f:
        for i in txt:
            f.write('\t'+str(i)+'\n')


def cutwords(data, stopwords):
    #分词
    word_lis = []
    for line in data:
        slist = jieba.cut(line, cut_all=False)
        output = " ".join(slist)
        for key in output.split(' '):
            if key not in stopwords:
                word_lis.append(key)
    return word_lis


def main():
    path_xls = ".\非遗国家级.xlsx"
    path_txt = ".\info.txt"
    path_stopword = ".\停用词.txt"
    title = readxls(path_xls, 0)
    info = readxls(path_xls, 8)
    stopwords = readtxt(path_stopword)   # 读取停用词
    data = uni(title, info)              # 标题和详细信息结合
    #random.shuffle(result)
    #train_list = data[:int(len(data) * 0.9)]
    #test_list = data[int(len(data) * 0.9):]
    writetxt(path_txt, data)             # 输出全部训练数据
    #train_data = cutwords(train_list, stopwords)            # 分词
    #test_data = cutwords(test_list, stopwords)
    #writetxt("./cut_words.txt", cutwords(data, stopwords))      # 分词


if __name__=="__main__":
    main()
