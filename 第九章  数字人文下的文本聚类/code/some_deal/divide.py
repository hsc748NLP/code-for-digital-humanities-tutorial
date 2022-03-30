import re
import codecs
import numpy as np
import jieba
import jieba.analyse

data0 = []
data1 = []
data2 = []
data3 = []
data4 = []
data5 = []
data6 = []
data7 = []
data8 = []
data9 = []

def readtxt(path):
    with codecs.open(path, 'r', 'utf8') as f:
        line = f.readlines()
    return line


def divide(line, lables):
    for i, j in zip(lables, range(len(line))):
        line[j] = line[j].strip('\n')
        #res = [x.strip() for x in line[j] if x.strip() != '']
        if i == 0:
            data0.append(line[j])
        elif i == 1:
            data1.append(line[j])
        elif i == 2:
            data2.append(line[j])
        elif i == 3:
            data3.append(line[j])
        elif i == 4:
            data4.append(line[j])
        else:
            data5.append(line[j])


    #print(data0)

def extract_kw(data):
    #print(str(data))
    kw = jieba.analyse.extract_tags(str(data), topK=30, withWeight=False, allowPOS=())
    print(kw)


def write(data, path):
    with codecs.open(path, 'a', encoding='utf8') as f:
        for line in data:
            f.write(line+' '+'\n')
        f.write('\n')
    f.close()


def main():
    path="D:/我/非遗/cut_words_entity.txt"
    pathtxt="D:/我/非遗/title_info.txt"
    path_lables = "D:\我\非遗\高维聚类结果\\400-ns\\6类_label.txt"
    line = readtxt(pathtxt)
    #print(line)
    lables = np.loadtxt(path_lables)
    #print(lables)
    divide(line, lables)
    write(data0, "D:\我\非遗\高维聚类结果\\400-6\\类1.txt")
    write(data1, "D:\我\非遗\高维聚类结果\\400-6\\类2.txt")
    write(data2, "D:\我\非遗\高维聚类结果\\400-6\\类3.txt")
    write(data3, "D:\我\非遗\高维聚类结果\\400-6\\类4.txt")
    write(data4, "D:\我\非遗\高维聚类结果\\400-6\\类5.txt")
    write(data5, "D:\我\非遗\高维聚类结果\\400-6\\类6.txt")
    #write(data6, "D:\我\非遗\高维聚类结果\\400-10\\类7.txt")
    #write(data7, "D:\我\非遗\高维聚类结果\\400-10\\类8.txt")
    #write(data8, "D:\我\非遗\高维聚类结果\\400-10\\类9.txt")
    #write(data9, "D:\我\非遗\高维聚类结果\\400-10\\类10.txt")

main()