import codecs

import xlrd
from pylab import *

mpl.rcParams['font.sans-serif'] = ['SimHei']


# 读取类别txt
def readtxt(path):
    with codecs.open(path, 'r', 'utf8') as f:
        line = f.readlines()
    return line

# 读取excel 非遗信息全
def readxls(path,col):
    xl=xlrd.open_workbook(path)
    sheet=xl.sheets()[0]
    data=list(sheet.col_values(col))[1:]
    return data

# 读类别里标题信息
def ex_title(data):
    ti=[]
    temp=[]
    for line in data:
        l = line.split(' ')
        ti.append(l[0])
    #print(ti)
    return ti


# 去除多余空字符
def corpus(data):
    final = []
    for line in data:
        l = line.strip(' ').split(' ')
        res = [x.strip() for x in l if x.strip() != '\xa0' or x.strip() != '\u3000' or x.strip() !='\ue81b'\
               or x.strip() !=' ']
        cor = " ".join(res)
        final.append(cor)
    #print(final)
    return final


# 计算各类数量
def calculate(ti,dic):
    sum=0
    list=[]
    dic1 = {'民间文学':0,'传统音乐':0,'传统舞蹈':0,'传统戏剧':0,'曲艺':0,'传统体育、游艺与杂技':0,\
          '传统美术':0,'传统技艺':0,'传统医药':0,'民俗':0}
    #print(dic1)
    for i in dic:
        for t in ti:
            if t == i:
                list.append(t)
                dic1[dic[i]]+=1
            #else:
             #   print(t)
    for v in dic1.values():
        sum=sum+v
    for i in ti:
        if i not in list:
            print(i)
    print(sum)
    print(dic1)
    #print(list)
    #print(len(list))
    return dic1

def writetxt(path,txt):
    with codecs.open(path,'a','utf-8') as f:
        for i in txt:
            f.write(str(i)+'\n')


def main():
    path1="D:\我\非遗\高维聚类结果\\400-10\\类2.txt"
    #path_cla = "D:\我\非遗\高维聚类结果\标题加类别.txt"
    path_xls = "D:\我\非遗\非遗初始语料\非遗国家级.xlsx"
    #path_list = "D:\我\非遗\高维聚类结果\\400维六类数据\类4结果.txt"
    #ti_cla = readtxt(path_cla)
    #print(ti_cla)
    title = readxls(path_xls, 0)
    classes = readxls(path_xls, 4)
    final_title=corpus(title)    # 去除奇异字符
    dic=dict(zip(final_title, corpus(classes)))   # 标题：类别 字典
    #print(dic)

    data = readtxt(path1)
    ti = corpus(ex_title(data))
    #print(ti)
    dic_num=calculate(ti,dic)

    #writetxt(path_list,calculate(ti,dic))
 

if __name__ == '__main__':
    main()