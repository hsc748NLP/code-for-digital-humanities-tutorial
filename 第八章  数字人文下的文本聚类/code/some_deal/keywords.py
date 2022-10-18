import codecs


def readtxt(path):
    with codecs.open(path,'r',encoding='utf-8') as f:
        line=f.readlines()  #line为一个列表 一行一个元素
        line = [x.strip('\r\n') for x in line]
        print(line)
        return line

def trans(data):
    words=[]
    for line in data:
        line = line.split(',')
        with codecs.open(r'D:\我\非遗\高维聚类结果\400维十类\掉包keywords_15.txt', 'a', 'utf-8')as f:
            for i in line:
                f.write(str(i)+'\n')
            f.write('\n')




def main():
    path6 = r'D:\我\非遗\高维聚类结果\400维六类数据\掉包keywords_15.txt'
    path8 = r'D:\我\非遗\高维聚类结果\400维八类\掉包keywords_15.txt'
    path10 = r'D:\我\非遗\高维聚类结果\400维十类\掉包keywords_15.txt'
    data = readtxt(path10)
    trans(data)



if __name__ == '__main__':
    main()