import codecs

import xlrd


def readxls(path,col):
    xl=xlrd.open_workbook(path)
    sheet=xl.sheets()[0]
    data=list(sheet.col_values(col))[1:]
    return data

def readtxt(path):
    with codecs.open(path,'r','utf8') as f:
        line=f.readline()
        data=list(line)
    return data

#union
def uni(title,info):
    uni_lis=[]
    n=''
    for i,j in zip(title,info):
        n=i+' '+j
        uni_lis.append(n)
    return uni_lis

def writetxt(path,txt):
    with codecs.open(path,'a','utf-8') as f:
        for i in txt:
            f.write(str(i)+'\n')

def main():
    path_xls=".\非遗国家级.xlsx"
    path_txt=".\\title_info.txt"
    path_stopword=".\停用词.txt"
    #path_txt=r'C:\Users\lenovo\Desktop\非遗\title_info_onehot.txt'
    title=readxls(path_xls,0)
    info=readxls(path_xls,8)
    classes = readxls(path_xls,4)
    stopwords=readtxt(path_stopword)
    #data=uni(title,info)
    #ti_class = uni(title,classes)
    #dic=dict.fromkeys(title,classes)
    print(dict(zip(title,classes)))
    #writetxt(path_txt, data)
    #writetxt("D:\我\非遗\高维聚类结果\标题加类别.txt",ti_class)
    ''''#分词
    word_lis=[]
    for line in data:
        slist = jieba.cut(line, cut_all=False)
        output = " ".join(slist)
        for key in output.split(' '):
            if key not in stopwords:
                word_lis.append(key)

    # 参考官方文档运用sklearn.feature_extraction.text.TfidfVectorizer,将corpus文本转换为tfidf值的svm向量
    tfidfvec = TfidfVectorizer()
    cop_tfidf = tfidfvec.fit_transform(word_lis)
    weight = cop_tfidf.toarray()


    #降维
    X = np.array(weight)  # 导入数据
    pca = PCA(n_components=2)  # 降到2维
    pca.fit(X)  # 训练
    newX = pca.fit_transform(X)  # 降维后的数据
    # PCA(copy=True, n_components=2, whiten=False)
    # print(pca.explained_variance_ratio_)  #输出贡献率
    print(newX)

    #层次聚类
    X = newX
    ##设置分层聚类函数
    linkages = ['ward', 'average', 'complete']
    n_clusters_ = 6
    ac = AgglomerativeClustering(linkage=linkages[2], n_clusters=n_clusters_)
    ##训练数据
    ac.fit(X)

    ##每个数据的分类
    lables = ac.labels_

    ##绘图
    plt.figure(1)
    plt.clf()

    colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
    for k, col in zip(range(n_clusters_), colors):
        # 根据lables中的值是否等于k，重新组成一个True、False的数组
        my_members = lables == k
        ##X[my_members, 0] 取出my_members对应位置为True的值的横坐标
        plt.plot(X[my_members, 0], X[my_members, 1], col + '.')

    plt.title('Estimated number of clusters: %d' % n_clusters_)
    plt.show()'''

if __name__=="__main__":
    main()
