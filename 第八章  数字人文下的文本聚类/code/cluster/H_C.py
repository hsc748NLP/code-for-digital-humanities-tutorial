from itertools import cycle  # python自带的迭代器模块

import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score


def H_C(X, n_clusters_):
    linkages = ['ward', 'average', 'complete']
    ac = AgglomerativeClustering(linkage=linkages[2], n_clusters=n_clusters_)
    # 训练数据
    ac.fit(X)
    # 每个数据的分类
    lables = ac.labels_
    print(lables[1])
   #print(np.array(lables))
    #np.savetxt("D:\我\非遗\高维聚类结果\\150-lables.txt", lables)
    # 绘图
    plt.figure(1)
    plt.clf()
    colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
    for k, col in zip(range(n_clusters_), colors):
        # 根据lables中的值是否等于k，重新组成一个True、False的数组
        my_members = lables == k
        # X[my_members, 0] 取出my_members对应位置为True的值的横坐标
        plt.plot(X[my_members, 0], X[my_members, 1], col + '.')

    plt.title('Estimated number of clusters: %d' % n_clusters_)
    plt.show()
    '''with open("D:\我\非遗\高维聚类结果\\150-lables.txt",'r',encoding="utf-8") as f:
        for i in lables:
            f.write(i+'\n')
    f.close()'''
    return lables

def Sil(X,Y):
    Scores = []  # 存放轮廓系数
    data=[6,8,10,13,16]
    for k in data:
        ac = AgglomerativeClustering(linkage='complete', n_clusters=k)
        ac.fit(X)
        Scores.append(silhouette_score(Y, ac.labels_, metric='euclidean'))  # euclidean 欧氏距离
    plt.title('Clustering Silhouette coefficient index-HC-w2v')
    x = data
    plt.xlabel("clusters")
    plt.ylabel("Silhouette coefficient")
    plt.plot(x, Scores, 'o-')
    for x, y in zip(x, Scores):
        plt.text(x, y + 0.001, '%.4f' % y, ha='center', va='bottom', fontsize=9)
    plt.show()


def CH(X):
    CH = []
    for k in range(4, 11):
        ac = AgglomerativeClustering(linkage='complete', n_clusters=k).fit(X)
        labels = ac.labels_
        CH.append(metrics.calinski_harabasz_score(X, labels))
    plt.title('Clustering calinski_harabasz_score index-HC-w2v')
    x = range(4, 11)
    plt.xlabel("clusters")
    plt.ylabel("calinski_harabasz_score")
    plt.plot(x, CH, 'o-')
    for x, y in zip(x, CH):
        plt.text(x, y + 0.001, '%.4f' % y, ha='center', va='bottom', fontsize=9)
    plt.show()



def main():
    X = np.loadtxt("tfidf_data.txt")     #only 结巴
    X1 = np.loadtxt("tfidf_data2.txt")     # jieba实体
    #X_d2v = np.loadtxt("./Word2vector/d2v_heritage_160_2d.txt")   #d oc2vec_sentence_size=160-2维
    X_w2v_50 = np.loadtxt("./Word2vector/w2v_sentence_vec_50D.txt")
    X_w2v_50_2d = np.loadtxt("./Word2vector/w2v_sentence_vec_50D-2d.txt")  # word2vec_sentence_size=50-2维
    X_w2v_100 = np.loadtxt("./Word2vector/w2v_sentence_vec_100D.txt")  # word2vec_sentence_size=100
    X_w2v_100_2d = np.loadtxt("./Word2vector/w2v_sentence_vec_100d-2d.txt")
    X_w2v_150 = np.loadtxt("./Word2vector/w2v_sentence_vec_150D.txt")
    X_w2v_150_2d = np.loadtxt("./Word2vector/w2v_sentence_vec_150D-2d.txt")  # word2vec_sentence_size=150-2维

    X_w2v_180 = np.loadtxt("./Word2vector/w2v_sentence_vec_180D-2d.txt")  # word2vec_sentence_size=180-2维
    X_w2v_200 = np.loadtxt("./Word2vector/w2v_sentence_vec_200d.txt")  # word2vec_sentence_size=200
    X_w2v_200_2d = np.loadtxt("./Word2vector/w2v_sentence_vec_200d-2d.txt")  # word2vec_sentence_size=200
    X_w2v_250 = np.loadtxt("./Word2vector/w2v_sentence_vec_250D.txt")  # word2vec_sentence_size=250
    X_w2v_250_2d = np.loadtxt("./Word2vector/w2v_sentence_vec_250D-2d.txt")  # word2vec_sentence_size=250
    X_w2v_300 = np.loadtxt("./Word2vector/w2v_sentence_vec_300D.txt")  # word2vec_sentence_size=300
    #X_w2v_300_2d = np.loadtxt("./Word2vector/w2v_sentence_vec_300D-2d.txt")  # word2vec_sentence_size=300
    X_w2v_350 = np.loadtxt("./Word2vector/w2v_sentence_vec_350D.txt")  # word2vec_sentence_size=300
    X_w2v_400 = np.loadtxt("./Word2vector/w2v_sentence_vec_400D.txt")  # word2vec_sentence_size=300
    lables = H_C(X_w2v_150, 6)
    test_stat = {}
    l=lables.tolist()
    print(l)
    for i in set(l):
        test_stat[i] = l.count(i)
    print(test_stat)
    #Sil(X)              # jieba分词
    #Sil(X_w2v_100_2d,X_w2v_100)             # 基于自定义词典jieba分词
    Sil(X_w2v_50, X_w2v_50)
    Sil(X_w2v_100, X_w2v_100)
    Sil(X_w2v_150, X_w2v_150)
    Sil(X_w2v_200, X_w2v_200)
    Sil(X_w2v_250, X_w2v_250)
    Sil(X_w2v_300, X_w2v_300)
    Sil(X_w2v_350, X_w2v_350)
    Sil(X_w2v_400, X_w2v_400)
    #CH(X_w2v_180)


if __name__ == '__main__':
    main()