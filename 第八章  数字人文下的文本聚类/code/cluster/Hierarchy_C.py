from itertools import cycle  ##python自带的迭代器模块

import jieba
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.decomposition import PCA


# 读取txt文档
def read_txt(path):
    f = open(path, 'r', encoding='UTF-8')
    lines = f.readlines()
    f.close()
    return lines


# onehot编码，返回np.array()
# Reference https://blog.csdn.net/Dorothy_Xue/article/details/84641417
def onehot(text):
    # 对原有文档用jieba分词，并建立字典
    data = []
    words = []
    for sentence in text:
        sentence = sentence.strip()
        seg_list = jieba.cut(sentence, cut_all=False)
        seg_list = '/'.join(seg_list)
        temp = seg_list.split('/')
        for word in temp:
            words.append(word)
        data.append(seg_list)
    dic = list(set(words))  # 去重

    # 手动onehot编码
    vector = []
    for i in range(0, len(data)):
        temp = []
        for j in range(0, len(dic)):
            if dic[j] in data[i].split('/'):
                temp.append(1)
            else:
                temp.append(0)
        temp = np.array(temp)
        vector.append(temp)
    length = len(vector)
    vector = np.array(vector)
    return vector


# 设置聚类函数，X是二维列表，绘制聚类示意图
# Reference https://www.cnblogs.com/lc1217/p/6963687.html
def Hierarchy(X):
    linkages = ['ward', 'average', 'complete']
    n_clusters_ = 6
    ac = AgglomerativeClustering(linkage=linkages[2], n_clusters=n_clusters_)
    # ac = DBSCAN(eps=0.1, min_samples=5)
    ##训练数据
    ac.fit(X)

    ##每个数据的分类
    lables = ac.labels_

    ##绘图
    plt.figure(1)
    plt.clf()

    colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
    for k, col in zip(range(n_clusters_), colors):
        ##根据lables中的值是否等于k，重新组成一个True、False的数组
        my_members = lables == k
        ##X[my_members, 0] 取出my_members对应位置为True的值的横坐标
        plt.plot(X[my_members, 0], X[my_members, 1], col + '.')

    plt.title('Estimated number of clusters: %d' % n_clusters_)
    plt.show()


def Mean_shift(X):
    # 产生随机数据的中心
    # centers = [[1, 1], [-1, -1], [1, -1]]
    # 产生的数据个数
    # n_samples=10000
    # 生产数据
    # X, _ = make_blobs(n_samples=n_samples, centers= centers, cluster_std=0.6,random_state =0)

    # 带宽，也就是以某个点为核心时的搜索半径
    bandwidth = estimate_bandwidth(X, quantile=0.2, n_samples=500)
    # 设置均值偏移函数
    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    # 训练数据
    ms.fit(X)
    # 每个点的标签
    labels = ms.labels_
    print(labels)
    # 簇中心的点的集合
    cluster_centers = ms.cluster_centers_
    # 总共的标签分类
    labels_unique = np.unique(labels)
    # 聚簇的个数，即分类的个数
    n_clusters_ = len(labels_unique)

    print("number of estimated clusters : %d" % n_clusters_)

    # 绘图
    plt.figure(1)
    plt.clf()

    colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
    for k, col in zip(range(n_clusters_), colors):
        # 根据lables中的值是否等于k，重新组成一个True、False的数组
        my_members = labels == k
        cluster_center = cluster_centers[k]
        # X[my_members, 0] 取出my_members对应位置为True的值的横坐标
        plt.plot(X[my_members, 0], X[my_members, 1], col + '.')
        plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col, markeredgecolor='k', markersize=14)
    plt.title('Estimated number of clusters: %d' % n_clusters_)
    plt.show()


def main():
    path = "./title_info.txt"
    text = read_txt(path)
    vector = onehot(text)
    # 降维 PCA/SVD
    pca = PCA(n_components=2)  # 降到2维
    pca.fit(vector)
    new_vector = pca.fit_transform(vector)
    # 层次聚类
    Hierarchy(new_vector)

    # 均值聚类
    # Mean_shift(new_vector)


if __name__ == '__main__':
    main()
