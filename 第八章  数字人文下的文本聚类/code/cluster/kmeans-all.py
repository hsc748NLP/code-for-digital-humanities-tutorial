import numpy as np
from scipy.spatial.distance import cdist
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def kmeans(X):
    K = range(2, 11)
    meandistortions = []
    Scores = []  # 存放轮廓系数
    CH=[]
    all = []
    '''kmeans = KMeans(n_clusters=8)
    kmeans.fit(X)
    lables = kmeans.labels_
    meandistortions.append(sum(np.min(cdist(X, kmeans.cluster_centers_, 'euclidean'), axis=1)) / X.shape[0])
    Scores.append(
        silhouette_score(X, lables, metric='euclidean'))  # euclidean 欧氏距离
    CH.append(metrics.calinski_harabasz_score(X, lables))
    print("标签：", lables)
    np.savetxt("D:\我\非遗\高维聚类结果\\400_6类_label.txt",lables)
    test_stat = {}
    l = lables.tolist()
    # print(l)
    for i in set(l):
        test_stat[i] = l.count(i)
    print(test_stat)'''
    for k in K:
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(X)
        lables = kmeans.labels_
        meandistortions.append(sum(np.min(cdist(X, kmeans.cluster_centers_, 'euclidean'), axis=1)) / X.shape[0])
        Scores.append(
            silhouette_score(X, lables, metric='euclidean'))  # euclidean 欧氏距离
        CH.append(metrics.calinski_harabasz_score(X, lables))
        print("标签：", lables)
        if(k==6):
            np.savetxt("D:\我\非遗\高维聚类结果\\tfidf\\6类_label3.txt", lables)
        elif(k==8):
            np.savetxt("D:\我\非遗\高维聚类结果\\tfidf\\8类_label3.txt", lables)
        elif(k==10):
            np.savetxt("D:\我\非遗\高维聚类结果\\tfidf\\10类_label3.txt", lables)
        test_stat = {}
        l = lables.tolist()
        # print(l)
        for i in set(l):
            test_stat[i] = l.count(i)
        print(test_stat)

        '''降维可视化
        tsne = TSNE(perplexity=30, n_components=2, init='pca')   TSNE降维，降到2D
        data = tsne.fit_transform(X)

        x_min, x_max = np.min(data, 0), np.max(data, 0)
        data = (data - x_min) / (x_max - x_min)  # 归一化

        plt.figure()
        for i in range(data.shape[0]):
            plt.text(data[i, 0], data[i, 1], str(lables[i]),
                     color=plt.cm.Set1(lables[i] / 10.),
                     fontdict={'weight': 'bold', 'size': 9})
        plt.xticks([])
        plt.yticks([])
        plt.show()'''


    #np.save("D:\我\非遗\高维聚类结果\\100w\\number.txt", test_stat)
    print("轮廓系数：", Scores)
    #np.savetxt("D:\我\非遗\Word2vector\标签\轮廓系数_50.txt", Scores)
    print("成本函数：", meandistortions)
    #np.savetxt("D:\我\非遗\Word2vector\标签\成本函数_50.txt", meandistortions)
    #print("CH：", CH)
    #np.savetxt("D:\我\非遗\Word2vector\标签\CH_50.txt", CH)


def main():
    #X_w2v_50 = np.loadtxt("./Word2vector/w2v_sentence_vec_50D.txt")
    X_w2v_100 = np.loadtxt("./Word2vector/cbow-hn/w2v_sentence_vec_100D_cbow.txt")
    X_w2v_200 = np.loadtxt("./Word2vector/cbow-hn/w2v_sentence_vec_200D_cbow.txt")
    X_w2v_300 = np.loadtxt("./Word2vector/cbow-hn/w2v_sentence_vec_300D_cbow.txt")
    X_w2v_400 = np.loadtxt("./Word2vector/cbow-hn/w2v_sentence_vec_400D_cbow.txt")
    X_tfidf = np.loadtxt("words_tfidf2.txt")

    X_w2v_100_ns = np.loadtxt("./Word2vector/cbow-ns/w2v_sentence_vec_100D_ns.txt")
    X_w2v_200_ns = np.loadtxt("./Word2vector/cbow-ns/w2v_sentence_vec_200D_ns.txt")
    X_w2v_300_ns = np.loadtxt("./Word2vector/cbow-ns/w2v_sentence_vec_300D_ns.txt")
    X_w2v_400_ns = np.loadtxt("./Word2vector/cbow-ns/w2v_sentence_vec_400D_ns.txt")

    kmeans(X_tfidf)

if __name__ == '__main__':
    main()