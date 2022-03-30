from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn.metrics import silhouette_score
from sklearn import metrics
from sklearn.manifold import TSNE
from scipy.spatial.distance import cdist


data = []
def kmeans(X, clusters):
    kmeans = KMeans(n_clusters=clusters).fit(X)
    lable_pred = kmeans.labels_
    print(lable_pred)  # 聚类标签
    kmeans.predict(X)
    cluster_centers = kmeans.cluster_centers_
    #print(cluster_centers)  # 聚类中心向量
    inertia = kmeans.inertia_  # 聚类中心均值向量的总和
    #print(inertia)
    data.append(inertia)
    test_stat = {}
    l = lable_pred.tolist()
    #print(l)
    for i in set(l):
        test_stat[i] = l.count(i)
    print(test_stat)
    ##绘图
    '''plt.figure(1)
    plt.clf()

    colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
    for k, col in zip(range(clusters), colors):
        # 根据lables中的值是否等于k，重新组成一个True、False的数组
        my_members = lable_pred == k
        # X[my_members, 0] 取出my_members对应位置为True的值的横坐标
        plt.plot(X[my_members, 0], X[my_members, 1], col + '.')

    plt.title('Estimated number of clusters: %d' % clusters)
    plt.show()'''
    return lable_pred

def plot_embedding(X, lables):
    tsne = TSNE(perplexity=30, n_components=2, init='pca')  # TSNE降维，降到2D
    data = tsne.fit_transform(X)

    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)

    plt.figure()
    for i in range(data.shape[0]):
        plt.text(data[i, 0], data[i, 1], str(lables[i]),
                 color=plt.cm.Set1(lables[i] / 10.),
                 fontdict={'weight': 'bold', 'size': 9})
    plt.xticks([])
    plt.yticks([])
    plt.show()
    #plt.title(title)

def meandistortions(X):
    K = range(2, 11)
    meandistortions = []
    for k in K:
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(X)
        meandistortions.append(sum(np.min(cdist(X, kmeans.cluster_centers_, 'euclidean'), axis=1)) / X.shape[0])
    plt.plot(K, meandistortions, 'o-')
    plt.xlabel('k')
    plt.ylabel('Mean distortions')
    plt.title('The best k value')
    plt.show()
    print("成本函数：", meandistortions)


def Sil(X):
    Scores = []  # 存放轮廓系数
    for k in range(2,17):
        estimator = KMeans(n_clusters=k)  # 构造聚类器
        estimator.fit(X)
        Scores.append(
            silhouette_score(X, estimator.labels_, metric='euclidean'))  # euclidean 欧氏距离
    x = range(2, 17)
    plt.title('Clustering Silhouette coefficient index-Kmeans')
    plt.xlabel("clusters")
    plt.ylabel("Silhouette coefficient")
    plt.plot(x, Scores, 'o-')
    plt.show()
    print("轮廓系数：", Scores)


def SSE_Sil(X):
    Scores = []  # 存放轮廓系数
    SSE = []
    #data=[5, 6, 7, 8, 9, 10, 13, 16]
    for k in range(5,17):
        estimator = KMeans(n_clusters=k)  # 构造聚类器
        estimator.fit(X)
        SSE.append(estimator.inertia_)

        Scores.append(
            silhouette_score(X, estimator.labels_, metric='euclidean'))  # euclidean 欧氏距离
    #plt.title('Clustering Silhouette coefficient and SSE index-Kmeans')
    x = range(5,17)
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    #ax1.title('Clustering Silhouette coefficient and SSE index-Kmeans')
    ax1.plot(x, SSE, 'go-')
    ax2.plot(x, Scores, 'b*-')
    ax1.set_xlabel('clusters')
    ax1.set_ylabel('SSE', color='g')
    ax2.set_ylabel('Silhouette coefficient', color='b')
    for x, y in zip(x, Scores):
        plt.text(x, y + 0.001, '%.4f' % y, ha='center', va='bottom', fontsize=9)
    plt.show()


def CH(X, lables):
    CH=[]
    data=[5, 6, 7, 8, 9, 10, 13, 16]
    for k in range(5, 17):
        #kmeans = KMeans(n_clusters=k).fit(X)
        #labels = kmeans.labels_
        CH.append(metrics.calinski_harabasz_score(X, lables))
    plt.title('Clustering calinski_harabasz_score index-Kmeans')
    x = range(5, 17)
    plt.xlabel("clusters")
    plt.ylabel("calinski_harabaz_score")
    plt.plot(x, CH, 'o-')
    for x, y in zip(x, CH):
        plt.text(x, y + 0.001, '%.4f' % y, ha='center', va='bottom', fontsize=9)
    plt.show()
    print("CH:", CH)




def main():
    clusters = 6
    X = np.loadtxt("tfidf_data.txt")
    X1 = np.loadtxt("tfidf_data2.txt")
    X_w2v_50 = np.loadtxt("./Word2vector/w2v_sentence_vec_50D.txt")  # word2vec_sentence_size=50
    X_w2v_100 = np.loadtxt("./Word2vector/w2v_sentence_vec_100D.txt")  # word2vec_sentence_size=100
    X_w2v_150 = np.loadtxt("./Word2vector/w2v_sentence_vec_150D.txt")  # word2vec_sentence_size=150
    # X_w2v_180 = np.loadtxt("./Word2vector/w2v_sentence_vec_180D.txt")  # word2vec_sentence_size=180
    X_w2v_200 = np.loadtxt("./Word2vector/w2v_sentence_vec_200d.txt")  # word2vec_sentence_size=200
    X_w2v_250 = np.loadtxt("./Word2vector/w2v_sentence_vec_250D.txt")  # word2vec_sentence_size=250
    X_w2v_300 = np.loadtxt("./Word2vector/w2v_sentence_vec_300D.txt")  # word2vec_sentence_size=300
    X_w2v_350 = np.loadtxt("./Word2vector/w2v_sentence_vec_350D.txt")  # word2vec_sentence_size=300
    X_w2v_400 = np.loadtxt("./Word2vector/w2v_sentence_vec_400D.txt")  # word2vec_sentence_size=300
    #d2v_200 = np.loadtxt("./Word2vector/d2v_heritage.txt")
    #for i in range(5, 17):
    #kmeans(X_w2v_400, i)
    #np.loadtxt("./350lables.xlsx",lables)
    #plot_embedding(X_w2v_350, lables)
    Sil(X_w2v_400)
    meandistortions(X_w2v_400)
    #SSE_Sil(X1)




main()
