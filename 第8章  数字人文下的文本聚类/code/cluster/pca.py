import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

def pca():
    f = open('./Word2vector/w2v_sentence_vec_250D.txt')
    line = f.readline()
    data = []
    data_list = []
    while line:
        num = list(map(float, line.split()))
        data_list.append(num)
        line = f.readline()
    f.close()
    X = np.array(data_list)  # 导入数据，维度为n
    pca = PCA(n_components=3)  # 降到2维
    pca.fit(X)  # 训练
    newX = pca.fit_transform(X)  # 降维后的数据
    # PCA(copy=True, n_components=2, whiten=False)
    # print(pca.explained_variance_ratio_)  #输出贡献率
    # print(newX)
    np.savetxt("./Word2vector/w2v_sentence_vec_250D-2d.txt", newX)


def tsne():
    X = np.loadtxt("D:\我\非遗\Word2vector\w2v_sentence_vec_100D.txt")
    tsne = TSNE(perplexity=30, n_components=2, init='pca')  # TSNE降维，降到2D
    data = tsne.fit_transform(X)


