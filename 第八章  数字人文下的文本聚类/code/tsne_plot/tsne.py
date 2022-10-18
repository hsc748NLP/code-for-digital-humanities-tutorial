import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE


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

def main():
    X = np.loadtxt("D:\我\非遗\Word2vector\w2v_sentence_vec_100D.txt")
    lables = np.loadtxt("D:\我\非遗\Word2vector\标签\\100-lables.txt")
    tsne = TSNE(perplexity=30, n_components=2, init='pca')  # TSNE降维，降到2D
    data = tsne.fit_transform(X)
    print(data)
    plot_embedding(data, lables)


main()
'''
X = np.loadtxt("D:\我\非遗\Word2vector\w2v_sentence_vec_100D.txt")
labels = np.loadtxt("D:\我\非遗\Word2vector\标签\\100-lables.txt")
tsne = TSNE(perplexity=30, n_components=2, init='pca')  # TSNE降维，降到2D
data = tsne.fit_transform(X)
plt.figure(1)
plt.clf()
colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
for k, col in zip(range(6), colors):
    # 根据lables中的值是否等于k，重新组成一个True、False的数组
    my_members = labels == k
    # X[my_members, 0] 取出my_members对应位置为True的值的横坐标
    plt.plot(data[my_members, 0], data[my_members, 1], col + '.')

#plt.title('Estimated number of clusters: %d' % n_clusters_)
plt.show()'''
