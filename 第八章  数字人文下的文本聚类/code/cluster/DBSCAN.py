import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import DBSCAN


def DBscan(X):
    ##产生随机数据的中心
    # centers = [[1, 1], [-1, -1], [1, -1]]
    ##产生的数据个数
    # n_samples = 750
    ##生产数据:此实验结果受cluster_std的影响，或者说受eps 和cluster_std差值影响
    # X, lables_true = make_blobs(n_samples=n_samples, centers=centers, cluster_std=0.4,
    #                           random_state=0)
    ##设置分层聚类函数
    db = DBSCAN(eps=0.5, min_samples=50)
    ##训练数据
    db.fit(X)
    ##初始化一个全是False的bool类型的数组
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    '''
       这里是关键点(针对这行代码：xy = X[class_member_mask & ~core_samples_mask])：
       db.core_sample_indices_  表示的是某个点在寻找核心点集合的过程中暂时被标为噪声点的点(即周围点
       小于min_samples)，并不是最终的噪声点。在对核心点进行联通的过程中，这部分点会被进行重新归类(即标签
       并不会是表示噪声点的-1)，也可也这样理解，这些点不适合做核心点，但是会被包含在某个核心点的范围之内
    '''
    core_samples_mask[db.core_sample_indices_] = True

    ##每个数据的分类
    lables = db.labels_

    ##分类个数：lables中包含-1，表示噪声点
    n_clusters_ = len(np.unique(lables)) - (1 if -1 in lables else 0)

    ##绘图
    unique_labels = set(lables)
    '''
       1)np.linspace 返回[0,1]之间的len(unique_labels) 个数
       2)plt.cm 一个颜色映射模块
       3)生成的每个colors包含4个值，分别是rgba
       4)其实这行代码的意思就是生成4个可以和光谱对应的颜色值
    '''
    colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))

    plt.figure(1)
    plt.clf()

    for k, col in zip(unique_labels, colors):
        ##-1表示噪声点,这里的k表示黑色
        if k == -1:
            col = 'k'

        ##生成一个True、False数组，lables == k 的设置成True
        class_member_mask = (lables == k)

        ##两个数组做&运算，找出即是核心点又等于分类k的值  markeredgecolor='k',
        xy = X[class_member_mask & core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', c=col, markersize=14)
        '''
           1)~优先级最高，按位对core_samples_mask 求反，求出的是噪音点的位置
           2)& 于运算之后，求出虽然刚开始是噪音点的位置，但是重新归类却属于k的点
           3)对核心分类之后进行的扩展
        '''
        xy = X[class_member_mask & ~core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', c=col, markersize=6)

    plt.title('Estimated number of clusters: %d' % n_clusters_)
    print(n_clusters_)
    plt.show()

def main():
    X_w2v_100 = np.loadtxt("./Word2vector/w2v_sentence_vec_100D.txt")  # word2vec_sentence_size=100
    DBscan(X_w2v_100)

if __name__ == '__main__':
    main()