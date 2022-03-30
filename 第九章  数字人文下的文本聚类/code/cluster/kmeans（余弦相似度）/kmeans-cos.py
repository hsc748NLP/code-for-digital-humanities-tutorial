import numpy as np
#from cluster.cluster.basefunction
from cluster.basealgorithm import kmeans
from sklearn.metrics import silhouette_score


def kmeans_cos(X):
    Scores = []
    for k in range(2,11):
        lables = kmeans(k=k, similarity_function="cos", max_iteration=300).train_predict(X.tolist())
        #print(lables)
        Scores.append(silhouette_score(X, lables, metric='euclidean'))
        test_stat = {}
        # print(l)
        for i in set(lables):
            test_stat[i] = lables.count(i)
        print(test_stat)
    print("轮廓系数：", Scores)


X_w2v_100 = np.loadtxt("./Word2vector/w2v_sentence_vec_400D.txt")
kmeans_cos(X_w2v_100)
