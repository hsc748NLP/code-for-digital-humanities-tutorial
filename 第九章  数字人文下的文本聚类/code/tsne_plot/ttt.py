import codecs
import numpy as np
import matplotlib.pyplot as plt
from sklearn import manifold, datasets
from itertools import cycle

'''# read data
path="D:\我\非遗\cut_words_entity.txt"
num = []
with codecs.open(path, 'r', 'utf8') as f:
    line = f.readlines()
    for i in line:
        l=i.strip().replace('\n', '').split(' ')
        res = [x.strip() for x in l if x.strip() != '']
        print(res)
        num.append(len(res))
print(max(num))
print(min(num))
print(np.average(num))'''
X = np.loadtxt("D:\我\非遗\Word2vector\w2v_sentence_vec_50D.txt")
y = np.loadtxt("D:\我\非遗\Word2vector\标签\lables_50.txt")
tsne = manifold.TSNE(n_components=2, init='pca', random_state=501)
X_tsne = tsne.fit_transform(X)

print("Org data dimension is {}.\
      Embedded data dimension is {}".format(X.shape[-1], X_tsne.shape[-1]))

#嵌入空间可视化
colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
x_min, x_max = X_tsne.min(0), X_tsne.max(0)
X_norm = (X_tsne - x_min) / (x_max - x_min)  # 归一化
plt.figure(figsize=(8, 8))
for i in range(X_norm.shape[0],):
    plt.text(X_norm[i, 0], X_norm[i, 1], str(y[i]), color=plt.cm.Set1(y[i] /20.),
             fontdict={'weight': 'bold', 'size': 9})
plt.xticks([])
plt.yticks([])
plt.show()
'''
plt.figure(1)
plt.clf()
colors =['k','darkgrey','brown','r','peru','tan','gold','olive','y','sage','palegreen','g','c','deepskyblue','b','m','pink']
#colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
for k, col in zip(range(6), colors):
    # 根据lables中的值是否等于k，重新组成一个True、False的数组
    my_members = y == k
    # X[my_members, 0] 取出my_members对应位置为True的值的横坐标
    plt.plot(X_tsne[my_members, 0], X_tsne[my_members, 1], col + '.')
plt.show()'''



