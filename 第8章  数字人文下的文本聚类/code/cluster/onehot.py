import jieba
import numpy as np
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


def main():
    path = "./title_info.txt"
    text = read_txt(path)
    vector = onehot(text)
    print(1)
    # 降维 PCA/SVD
    pca = PCA(n_components=2)  # 降到2维
    pca.fit(vector)
    new_vector = pca.fit_transform(vector)
    print(2)

    # 写入文本
    np.savetxt("onehot_data.txt", new_vector)



if __name__ == '__main__':
    main()
