import re
import time
from tqdm import tqdm, trange


# 读取txt文件，path为txt文件路径/string，text为txt中文本内容/string
def read_txt(path):
    f = open(path, 'r', encoding='UTF-8')
    text = f.read()
    f.close()
    print("文本文件已完成读取 path:" + path)
    text = text.strip()
    return text


# 写入txt文件，path为txt文件路径/string，sentence为要写入的文本/string
def write_txt_0(path, sentence):
    with open(path, 'a+', encoding='utf-8') as f:
        f.write(str(sentence) + "\n")
    # print("文本文件已完成写入 path:" + path)


# 将已分词文本text/string中的词和词性提取，返回词words/list、词性pos/list
def separate(text):
    words = []  # 词语
    pos = []  # 词性标签
    while len(text) > 0:
        try:
            temp = re.match(r"\S(.*?)/[a-z0-9\S](.*?) ", text).group(0)  # 正则表达式提取“词/标签”序列 eg.太阳/n
            text = text[len(temp):]
            words.append(re.match(r"(.*)/(.*) ", temp).group(1))  # 正则表达式提取词 eg.太阳
            pos.append(re.match(r"(.*)/(.*) ", temp).group(2))  # 获取词性标签 eg.n
        except AttributeError:  # 无法匹配错误
            text = text.strip()
            if len(text):
                text = text + " "
    return words, pos


# 将已分词文本text/string中的词和词性提取，返回词words/list、词性pos/list
def separate_seg(text):
    char = []  # 字符
    pos = []  # 分词标签
    while len(text) > 0:
        try:
            temp = re.match(r"\S(.*?)/[a-z0-9\S](.*?) ", text).group(0)  # 正则表达式提取“词/标签”序列 eg.太阳/n
            text = text[len(temp):]
            word = re.match(r"(.*)/(.*) ", temp).group(1)  # 正则表达式提取词 eg.太阳
            length = len(word)
            if length == 1:
                char.append(word)
                pos.append("S")
            elif length >= 2:
                char.append(word[len(word) - length])
                pos.append("B")
                length = length - 1
                while length > 1:
                    char.append(word[len(word) - length])
                    pos.append("M")
                    length = length - 1
                char.append(word[len(word) - length])
                pos.append("E")
        except AttributeError:  # 无法匹配错误
            text = text.strip()
            if len(text):
                text = text + " "
    return char, pos


# 估计HMM的三个参数pi、A、B，words/list为文本中所有的词、pos/list为词性，返回词性的先验概率pi/list，词性状态转移矩阵A/list2，词性序列混淆矩阵B/list2
def train_parameter(words, pos):
    tags = list(set(pos))  # 去重，得到词性标签
    n_tags = len(tags)  # 词性标签种类数
    terms = list(set(words))
    n_terms = len(terms)
    # 计算词性的先验概率pi
    pi = [0 for i in range(n_tags)]  # 词性的先验概率pi/list，列表长度等于词性标签种类数
    for i in range(n_tags):
        pi[i] = pos.count(tags[i]) / len(pos)  # 词性的先验概率=词性为i的词的个数/语料总词数
    print("先验概率计算完成")
    # 计算词性的状态转移矩阵A
    A = [[0 for behind in range(n_tags)] for front in range(n_tags)]  # 定义词性的状态转移矩阵A/list2，行数和列数均等于词性标签种类数
    num = [[0 for behind in range(n_tags)] for front in range(n_tags)]  # 定义词性ij相连出现的频次num/list2，行数和列数同A
    for i in range(1, len(pos)):  # 统计两个词性相连出现的频次
        front = tags.index(pos[i - 1])  # 前驱词性
        behind = tags.index(pos[i])  # 后继词性
        num[front][behind] += 1  # 频次加一
    for front in range(n_tags):
        for behind in range(n_tags):
            A[front][behind] = num[front][behind] / pos.count(tags[front])  # 词性为front的词其后继词的词性为behind的概率
    print("状态转移矩阵计算完成")
    # 计算词性序列混淆矩阵
    print("计算混淆矩阵...")
    B = [[0 for term in range(n_terms)] for tag in range(n_tags)]  # 定义词性序列混淆矩阵B/list2，行数等于词性标签种类数，列数等于词种类数
    freq = [[0 for term in range(n_terms)] for tag in range(n_tags)]  # 定义词性i序列j同时出现的频次freq/list2，行数和列数同B
    for i in trange(len(pos)):
        tag = tags.index(pos[i])  # 词性状态
        term = terms.index(words[i])  # 词序列
        freq[tag][term] += 1  # 频次加一
    for tag in trange(n_tags):
        for term in range(n_terms):
            B[tag][term] = freq[tag][term] / pos.count(tags[tag])  # 词性为tag时词term出现的概率
            # B[tag][term] = freq[tag][term] / words.count(terms[term])
    print("混淆矩阵计算完成")
    return pi, A, B


# 维特比算法对序列进行标注，O/list为待标注序列，tags/list为词性标签，terms/list为所有词，pi、A、B为HMM的三个参数，返回词性标注结果S/list
def viterbi(O, tags, terms, pi, A, B):
    length = len(O)  # 待标注序列长度
    n_tags = len(tags)  # 词性标签种类数
    delta = [[0 for i in range(n_tags)] for t in range(length)]
    S = ["" for t in range(length)]
    for t in range(length):
        max_d = max(delta[t])
        max_i = delta[t].index(max_d)
        try:
            if t == 0:
                for i in range(n_tags):
                    term = terms.index(O[t].strip())
                    delta[t][i] = pi[i] * B[i][term]
            else:
                try:
                    for i in range(n_tags):
                        term = terms.index(O[t].strip())
                        delta[t][i] = max_d * A[max_i][i] * B[i][term]
                except UnboundLocalError:
                    for i in range(n_tags):
                        term = terms.index(O[t].strip())
                        delta[t][i] = pi[i] * B[i][term]
            max_d = max(delta[t])
            max_i = delta[t].index(max_d)
            S[t] = tags[max_i]
        except ValueError:
            if O[t] == "\n":
                S[t] = ""
    # print(S)
    return S


def output_seg(char, label):
    text = ""
    i = 0
    while i < len(char):
        if label[i] == "S":
            text = text + char[i] + "/"
        elif label[i] == "B":
            try:
                while label[i] != "E":
                    text = text + char[i]
                    i = i + 1
                text = text + char[i] + "/"
            except IndexError:
                text = text + "/"
        i = i + 1
    return text


def output_tag(word, label):
    text = ""
    i = 0
    while i < len(word):
        text = text + word[i] + "/" + label[i] + " "
        i = i + 1
    return text


def main():
    path_train = "./data/corpus.txt"
    path_test = "./data/test.txt"
    path_output = "./data/output.txt"
    corpus = read_txt(path_train)

    # 词性标注 - 参数计算
    words, pos = separate(corpus)
    print("开始参数训练")
    train_start = time.time()
    pi, A, B = train_parameter(words, pos)
    train_end = time.time()
    print("隐马尔可夫词性标注参数训练完成，耗时: " + str(train_end - train_start) + "s")

    # 词性标注 - 预测
    tags = list(set(pos))
    terms = list(set(words))
    test = read_txt(path_test)
    test = test.split("\n")
    most_tag = max(pos, key=pos.count)  # 如果出现新词，标记为出现频率最高的词性
    print("开始序列预测")
    predict_start = time.time()
    for sentence in tqdm(test):
        sentence = sentence.strip()
        O = sentence.split("/")
        S = viterbi(O, tags, terms, pi, A, B)
        S = [most_tag if i == "" else i for i in S]
        write_txt_0(path_output, output_tag(O, S))
    predict_end = time.time()
    print("隐马尔可夫词性标注预测完成，耗时: " + str(predict_end - predict_start) + "s")


if __name__ == '__main__':
    main()
