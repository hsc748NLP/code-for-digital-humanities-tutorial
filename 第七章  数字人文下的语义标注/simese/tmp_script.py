import nlpertools
import random


# 定义选择数据的函数
def select_data():
    # 读出所有生成的负类样本
    negative = nlpertools.pickle_load('data/negative_data.pkl')
    # 读出所有生成的正类样本
    positive = nlpertools.pickle_load('data/positive_data.pkl')
    # 定义数组变量，用于存放取出的训练数据
    new_negative, new_positive = [], []
    # 取负类
    for i in negative:
        # negative中每个元素是某一组词的所有训练数据，我们首先进行打乱
        random.shuffle(i)
        # 然后每组词取19条数据
        new_negative.extend(i[:19])
    # 正类取数据方法同负类
    for i in negative:
        random.shuffle(i)
        new_positive.extend(i[:19])
    # 打印取出的正类数据数量
    print(len(new_positive))
    # 打印取出的负类数据数量
    print(len(new_negative))
    # 保存取出的正类数据
    nlpertools.save_to_json(new_positive, 'data/positive_data.json')
    # 保存取出的负类数据
    nlpertools.save_to_json(new_negative, 'data/negative_data.json')


# 调用取数据的函数
select_data()