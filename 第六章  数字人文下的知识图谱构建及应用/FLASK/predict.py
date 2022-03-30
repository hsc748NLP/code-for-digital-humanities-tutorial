#!/usr/bin/env python  
# -*- coding:utf-8 _*-  
"""
@Time:2021-05-10 20:06
@Author:Veigar
@File: predict.py
@Github:https://github.com/veigaran
"""
import pickle
import torch
import jieba


def get_seg_features(string):
    """
    对句子分词，构造词的长度特征，为BIES格式,
    [对]对应的特征为[4], 不设为0，因为pad的id就是0
    [句子]对应的特征为[1,3],
    [中华人民]对应的特征为[1,2,2,3]
    """
    seg_feature = []

    for word in jieba.cut(string):
        if len(word) == 1:
            seg_feature.append(4)
        else:
            tmp = [2] * len(word)
            tmp[0] = 1
            tmp[-1] = 3
            seg_feature.extend(tmp)
    return seg_feature


def prepare_dataset(sentences, char_to_id, tag_to_id, lower=False, test=False):
    """
    把文本型的样本和标签，转化为index，便于输入模型
    需要在每个样本和标签前后加<start>和<end>,
    但由于pytorch-crf这个包里面会自动添加<start>和<end>的转移概率，
    所以我们不用在手动加入。
    """

    def f(x):
        return x.lower() if lower else x

    data = []
    for s in sentences:

        chars = [w[0] for w in s]
        tags = [w[-1] for w in s]

        """ 句子转化为index """
        chars_idx = [char_to_id[f(c) if f(c) in char_to_id else '<unk>'] for c in chars]

        """ 对句子分词，构造词的长度特征 """
        segs_idx = get_seg_features("".join(chars))

        if not test:
            tags_idx = [tag_to_id[t] for t in tags]

        else:
            tags_idx = [tag_to_id["<pad>"] for _ in tags]

        assert len(chars_idx) == len(segs_idx) == len(tags_idx)
        data.append([chars, chars_idx, segs_idx, tags_idx])

    return data


def result_to_json(string, tags):
    """ 按规范的格式输出预测结果 """
    item = {"string": string, "entities": []}
    entity_name = ""
    entity_start = 0
    idx = 0
    for char, tag in zip(string, tags):
        if tag[0] == "S":
            item["entities"].append({"word": char, "start": idx, "end": idx + 1, "type": tag[2:]})
        elif tag[0] == "B":
            entity_name += char
            entity_start = idx
        elif tag[0] == "I":
            entity_name += char
        elif tag[0] == "E":
            entity_name += char
            item["entities"].append({"word": entity_name, "start": entity_start, "end": idx + 1, "type": tag[2:]})
            entity_name = ""
        else:
            entity_name = ""
            entity_start = idx
        idx += 1
    return item


def predict(input_str):
    map_file = r'./model/maps.pkl'
    with open(map_file, "rb") as f:
        char_to_id, id_to_char, tag_to_id, id_to_tag = pickle.load(f)

    """ 用cpu预测 """
    model_file = r'./model/medical_ner.ckpt'
    model = torch.load(model_file, map_location="cpu")
    # model.eval()

    if not input_str:
        input_str = input("请输入文本: ")

    _, char_ids, seg_ids, _ = prepare_dataset([input_str], char_to_id, tag_to_id, test=True)[0]
    char_tensor = torch.LongTensor(char_ids).view(1, -1)
    seg_tensor = torch.LongTensor(seg_ids).view(1, -1)

    with torch.no_grad():
        """ 得到维特比解码后的路径，并转换为标签 """
        paths = model(char_tensor, seg_tensor)
        tags = [id_to_tag[idx] for idx in paths[0]]
    res = result_to_json(input_str, tags)
    entity_type = res["entities"][0]['type']
    word = res["entities"][0]['word']
    result = {}
    if entity_type == "DRU":
        result["person"] = [word]
    # pprint(result_to_json(input_str, tags))
    print(entity_type, word, '\n', result)
    return result


