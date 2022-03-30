'''
author: 纪有书
datafeature.py
'''
import os
import nlpertools
import torch.utils.data as Data
import torch
import sys
import random
from datahelper import get_tokens, get_dx, PAD
from itertools import chain
random.seed(7)


output_dir = 'data'
vocab_path = os.path.join(output_dir, 'id2char.pkl')

def load_vocab():
    id2char = nlpertools.pickle_load(vocab_path)
    return id2char


def read_data(max_seq_len, negative_path=None, positive_path=None):
    positive, negative = [], []
    if negative_path:
        negative = nlpertools.load_from_json(negative_path)
    if positive_path:
        positive = nlpertools.load_from_json(positive_path)
    print('正类数量:{}'.format(len(positive)))
    print('负类数量:{}'.format(len(negative)))
    random.shuffle(negative)
    negative = negative[:len(positive) * 6]
    data = []
    data.extend(negative)
    data.extend(positive)
    random.shuffle(data)
    sentence_a, sentence_b, idx, jdx, label = [], [], [], [], []
    for each in data:
        if each['input']['idx'] >= max_seq_len or each['input']['jdx'] >= max_seq_len:
            continue
        sentence_a.append(each['input']['sentence_a'])
        sentence_b.append(each['input']['sentence_b'])
        idx.append(each['input']['idx'])
        jdx.append(each['input']['jdx'])
        label.append(each['label'])
    sentence_a_for_siamese = get_tokens(sentence_a, max_seq_len)
    sentence_b_for_siamese = get_tokens(sentence_b, max_seq_len)
    
    tokens = []
    tokens.extend(sentence_a_for_siamese)
    tokens.extend(sentence_b_for_siamese)
    # vocab
    if os.path.exists(vocab_path):
        id2char = load_vocab()
    else:
        chars = set(chain(*tokens)).symmetric_difference({PAD})
        id2char = [PAD]
        id2char.extend(list(chars))
        nlpertools.pickle_save(id2char, vocab_path)

    char2id = {u: i for i, u in enumerate(id2char)}
    sentence_a_indices = [[char2id[w] for w in seq] for seq in sentence_a_for_siamese]
    sentence_b_indices = [[char2id[w] for w in seq] for seq in sentence_b_for_siamese]
    idx_for_indices = get_dx(idx, max_seq_len)
    jdx_for_indices = get_dx(jdx, max_seq_len)
    label_for_indices = [i * 1.0 for i in label]
    dataset = Data.TensorDataset(torch.tensor(sentence_a_indices), torch.tensor(sentence_b_indices), torch.tensor(idx_for_indices), torch.tensor(jdx_for_indices), torch.tensor(label_for_indices))
    return id2char, char2id, dataset

