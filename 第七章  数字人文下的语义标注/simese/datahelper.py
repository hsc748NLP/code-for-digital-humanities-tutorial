'''
author: 纪有书
datahelper.py
'''
import torch
import nlpertools
import random
import os
import time


random.seed(8)
PAD = '<pad>'
output_dir = 'input_and_output'
vocab_path = os.path.join(output_dir, 'id2char.pkl')
pairs_read_db_path = os.path.join('/home/admin/youshuJi/GRADUATIONPRIJECT/Experiment', 'Common/pairs_read_file.pkl')
def get_vocab():
    pairs = nlpertools.pickle_load(pairs_read_db_path)
    sentences = []
    sentences.extend(pairs['raw_sentences'].values.tolist())
    sentences.extend(pairs['yiwen_sentences'].values.tolist())
    tokens = [i.split() for i in sentences]
    chars = set(nlpertools.j_chain(tokens))# .symmetric_difference({PAD})
    id2char = [PAD]
    id2char.extend(list(chars))
    nlpertools.pickle_save(id2char, vocab_path)

def get_each_tokens(i, max_seq_len):
    cur_tokens = i[:max_seq_len]
    cur_tokens.extend([PAD] * (max_seq_len - len(i)))
    return cur_tokens

def get_each_dx(i, max_seq_len):
    tmp = [0] * max_seq_len
    tmp[i] = 1.0
    return tmp

def get_tokens(seqs, max_seq_len):
    tokens = []
    for i in seqs:
        cur_tokens = get_each_tokens(i, max_seq_len)
        tokens.append(cur_tokens)
    return tokens

def get_dx(dx, max_seq_len):
    dx_for_indices = []
    for i in dx:
        tmp = get_each_dx(i, max_seq_len)
        dx_for_indices.append(tmp)
    return dx_for_indices


def convert_userinput_into_feature(userinput, max_seq_len, id2char):
    sentence_a, sentence_b, idx, jdx = userinput[0], userinput[1], userinput[2], userinput[3]
    char2id = {u: i for i, u in enumerate(id2char)}
    sentence_a_for_siamese = get_tokens([sentence_a.split()], max_seq_len)
    sentence_b_for_siamese = get_tokens([sentence_b.split()], max_seq_len)
    sentence_a_indices = [[char2id[w] for w in seq] for seq in sentence_a_for_siamese]
    sentence_b_indices = [[char2id[w] for w in seq] for seq in sentence_b_for_siamese]
    adx_for_indices = get_dx([idx], max_seq_len)
    bdx_for_indices = get_dx([jdx], max_seq_len)
    return torch.tensor(sentence_a_indices), torch.tensor(sentence_b_indices), torch.tensor(adx_for_indices), torch.tensor(bdx_for_indices)



