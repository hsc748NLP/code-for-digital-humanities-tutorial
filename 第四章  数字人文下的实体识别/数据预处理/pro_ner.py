import os
from os import path
from os import listdir
import re
from tqdm import tqdm

from zhon import hanzi
from string import punctuation

punc = hanzi.punctuation + punctuation


def word_pos2word_seq(filepath, resultfolder='data_seq'):
    """
    word/tag转换为word\ttag
    word指代词，tag指代词性标签
    (不带BIES)
    """
    if resultfolder=='data_seq': 
        if not os.path.exists('data_seq'):os.makedirs('data_seq') # 创建输出结果文件夹
    data_name = os.path.split(filepath)[1][:-4]  # 获取当前输入数据文件文件名（不含前面的文件夹路径和最后的.txt）
    with open(filepath, 'rt', encoding='utf8')as f:
        with open('{}/{}.txt'.format(resultfolder, data_name), 'w', encoding='utf8') as r:
            for line in tqdm(f.readlines()): # 遍历读取每一行数据
                if line == '\n': # 若该行为空行则跳过
                    # r.write('\n')
                    continue
                content_lst = line.strip('\n\r').strip(' ') # 去除每行末尾空格
                content_lst = re.sub('  ', ' ', content_lst).split(' ')# 去除连续多余的空格为1个并按照空格拆分为列表: [word/tag, word2/tag2, ……]

                char_tag_lst = [c.split('/') for c in content_lst]
                char_lst = [c[0] for c in char_tag_lst] # word列表
                tag_lst = [c[1] for c in char_tag_lst] # tag列表

                for char, tag in zip(char_lst, tag_lst):
                    r.write(char + '\t' + tag + '\n')
                r.write('\n') # 每行结束之后增加一个空行用于区分不同行转换出的序列


def word_seq2char_seq(filepath,resultfolder='data_charseq'):
    """
    word\ttag转换为char\ttag
    (带BIES)
    """
    if resultfolder=='data_charseq':
        if not os.path.exists('data_charseq'):
            os.makedirs('data_charseq')
    data_name = os.path.split(filepath)[1] #[:-4]  # 数据文件名
    sep_char = ' '  # 生成的文件 word tag中的分隔符

    with open(filepath, 'rt', encoding='utf-8-sig')as f:
        with open('{}/{}'.format(resultfolder,data_name), 'w', encoding='utf-8')as r:
            for line in tqdm(f.readlines()): # 遍历读取每行 word\t tag\n
                if line == '\n':
                    # r.write(' \n')  # crf_learn并不认可数据中使用’\n’作为sentence间的分割符（空行），但能够识别‘space（空格）\n’的空行分隔符。
                    r.write('\n')   # bert_ner_pytorch的断句 使用’\n’作为sentence间的分割符（空行）
                    continue
                word_tag_lst = line.strip('\n').split('\t')
                word = word_tag_lst[0] # word
                tag = word_tag_lst[1] # tag

                char_lst = list(word) # 每行单个字组成的列表
                tag_lst = []
                if tag not in ['nr', 'ns', 't']: # 本次识别的实体词性标签
                    for char in word:
                        r.write(char + sep_char +'O\n') # 将非此次需要识别的标签认定为O
                    continue

                if len(word) == 1: # 单个字组成的实体，用S-tag表示
                    # char_lst.append(word)
                    tag_lst.append('S-' + tag)
                elif len(word) == 2: # 双字实体
                    # char_lst.extend([word[0],word[1]])
                    tag_lst.extend(['B-' + tag, 'E-' + tag])
                else: # 三字以上实体
                    for id, char in enumerate(word):
                        # char_lst.append(char)
                        if id == 0:
                            tag_lst.append('B-' + tag)
                        elif id < len(word) - 1:
                            tag_lst.append('I-' + tag)
                        else:
                            tag_lst.append('E-' + tag)
                for char, tag in zip(char_lst, tag_lst):
                    r.write(char + sep_char + tag + '\n')


def main():
    word_pos2word_seq('data/filename.txt')
    word_seq2char_seq('data_seq/filename.txt')

if __name__ == '__main__':
    main()
