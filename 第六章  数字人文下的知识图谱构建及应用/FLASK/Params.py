#!/usr/bin/env python  
# -*- coding:utf-8 _*-  
"""
@Time:2021-05-10 20:07
@Author:Veigar
@File: Params.py
@Github:https://github.com/veigaran
"""
import os
import pickle
import ahocorasick
import joblib


class Params:
    def __init__(self):
        cur_dir = '/'.join(os.path.abspath(__file__).split('/')[:-1])
        # 路径
        self.vocab_path = os.path.join(cur_dir, 'data/vocab.txt')
        self.stopwords_path = os.path.join(cur_dir, 'data/stop_words.utf8')
        self.word2vec_path = r'F:\A文档\python学习\Competition\Medication\代码\data\merge_sgns_bigram_char300.txt'  # os.path.join(cur_dir, 'data/merge_sgns_bigram_char300.txt')
        self.stopwords = [w.strip() for w in open(self.stopwords_path, 'r', encoding='utf8') if w.strip()]

        # 意图分类模型文件
        self.tfidf_path = os.path.join(cur_dir, 'model/tf.pkl')
        self.nb_test_path = os.path.join(cur_dir, 'model/SVM.m')  # 测试nb模型
        self.tfidf_model = pickle.load(open(self.tfidf_path, "rb"))
        self.nb_model = joblib.load(self.nb_test_path)

        self.person_path = os.path.join(cur_dir, 'data/人物.txt')
        self.alias_path = os.path.join(cur_dir, 'data/别名.txt')
        self.surname_path = os.path.join(cur_dir, 'data/姓氏.txt')
        self.country_path = os.path.join(cur_dir, 'data/国家.txt')
        self.school_path = os.path.join(cur_dir, 'data/学派.txt')
        self.rank_path = os.path.join(cur_dir, 'data/等级.txt')
        self.field_path = os.path.join(cur_dir, 'data/领域.txt')

        self.person_entities = [w.strip() for w in open(self.person_path, encoding='utf8') if w.strip()]
        self.alias_entities = [w.strip() for w in open(self.alias_path, encoding='utf8') if w.strip()]
        self.surname_entities = [w.strip() for w in open(self.surname_path, encoding='utf8') if w.strip()]
        self.country_entities = [w.strip() for w in open(self.country_path, encoding='utf8') if w.strip()]
        self.school_entities = [w.strip() for w in open(self.school_path, encoding='utf8') if w.strip()]
        self.rank_entities = [w.strip() for w in open(self.rank_path, encoding='utf8') if w.strip()]
        self.field_entities = [w.strip() for w in open(self.field_path, encoding='utf8') if w.strip()]

        # 构造领域actree
        self.person_tree = self.build_actree(list(set(self.person_entities)))
        self.alias_tree = self.build_actree(list(set(self.alias_entities)))
        self.surname_tree = self.build_actree(list(set(self.surname_entities)))
        self.country_tree = self.build_actree(list(set(self.country_entities)))
        self.school_tree = self.build_actree(list(set(self.school_entities)))
        self.rank_tree = self.build_actree(list(set(self.rank_entities)))
        self.field_tree = self.build_actree(list(set(self.field_entities)))

        self.name_qwds = ['英文名是什么', '通用名是什么', '一般叫什么', '哪些名字', '什么名字']
        self.country_qwds = ['国家是什么', '国家', '属于哪个国家']
        self.children_qwds = ['子女有哪些', '子女是谁', '儿子是谁', '孩子有哪些', '孩子是谁']
        self.father_qwds = ['父亲是谁', '爸爸是谁', '父亲', '爸爸', '爸爸是什么名字']

    def build_actree(self, wordlist):
        """
        构造actree，加速过滤
        :param wordlist:
        :return:
        """
        actree = ahocorasick.Automaton()
        # 向树中添加单词
        for index, word in enumerate(wordlist):
            actree.add_word(word, (index, word))
        actree.make_automaton()
        return actree
