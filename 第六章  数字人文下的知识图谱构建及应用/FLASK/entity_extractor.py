#!/usr/bin/env python  
# -*- coding:utf-8 _*-  
"""
@Time:2021-05-10 20:05
@Author:Veigar
@File: entity_extractor.py
@Github:https://github.com/veigaran
"""

from Params import Params
import jieba
from predict import predict
from FindSim import FindSim


class EntityExtractor(Params):
    def __init__(self):
        super().__init__()
        self.result = {}
        self.find_sim_words = FindSim.find_sim_words

    def entity_reg(self, question):
        """
        模式匹配, 得到匹配的词和类型。
        :param question:str
        :return:
        """
        self.result = {}

        for i in self.person_tree.iter(question):
            word = i[1][1]
            if "person" not in self.result:
                self.result["person"] = [word]
            else:
                self.result["person"].append(word)

        for i in self.alias_tree.iter(question):
            word = i[1][1]
            if "alias" not in self.result:
                self.result["alias"] = [word]
            else:
                self.result["alias"].append(word)

        for i in self.surname_tree.iter(question):
            word = i[1][1]
            if "surname" not in self.result:
                self.result["surname"] = [word]
            else:
                self.result["surname"].append(word)

        for i in self.country_tree.iter(question):
            word = i[1][1]
            if "country" not in self.result:
                self.result["country"] = [word]
            else:
                self.result["country"].append(word)
        return self.result

    def check_words(self, wds, sent):
        """
        基于特征词分类
        :param wds:
        :param sent:
        :return:
        """
        for wd in wds:
            if wd in sent:
                return True
        return False

    def tfidf_features(self, text, vectorizer):
        """
        提取问题的TF-IDF特征
        :param text:
        :param vectorizer:
        :return:
        """
        jieba.load_userdict(self.vocab_path)
        words = [w.strip() for w in jieba.cut(text) if w.strip() and w.strip() not in self.stopwords]
        sents = [' '.join(words)]

        tfidf = vectorizer.transform(sents).toarray()
        return tfidf

    def model_predict(self, x, model):
        """
        预测意图
        :param x:
        :param model:
        :return:
        """
        pred = model.predict(x)
        return pred

    # 实体抽取主函数
    def extractor(self, question):
        self.entity_reg(question)
        if not self.result:
            # LSTM命名实体识别
            self.result = predict(question)
            if not self.result:
                self.find_sim_words(question)

        types = []  # 实体类型
        for v in self.result.keys():
            types.append(v)

        intentions = []  # 查询意图

        # 意图预测
        tfidf_feature = self.tfidf_features(question, self.tfidf_model)
        predicted = self.model_predict(tfidf_feature, self.nb_model)
        intentions.append(predicted[0])

        if self.check_words(self.name_qwds, question) and ('person' in types or 'alias' in types):
            intention = "query_alias"
            if intention not in intentions:
                intentions.append(intention)

        if self.check_words(self.name_qwds, question) and ('person' in types or 'alias' in types):
            intention = "query_people_country"
            if intention not in intentions:
                intentions.append(intention)

        if self.check_words(self.name_qwds, question) and ('person' in types or 'alias' in types):
            intention = "query_school"
            if intention not in intentions:
                intentions.append(intention)

        if self.check_words(self.name_qwds, question) and ('person' in types or 'alias' in types):
            intention = "query_father"
            if intention not in intentions:
                intentions.append(intention)

        if self.check_words(self.name_qwds, question) and ('person' in types or 'alias' in types):
            intention = "query_children"
            if intention not in intentions:
                intentions.append(intention)

        if self.check_words(self.name_qwds, question) and ('person' in types or 'alias' in types):
            intention = "query_birth_death"
            if intention not in intentions:
                intentions.append(intention)

        if self.check_words(self.name_qwds, question) and ('person' in types or 'alias' in types):
            intention = "query_field"
            if intention not in intentions:
                intentions.append(intention)

        if self.check_words(self.name_qwds, question) and ('person' in types or 'alias' in types):
            intention = "field_to_people"
            if intention not in intentions:
                intentions.append(intention)

        # 若没有识别出实体或意图则调用其它方法
        if not intentions or not types:
            intention = "QA_matching"
            if intention not in intentions:
                intentions.append(intention)

        self.result["intentions"] = intentions
        # print(self.result)
        return self.result


if __name__ == '__main__':
    test = EntityExtractor()
    question = '鲁桓公有哪些名字'
    a = test.extractor(question)
    print(a)
