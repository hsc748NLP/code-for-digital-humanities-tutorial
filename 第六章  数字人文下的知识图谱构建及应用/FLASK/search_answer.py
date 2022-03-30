#!/usr/bin/env python  
# -*- coding:utf-8 _*-  
"""
@Time:2021-05-10 20:09
@Author:Veigar
@File: search_answer.py
@Github:https://github.com/veigaran
"""
#!/usr/bin/env python3
# coding: utf-8
from py2neo import Graph


class AnswerSearching:
    def __init__(self):
        self.graph = Graph("http://localhost:7474", username="neo4j", password="123456")
        self.top_num = 10

    def question_parser(self, data):
        """
        """
        sqls = []
        if data:
            for intent in data["intentions"]:
                sql_ = {}
                sql_["intention"] = intent
                sql = []
                if data.get("person"):
                    sql = self.transfor_to_sql("person", data["person"], intent)
                elif data.get("alias"):
                    sql = self.transfor_to_sql("alias", data["alias"], intent)
                elif data.get("country"):
                    sql = self.transfor_to_sql("country", data["country"], intent)
                elif data.get("rank"):
                    sql = self.transfor_to_sql("rank", data["rank"], intent)
                if sql:
                    sql_['sql'] = sql
                    sqls.append(sql_)
        return sqls

    def transfor_to_sql(self, label, entities, intent):
        """
        将问题转变为cypher查询语句
        :param label:实体标签
        :param entities:实体列表
        :param intent:查询意图
        :return:cypher查询语句
        """
        if not entities:
            return []
        sql = []

        if intent == "query_alias" and label == "person":
            sql = ["MATCH(m:person)-[:person_is_alias]->(i) WHERE m.name=~'{0}.*' RETURN m.name,i.name".format(e)
                   for e in entities]

        if intent == "query_people_country" and label == "person":
            sql = ["MATCH(m:person)-[:person_is_country]->(i) WHERE m.name=~'{0}.*' RETURN m.name,i.name".format(e)
                   for e in entities]

        if intent == "query_people_father" and label == "person":
            sql = ["MATCH(m:person)-[:person_is_father]->(i) WHERE m.name=~'{0}.*' RETURN m.name,i.name".format(e)
                   for e in entities]

        if intent == "query_people_school" and label == "person":
            sql = ["MATCH(m:person)-[:person_is_school]->(i) WHERE m.name=~'{0}.*' RETURN m.name,i.name".format(e)
                   for e in entities]

        if intent == "query_people_rank" and label == "person":
            sql = ["MATCH(m:person)-[:person_is_rank]->(i) WHERE m.name=~'{0}.*' RETURN m.name,i.name".format(e)
                   for e in entities]

        if intent == "query_people_children" and label == "person":
            sql = ["MATCH(m:person)-[:person_is_children]->(i) WHERE m.name=~'{0}.*' RETURN m.name,i.name".format(e)
                   for e in entities]

        return sql

    def searching(self, sqls):
        """
        执行cypher查询，返回结果
        :param sqls:
        :return:str
        """
        final_answers = []
        for sql_ in sqls:
            intent = sql_['intention']
            queries = sql_['sql']
            answers = []
            for query in queries:
                ress = self.graph.run(query).data()
                answers += ress
            final_answer = self.answer_template(intent, answers)
            if final_answer:
                final_answers.append(final_answer)
        return final_answers

    def answer_template(self, intent, answers):
        """
        根据不同意图，返回不同模板的答案
        :param intent: 查询意图
        :param answers: 知识图谱查询结果
        :return: str
        """
        final_answer = ""
        if not answers:
            return ""

        if intent == "query_alias":
            person_dic = {}
            for data in answers:
                d = data['m.name']
                s = data['i.name']
                if d not in person_dic:
                    person_dic[d] = [s]
                else:
                    person_dic[d].append(s)
            i = 0
            for k, v in person_dic.items():
                if i >= 10:
                    break
                final_answer += "人物{0} 的别名有：{1}\n".format(k, ','.join(list(set(v))))
                i += 1

        if intent == "query_country":
            dic = {}
            for data in answers:
                m = data['m.name']
                g = data['c.name']
                if m not in dic:
                    dic[m] = [g]
                else:
                    dic[m].append(g)
            i = 0
            for k, v in dic.items():
                if i >= 10:
                    break
                final_answer += "人物 {0} 的国家为：{1}\n".format(k, ','.join(list(set(v))))
                i += 1

        return final_answer



