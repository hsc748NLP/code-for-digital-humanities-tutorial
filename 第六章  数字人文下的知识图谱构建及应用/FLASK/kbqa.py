#!/usr/bin/env python  
# -*- coding:utf-8 _*-  
"""
@Time:2021-05-10 20:09
@Author:Veigar
@File: kbqa.py
@Github:https://github.com/veigaran
"""

from entity_extractor import EntityExtractor
from search_answer import AnswerSearching


class KBQA:
    def __init__(self):
        pass
        # self.extractor = EntityExtractor()
        # self.searcher = AnswerSearching()

    def qa_main(self, input_str):
        answer = "对不起，您的问题我不知道，我今后会努力改进的。"
        entities = self.extractor.extractor(input_str)
        if not entities:
            return answer
        sqls = self.searcher.question_parser(entities)
        final_answer = self.searcher.searching(sqls)
        if not final_answer:
            return answer
        else:
            return '\n'.join(final_answer)


if __name__ == "__main__":
    handler = KBQA()
    while True:
        question = input("请输入：")
        if not question:
            break
        answer = handler.qa_main(question)
        print("", answer)
        print("*"*50)