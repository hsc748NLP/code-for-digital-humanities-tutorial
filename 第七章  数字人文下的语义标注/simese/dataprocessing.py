'''
author: 纪有书
dataprocessing.py
'''
import os
import nlpertools
import sys
import random
from tqdm import tqdm
root_path = '/home/admin/youshuJi/GRADUATIONPRIJECT/Experiment/Outputs/align_res'
align_synonym_path = os.path.join(root_path, 'forward.synonyms')
align_synonym_334_all_path = 'data/synonym_334_all.txt'
align_synonym_334_right_path = 'data/synonym_334_right.txt'
align_synonym_334_wrong_path = 'data/synonym_334_wrong.txt'
forward_path = os.path.join(root_path, 'forward.align')
pairs_path = os.path.join(root_path, 'pairs.tra2sim')
sentence_len_limit = 40

random.seed(8)
'''
data_for_siamesenet_right_len_limit
[
["鲁", "山", "，", "滍水", "所", "出", "，", "东北", "至", "定陵", "入", "汝", "。"], 
["鲁山", "，", "是", "溃水", "的", "发源地", "，", "东北", "到", "定陵", "流入", "汝水", "。"], 
[3, 0], 
"滍水", 
"鲁山"
]
'''
class DataProcessing(object):
    def __init__(self):
        self.pairs_source, self.pairs_target, self.forward, self.all_334_simple_chinese, self.all_simple_chinese, self.synonyms, self.synonyms_334_right, self.synonyms_334_wrong = self.read_related()
        self.synonyms = self.synonyms[:100]
        self.all_simple_chinese = self.all_simple_chinese[:100]
        self.data_for_siamesenet_right_len_limit, self.data_for_siamesenet_wrong_len_limit = self.get_data_for_siamesenet()
        self.data_for_siamesenet_predict = self.get_prediction_data_for_siameset()
        a = self.build_train_data_for_siamesenet()
        # b = self.build_predict_data_for_siamesenet()
        print('over')

    def read_related(self):
        # synonyms
        align_synonym = nlpertools.readtxt_list_all_strip(align_synonym_path)
        synonyms = [i.split()[:20] for idx, i in enumerate(align_synonym) if idx % 2 != 1]
        align_synonym_334_right = nlpertools.readtxt_list_all_strip(align_synonym_334_right_path)
        synonyms_334_right = [i.split() for idx, i in enumerate(align_synonym_334_right)]
        align_synonym_334_wrong = nlpertools.readtxt_list_all_strip(align_synonym_334_wrong_path)
        synonym_334_wrong = [i.split() for idx, i in enumerate(align_synonym_334_wrong)]
        # 错误的不需要任务词，不然与正确的对应会同对同
        synonym_334_wrong = [[j for jdx, j in enumerate(i) if jdx != 1] for i in synonym_334_wrong]
        all_334_simple_chinese = [i.split()[0] for idx, i in enumerate(align_synonym_334_wrong)]
        all_simple_chinese = [i[0] for idx, i in enumerate(synonyms)]
        # 直接加载json
        if os.path.exists('data/pairs_source.json'):
            pairs_source = nlpertools.load_from_json('data/pairs_source.json')
            pairs_target = nlpertools.load_from_json('data/pairs_target.json')
            forward = nlpertools.load_from_json('data/forward.json')
            return pairs_source, pairs_target, forward, all_334_simple_chinese, all_simple_chinese, synonyms, synonyms_334_right, synonym_334_wrong
        # forward
        forward = nlpertools.readtxt_list_all_strip(forward_path)
        # pairs
        pairs = nlpertools.readtxt_list_all_strip(pairs_path)
        pairs_source, pairs_target = [], []
        for pair in pairs:
            line = pair.split(' ||| ')
            pairs_source.append(line[0].split())
            pairs_target.append(line[1].split())
        # check check
        new_pairs_source, new_pairs_target, new_forward = [], [], []        
        for idx, i in enumerate(forward):
            src_tgt_pairs = i.split()
            source_words, target_words = pairs_source[idx], pairs_target[idx]
            need_to_add = True
            if len(src_tgt_pairs) == 0:
                need_to_add = False
            last_src_idx, last_tgt_idx = src_tgt_pairs[-1].split('-')
            if int(last_src_idx) + 1 != len(source_words) or int(last_tgt_idx) + 1 != len(target_words):
                need_to_add = False
            if need_to_add:
                new_pairs_source.append(source_words)
                new_pairs_target.append(target_words)
                new_forward.append(i)
        nlpertools.save_to_json(new_pairs_source, 'data/pairs_source.json')
        nlpertools.save_to_json(new_pairs_target, 'data/pairs_target.json')
        nlpertools.save_to_json(new_forward, 'data/forward.json')
        return new_pairs_source, new_pairs_target, new_forward, all_334_simple_chinese, all_simple_chinese, synonyms, synonyms_334_right, synonym_334_wrong
    
    def build_predict_data_for_siamesenet(self):
        def _build_one_date(i, j, align):
            one_data = dict()
            one_data['input'] = {
                'sentence_a': i[0],
                'sentence_b': j[0],
                'idx': i[2][0],
                'jdx': j[2][0]
            }
            one_data['label'] = 1
            if not align:
                one_data['label'] = 0
                new_jdx = random.randint(0, len(j[0]))
                while new_jdx == jdx:
                    new_jdx = random.randint(0, len(j[0]))
                one_data['input']['jdx'] = new_jdx
            # print(one_data)
            # sys.exit()
            return one_data
            
        def _get_cur_word_data(data_for_siamesenet_len_limit):
            _cur_i3s = [] # 维护一个i[3]的列表，使相同i[3]每次添加收到概率制约 主要解决“我”这个字的问题
            cur_word_data = []
            for i in data_for_siamesenet_len_limit:
                if i[4] == simple_chinese:
                    if i[3] in _cur_i3s:
                        pass
                    else:
                        cur_word_data.append(i)
                        _cur_i3s.append(i[3])
            return cur_word_data
        positive_data = []
        for simple_chinese_idx in tqdm(range(len(self.all_simple_chinese))):
            simple_chinese = self.all_simple_chinese[simple_chinese_idx]
            cur_word_positive_data = []
            cur_word_data = _get_cur_word_data(self.data_for_siamesenet_predict)
            # 正类的构建
            for idx in range(1, len(cur_word_data)):
                one_data = _build_one_date(cur_word_data[1], cur_word_data[idx], align=True)
                cur_word_positive_data.append(one_data)
            positive_data.append(cur_word_positive_data)
        nlpertools.pickle_save(positive_data, 'data/for_predict_data.pkl')
        return positive_data

    def build_train_data_for_siamesenet(self):
        # build 低效的训练数据
        def _build_one_date(i, j, align):
            one_data = dict()
            one_data['input'] = {
                'sentence_a': i[0],
                'sentence_b': j[0],
                'idx': i[2][0],
                'jdx': j[2][0]
            }
            one_data['label'] = 1
            if not align:
                one_data['label'] = 0
                new_jdx = random.randint(0, len(j[0]))
                while new_jdx == jdx:
                    new_jdx = random.randint(0, len(j[0]))
                one_data['input']['jdx'] = new_jdx
            # print(one_data)
            # sys.exit()
            return one_data
            
        def _get_cur_word_data(data_for_siamesenet_len_limit):
            _cur_i3s = [] # 维护一个i[3]的列表，使相同i[3]每次添加收到概率制约 主要解决“我”这个字的问题
            cur_word_data = []
            for i in data_for_siamesenet_len_limit:
                if i[4] == simple_chinese:
                    has_count = _cur_i3s.count(i)
                    if has_count <= 10:
                        cur_word_data.append(i)
                        _cur_i3s.append(i[3])
            return cur_word_data
        
        positive_data, negative_data = [], []
        for simple_chinese_idx in tqdm(range(len(self.all_334_simple_chinese))):
            simple_chinese = self.all_334_simple_chinese[simple_chinese_idx]
            cur_word_positive_data, cur_word_negative_data = [], []
            cur_word_right_data = _get_cur_word_data(self.data_for_siamesenet_right_len_limit)
            cur_word_wrong_data = _get_cur_word_data(self.data_for_siamesenet_wrong_len_limit)
            random.shuffle(cur_word_wrong_data)
            cur_word_wrong_data = cur_word_wrong_data[:3]
            # 正类的构建
            if len(set([i[3] for i in cur_word_right_data])) >= 2:
                word_pairs = []
                for idx in range(len(cur_word_right_data)):
                    for jdx in range(idx + 1, len(cur_word_right_data)):
                        cur_pair = (cur_word_right_data[idx][3], cur_word_right_data[jdx][3])
                        if cur_word_right_data[idx][3] != cur_word_right_data[jdx][3] and cur_pair not in word_pairs:
            
                            word_pairs.append(cur_pair)
                            word_pairs.append((cur_word_right_data[jdx][3], cur_word_right_data[idx][3]))
                            one_data = _build_one_date(cur_word_right_data[idx], cur_word_right_data[jdx], align=True)
                            cur_word_positive_data.append(one_data)
                        else:
                            pass

            # 负类的构建
            # a1 a2 i k
                        one_data = _build_one_date(cur_word_right_data[idx], cur_word_right_data[jdx], align=False)
                        cur_word_negative_data.append(one_data)
            # a1 b1 i j
            if len(cur_word_right_data) > 0 and len(cur_word_wrong_data) > 0:
                for idx in range(len(cur_word_right_data)):
                    for jdx in range(len(cur_word_wrong_data)):
                        one_data = _build_one_date(cur_word_right_data[idx], cur_word_wrong_data[jdx], align=False)
                        cur_word_negative_data.append(one_data)
            # a1 b1 i k
                        one_data = _build_one_date(cur_word_right_data[idx], cur_word_wrong_data[jdx], align=False)
                        cur_word_negative_data.append(one_data)
            # b1 b2 i j
            if len(set([i[3] for i in cur_word_wrong_data])) >= 2:
                for idx in range(len(cur_word_wrong_data)):
                    for jdx in range(idx, len(cur_word_wrong_data)):
                        one_data = _build_one_date(cur_word_wrong_data[idx], cur_word_wrong_data[jdx], align=False)
                        cur_word_negative_data.append(one_data)
            # b1 b2 i k
                        one_data = _build_one_date(cur_word_wrong_data[idx], cur_word_wrong_data[jdx], align=False)
                        cur_word_negative_data.append(one_data)
            if cur_word_positive_data:
                positive_data.append(cur_word_positive_data)
            else:
                pass
                # print()
            if cur_word_negative_data:
                negative_data.append(cur_word_negative_data)
        print('positive{}'.format(len(positive_data)))
        print('negative{}'.format(len(negative_data)))
        nlpertools.pickle_save(positive_data, 'data/positive_data.pkl')
        nlpertools.pickle_save(negative_data, 'data/negative_data.pkl')
        return positive_data, negative_data


    def get_prediction_data_for_siameset(self):
        if os.path.exists('data/data_for_siamesenet_predict.json'):
            data_for_siamesenet_predict = nlpertools.load_from_json('data/data_for_siamesenet_predict.json')
        else:
            data_for_siamesenet_predict = self.build_primary_data_for_get(mode="predict", train_or_predict_mode='predict')
        # data_for_siamesenet_wrong_len_limit = [i for i in data_for_siamesenet_wrong if len(i[0]) <= sentence_len_limit and len(i[1]) <= sentence_len_limit]
        to_write = ['\t'.join([' '.join(i[0]), ' '.join(i[1]), '{} {}'.format(i[2][0], i[2][1]), i[3], i[4]]) for i in data_for_siamesenet_predict]
        nlpertools.writetxt_w_list(to_write, 'data/predict_to_write.tsv', num_lf=1)
        return data_for_siamesenet_predict

    def get_data_for_siamesenet(self):
        # 获取所有是同义词的句子，与不是同义词的句子，excel可读
        # 正例
        if os.path.exists('data/data_for_siamesenet_right.json'):
            data_for_siamesenet_right = nlpertools.load_from_json('data/data_for_siamesenet_right.json')
        else:
            data_for_siamesenet_right = self.build_primary_data_for_get(mode="right")
        data_for_siamesenet_right_len_limit = [i for i in data_for_siamesenet_right if len(i[0]) <= sentence_len_limit and len(i[1]) <= sentence_len_limit]
        to_write = ['\t'.join([' '.join(i[0]), ' '.join(i[1]), '{} {}'.format(i[2][0], i[2][1]), i[3], i[4]]) for i in data_for_siamesenet_right_len_limit]
        nlpertools.writetxt_w_list(to_write, 'data/positive_to_write.tsv', num_lf=1)
        # 负例
        if os.path.exists('data/data_for_siamesenet_wrong.json'):
            data_for_siamesenet_wrong = nlpertools.load_from_json('data/data_for_siamesenet_wrong.json')
        else:
            data_for_siamesenet_wrong = self.build_primary_data_for_get(mode="wrong")
        data_for_siamesenet_wrong_len_limit = [i for i in data_for_siamesenet_wrong if len(i[0]) <= sentence_len_limit and len(i[1]) <= sentence_len_limit]
        to_write = ['\t'.join([' '.join(i[0]), ' '.join(i[1]), '{} {}'.format(i[2][0], i[2][1]), i[3], i[4]]) for i in data_for_siamesenet_wrong_len_limit]
        nlpertools.writetxt_w_list(to_write, 'data/negative_to_write.tsv', num_lf=1)
        return data_for_siamesenet_right_len_limit, data_for_siamesenet_wrong_len_limit

    def build_primary_data_for_get(self, mode, train_or_predict_mode='train'):
        # 按照正确同义词与错误同义词 生成 古-白 对 json格式，可读
        if mode == 'right':
            synonyms_334 = self.synonyms_334_right
            save_path = 'data/data_for_siamesenet_right.json'
        elif mode == 'wrong':
            synonyms_334 = self.synonyms_334_wrong
            save_path = 'data/data_for_siamesenet_wrong.json'
        else:
            synonyms_334 = self.synonyms
            save_path = 'data/data_for_siamesenet_predict.json'
        # source->trandition 古文  target->simple 白话文
        input_one_zu = []
        for line in synonyms_334:
            # print(line)
            simple_chinese = line[0]
            has_this_simple_chinese_sentence_idxs = []
            for idx, i in enumerate(self.pairs_target):
                if simple_chinese in i:
                    has_this_simple_chinese_sentence_idxs.append(idx)
            for tradition_chinese in line[1:]:
                for idx in has_this_simple_chinese_sentence_idxs:
                    cur_forward = self.forward[idx]
                    cur_src_tgt = cur_forward.split()
                    cur_src_tgt_str_list = [m.split('-') for m in cur_src_tgt]
                    cur_src_tgt_list = [[int(n) for n in m] for m in cur_src_tgt_str_list]
                    cur_simple_chinese_sentence = self.pairs_target[idx]
                    cur_trandition_chinese_sentence = self.pairs_source[idx]
                    if tradition_chinese in cur_trandition_chinese_sentence:
                        # 上面一行是判断是否存在。但由于目标是0-len(sentence)，每个只有一个的，所以下面用现代文。
                        for jdx, j in enumerate(cur_simple_chinese_sentence):
                            # print(j)
                            # print(cur_simple_chinese_sentence[[m[1] for m in cur_src_tgt_list].index(jdx)])
                            
                            if j != simple_chinese:
                                continue
                            # cur_src_tgt_list[jdx] 's [i,j] should satisfied i's trandition == tradition_chinese
                            # 因为存在缺失，所以得用 tgt为jdx是的索引号(tgt_mdx)，而不能直接用jdx
                            if tradition_chinese == '，':
                                pass
                                # print()
                            if jdx not in [m[1] for m in cur_src_tgt_list]:
                                continue
                            tgt_mdx = [m[1] for m in cur_src_tgt_list].index(jdx)
                            # print(simple_chinese)
                            # print(cur_simple_chinese_sentence[cur_src_tgt_list[tgt_mdx][1]])
                            if tradition_chinese == cur_trandition_chinese_sentence[cur_src_tgt_list[tgt_mdx][0]]:
                                input_one_zu.append([cur_trandition_chinese_sentence, cur_simple_chinese_sentence, cur_src_tgt_list[tgt_mdx], tradition_chinese, simple_chinese])
                                break
                                # print(cur_trandition_chinese_sentence)
                                # print(cur_simple_chinese_sentence)
                                # print(cur_src_tgt_list[tgt_mdx])
                                # print(tradition_chinese)
                                # print(cur_trandition_chinese_sentence[cur_src_tgt_list[tgt_mdx][0]])
                                # print(simple_chinese)
                                # print(cur_simple_chinese_sentence[cur_src_tgt_list[tgt_mdx][1]])
                # if train_or_predict_mode == 'predict':
                #     break
        nlpertools.save_to_json(input_one_zu, save_path)
        return input_one_zu


dp = DataProcessing()

