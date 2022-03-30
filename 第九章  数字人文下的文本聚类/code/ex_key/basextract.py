import math
from gensim.models.word2vec import Word2Vec

class tfidf:
    def __init__(self):
        self.idf={}
        self.keyData={}

    def load_data(self,data):
        tmp_dict={}
        for key in data.keys():
            word_list=data[key].split(" ")
            for word in word_list:
                tmp_dict[word]=tmp_dict.get(word,[])
                tmp_dict[word].append(key)
                self.keyData[key]=self.keyData.get(key,{})
                self.keyData[key][word]=self.keyData[key].get(word,0)+1
        docs_len=len(data.keys())
        for word in tmp_dict.keys():
            self.idf[word]=math.log(docs_len-len(set(tmp_dict[word]))+0.5)-math.log(len(set(tmp_dict[word]))+0.5)
        for key in self.keyData.keys():
            for word in self.keyData[key].keys():
                self.keyData[key][word]=self.keyData[key][word]*self.idf[word]
        return self

    def extractKeyword(self, data ,n):
        self.load_data(data)
        tmp_dict={}
        for key in self.keyData.keys():
            sort_list=sorted([[word,self.keyData[key][word]] for word in self.keyData[key].keys()],key=lambda x:x[1],reverse=True)
            tmp_dict[key]=[ word for word,_ in sort_list][0:n]
        return tmp_dict

    def get_vector(self):
        pass

class word2vect_model:
    def __init__(self,path,min_count=2,embedding_dim=64,max_vocab_size=3000,window_size=5):
        self.min_count=min_count
        self.embedding_dim=embedding_dim
        self.max_vocab_size=max_vocab_size
        self.window_size=window_size
        self.modelpath=path

    def train(self,sentences):
        self.model=Word2Vec(sentences,size=self.embedding_dim, window=self.window_size, min_count=self.min_count,
                              max_vocab_size=self.max_vocab_size)
        self.model.save(self.modelpath)
        return self

    def load_model(self):
        self.model=Word2Vec.load(self.modelpath)
        return self

    def get_vector(self,word):
        return self.model[word]

    def is_haveword(self,word):
        try:
            vector=self.model[word]
            return True
        except:
            return False

if __name__=="__main__":
    import jieba
    import re
    '''
    crime_re = re.compile(r"(.*?罪)、{0,1}")
    docs={}
    model=tfidf()
    f = open("C:/Users/sfe_williamsL/Desktop/毕业论文/result_id.txt", "rt", encoding="utf-8")
    for line in f.readlines():
        datas = line.split("\t")
        if (len(datas) < 2):
            continue
        crimes=crime_re.findall(datas[1])
        content = datas[3]
        for ctype in crimes:
            ctype=ctype.strip("、")
            docs[ctype] = docs.get(ctype,"")+" ".join([word for word in jieba.cut(content) if (len(word) > 1)])
    f.close()
    tmp_dict=model.extractKeyword(docs,10)
    w=open("C:/Users/sfe_williamsL/Desktop/毕业论文/keyword_10.txt","wt",encoding="utf-8")
    print(tmp_dict)
    for key in tmp_dict.keys():
        w.write("\n".join(tmp_dict[key])+"\n")
    w.close()
    
    
    #word2vec计算
    doc_data = []
    i = 0
    f = open("C:/Users/sfe_williamsL/Desktop/毕业论文/result_id.txt", "rt", encoding="utf-8")
    for line in f.readlines():
        tmp_data = []
        datas = line.split("\t")
        if (len(datas) < 2):
            continue
        docid = datas[0]
        content = datas[3]
        word_list =list(jieba.cut(content))
        doc_data.append(word_list)
        i = i + 1
    f.close()
    wm=word2vect_model(path="C:/Users/sfe_williamsL/Desktop/毕业论文/data/word2vect_8",embedding_dim=8)
    wm.train(doc_data)
    '''
    docs = {}
    model = tfidf()
    f = open("D:\我\非遗\高维聚类结果\\400维六类数据\\总.txt", "rt", encoding="utf-8")
    i=0
    for line in f.readlines():
        if(not line.replace("\r","").replace("\n","")):
            continue
        i=i+1
        content = line.replace("\r","").replace("\n","")
        docs[i] = docs.get(i,content)
    f.close()
    print(docs)
    tmp_dict = model.extractKeyword(docs, 15)
    w = open("D:\我\非遗\高维聚类结果\\400维六类数据\\keyword_15.txt", "wt", encoding="utf-8")
    print(tmp_dict)
    for key in tmp_dict.keys():
        w.write("\t".join(tmp_dict[key]) + "\n")
    w.close()