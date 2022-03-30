import codecs
import xlrd
import jieba


def readtxt(path):
    with codecs.open(path,'r',encoding='utf-8') as f:
        line=f.readlines()  #line为一个列表 一行一个元素
        line = [x.strip('\r\n') for x in line]
        #print(line)
        return line


# jieba分词（去除停用词后每行分词）
def cut_words(data, stopwords):
    cut_word=[]
    words=''
    for line in data:
        jieba.load_userdict('./heritage_entity.txt')
        slist=jieba.cut(line,cut_all=False)
        #slist = [x.strip() for x in list(slist) if x.strip() != '\xa0' or x.strip() != '\u3000' or x.strip() !='\ue81b'\
         #      or x.strip() !=' ']
        for key in slist:
            if key not in stopwords and key !=' ' and key != '\xa0' and key !='\u3000' and key !='\ue81b':
                words+=key+' '
        #output=" ".join(words)
        words+='\n'
    cut_word.append(words)
    return cut_word


# 去除停用词、重复词词总量
def wordlist(cut_word, stopwords):
    final=[]
    for line in data:
        slist=jieba.cut(line,cut_all=False)
        output=" ".join(list(slist))
        for key in output.split(' '):
            if (key not in stopwords) and (key not in cut_word):
                final.append(key)
    return final


def write(path,data):
    with codecs.open(path,'a','utf-8')as f:
        for i in data:
            f.write(str(i))

def corpus(data):
    final = []
    for line in data:
        l = line.strip(' ').split(' ')
        res = [x.strip() for x in l if x.strip() != '\xa0' or x.strip() != '\u3000' or x.strip() !='\ue81b'\
               or x.strip() !=' ']
        cor = " ".join(res)
        final.append(cor)
    #print(final)
    return final

def main():
    path1 = "./title_info.txt"
    path2 = "./cut_words.txt"
    path3 = "./CW_noplace.txt"   # 基于实体词典分词
    stopwords_path = "./停用词.txt"
    data = readtxt(path1)
    stopwords = readtxt(stopwords_path)
    #stopwords = [x.replace('\r\n', '') for x in stopwords]

    #cut_word = cut_words(data, stopwords)
    cut_word_entity = cut_words(data, stopwords)
    print(cut_word_entity)
    #print(corpus(cut_word_entity))
    #write(path2, cut_word)
    write(path3, cut_word_entity)


if __name__=="__main__":
        main()
