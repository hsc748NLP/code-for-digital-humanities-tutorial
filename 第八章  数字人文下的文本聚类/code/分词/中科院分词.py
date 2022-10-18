import pynlpir
pynlpir.open()
import codecs
from ctypes import c_char_p


def readtxt(path):
    with codecs.open(path,'r',encoding='utf-8') as f:
        line=f.readlines()  #line为一个列表 一行一个元素
        line = [x.strip('\r\n') for x in line]
        #print(line)
        return line

def cutwords(data,stopwords,en):
    words=''
    cut_word= []
    for word in en:
        pynlpir.nlpir.ImportUserDict(c_char_p(word.encode()))
    for line in data:
        slist = pynlpir.segment(line, pos_tagging=False)
        print(slist)
        for key in slist:
            if key not in stopwords and key != ' ' and key != '\xa0' and key != '\u3000' and key != '\ue81b':
                words += key+' '
        # output=" ".join(words)
        words += '\n'
    cut_word.append(words)
    print(cut_word)
    return cut_word


def write(path,data):
    with codecs.open(path,'a','utf-8')as f:
        for i in data:
            f.write(str(i))


def main():
    path1 = "D:\我\非遗\example.txt"
    stopwords_path = "D:\我\非遗\停用词.txt"
    path_entity = "D:\我\非遗\heritage_entity.txt"
    data = readtxt(path1)
    stopwords = readtxt(stopwords_path)
    entity = readtxt(path_entity)
    #print(entity)
    cutwords(data,stopwords,entity)


if __name__ == '__main__':
    main()