import codecs
import xlrd
import re


lables = ['<ICH-TITLE>', '<ICH-TITLE/>', '<ICH-TERM>', '<ICH-TERM/>', '<ICH-INHERITOR>', '<ICH-INHERITOR/>',\
                   '<ICH-PLACE>', '<ICH-PLACE/>', '<ICH-WORKS>', '<ICH-WORKS/>', '<ICH-INST>', '<ICH-INST/>',\
                   '<ICH-FOOD>', '<ICH-FOOD/>']

# 读取标题和详细信息
def readxls(path, col):
    xl = xlrd.open_workbook(path)
    sheet = xl.sheets()[0]
    data = list(sheet.col_values(col))[1:]
    return data

# 提取实体
def ex_info_title(title,info):
    entity = []
    for i in title:   # 标题实体
        pattern = re.compile(r'<ICH-TITLE>(.*?)<ICH-TITLE/>', re.S)  # 非遗项目
        pTERM = re.compile(r'<ICH-TERM>(.*?)<ICH-TERM/>', re.S)  # 术语
        titledata = pattern.findall(i)
        termdata = pTERM.findall(i)
        for j in titledata:
            entity.append(j)
        for j in termdata:
            entity.append(j)

    for i in info:     # 详细介绍实体
        pINHERITOR = re.compile(r'<ICH-INHERITOR>(.*?)<ICH-INHERITOR/>', re.S)  # 继承人姓名
        #pPLACE = re.compile(r'<ICH-PLACE>(.*?)<ICH-PLACE/>', re.S)     # 地名
        pWORKS = re.compile(r'<ICH-WORKS>(.*?)<ICH-WORKS/>', re.S)    # 作品名
        pINST = re.compile(r'<ICH-INST>(.*?)<ICH-INST/>', re.S)     # 工具名
        pFOOD = re.compile(r'<ICH-FOOD>(.*?)<ICH-FOOD/>', re.S)     # 食物名
        FOODdata = pFOOD.findall(i)
        INSTdata = pINST.findall(i)
        WORKSdata = pWORKS.findall(i)
        #PLACEdata = pPLACE.findall(i)
        INHERITORdata = pINHERITOR.findall(i)
        for j in FOODdata:
            entity.append(j)
        for j in INSTdata:
            entity.append(j)
        #for j in PLACEdata:
         #   entity.append(j)
        for j in WORKSdata:
            entity.append(j)
        for j in INHERITORdata:
            entity.append(j)
    return entity
    '''final = []
    for i, k in zip(data, range(len(data))):
        entity.append([])     # 二维列表  一维一个项目信息
        pattern = re.compile(r'<ICH-TITLE>(.*?)<ICH-TITLE/>', re.S)  # 非遗项目
        pTERM = re.compile(r'<ICH-TERM>(.*?)<ICH-TERM/>', re.S)  # 术语
        pINHERITOR = re.compile(r'<ICH-INHERITOR>(.*?)<ICH-INHERITOR/>', re.S)  # 继承人姓名
        #pPLACE = re.compile(r'<ICH-PLACE>(.*?)<ICH-PLACE/>', re.S)  # 地名
        pWORKS = re.compile(r'<ICH-WORKS>(.*?)<ICH-WORKS/>', re.S)  # 作品名
        pINST = re.compile(r'<ICH-INST>(.*?)<ICH-INST/>', re.S)  # 工具名
        pFOOD = re.compile(r'<ICH-FOOD>(.*?)<ICH-FOOD/>', re.S)  # 食物名
        FOODdata = pFOOD.findall(i)
        INSTdata = pINST.findall(i)
        WORKSdata = pWORKS.findall(i)
        #PLACEdata = pPLACE.findall(i)
        INHERITORdata = pINHERITOR.findall(i)
        titledata = pattern.findall(i)
        termdata = pTERM.findall(i)
        for j in titledata:
            entity[k].append(j)
        for j in termdata:
            entity[k].append(j)
        for j in FOODdata:
            entity[k].append(j)
        for j in INSTdata:
            entity[k].append(j)
        #for j in PLACEdata:
         #   entity[k].append(j)
        for j in WORKSdata:
            entity[k].append(j)
        for j in INHERITORdata:
            entity[k].append(j)
        m = " ".join(entity[k])    # 每行信息 每个实体用空格隔开
        final.append(m)
    #print(final)
    return final'''

# 去掉词性标注
def throw_tag(data):
    finalentity = []
    entity_corpus = []
    m =[]
    word = ['/n', '/ns', '/nz', '/vg', '/vi', '/wyz', '/wyy', '/v', '/ng', '/nr1', '/b', '/a', '/gms', '/nr',\
            '/nr2', '/f', '/vn', '/cc', '/k', '/vi', '/ad', '/tg', '/m', '/wn', '/p', '/udel', '/udeng', '/uguo',\
           '/wn', '/uzhi', '/vshi', '/rz']
    '''for k, line in zip(range(len(data)), data):
        m.append([])
        l = line.split(' ')
        for i in l:
            i = re.sub(r'[/n/nz/ns/vg/vi/wyz/wyy/v/ng/nr1/b/a/gms/nr/nr2/f/vn/cc/k/vi/ad/tg/m/wn/p/udel/udeng/uguo\
        /wn/uzhi/vshi/rz/q_4《》【】8R]', '', str(i))             # 去掉词性标注  需要字符型 i为每个实体
        #print(i)
        # if i not in finalentity and i !='':               # 去重
            # finalentity.append(i))
        #m = " ".join(i)
            m[k].append(i)    # 每条信息
            #print(m[k])
        res = " ".join(m[k])
            #print(res)
        entity_corpus.append(res)
        print(entity_corpus)'''
    for i in data:
        i = re.sub(r'[/n/nz/ns/vg/vi/wyz/wyy/v/ng/nr1/b/a/gms/nr/nr2/f/vn/cc/k/vi/ad/tg/m/wn/p/udel/udeng/uguo\
                /wn/uzhi/vshi/rz/q_4《》【】8R]', '', i)  # 去掉词性标注  需要字符型 i为每个实体
        if i not in finalentity and i != '':
            finalentity.append(i)
        print(finalentity)
    return finalentity


def uni(title, info):
    uni_lis = []
    for i, j in zip(title, info):
        if i != '' and j != '':
            n = i+' '+j
            uni_lis.append(n)
    return uni_lis


def writetxt(path, data):
    with codecs.open(path, 'a', 'utf-8') as f:
        for i in data:
            f.write(str(i)+'\n')


def corpus(data):
    final = []
    for line in data:
        l = line.split(' ')
        res = [x.strip() for x in l if x.strip() != '']
        cor = " ".join(res)
        final.append(cor)
    #print(final[1])
    return final


def main():
    pathxls = "./非遗信息 全.xlsx"
    pathtxt = "./heritage_entity_noplace.txt"
    path_only_entity = "./only_entity_corpus.txt"
    title = readxls(pathxls, 13)
    info = readxls(pathxls, 14)
    entity = ex_info_title(title, info)
    writetxt(pathtxt, throw_tag(entity))
    #data = corpus(uni(title, info))
    #final = ex_info_title(data)
    #entity_cor = throw_tag(final)
    #writetxt(path_only_entity, entity_cor)
    #print(data)

main()
