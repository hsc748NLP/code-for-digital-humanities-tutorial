import codecs
import re
import xlrd


def readxls(path):
    xl=xlrd.open_workbook(path)
    sheet=xl.sheets()[0]
    data=[]
    for i in range(1, sheet.nrows):
        data.append(sheet.row_values(i))
    return data


def form(data):
    for i in data:
        if i[1]!='':
            i[1] = '\t'+'SC:'+i[1]
        if i[2]!='':
            i[2] = '\t' + 'USE:' + i[2]
        if i[3]!='':
            i[3] = '\t' + 'UF:' + i[3]
        if i[4]!='':
            i[4] = '\t' + 'AD:' + i[4]
        if i[5] != '':
            i[5] = '\t' + 'NT:' + i[5]
        if i[6] != '':
            i[6] = '\t' + 'BT:' + i[6]
        if i[7] != '':
            i[7] = '\t' + 'RT:' + i[7]
    for j in data:
        for k in range(len(j)):
            j[k]=j[k].replace('/','\n\t   ')
    return data

def write(path,data):
    with codecs.open(path,'w',encoding='utf8') as f:
        for i in data:
            for j in i:
                if j != '':
                    f.write(j+'\n')
            f.write('\n')


def main():
    path=r'C:\Users\lenovo\Desktop\情报语言学\书法.xlsx'
    path_put = r'C:\Users\lenovo\Desktop\情报语言学\书法.txt'
    data = readxls(path)
    write(path_put,form(data))


main()