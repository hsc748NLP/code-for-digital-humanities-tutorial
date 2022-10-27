import numpy as np

'''
计算向量相似度的几种算法
待填坑：
1.getCenter_weight 根据重心计算类别中心
'''

def similarity_cos(pre_vector,back_vector):
    prev = np.array(pre_vector)
    backv = np.array(back_vector)
    cos_value = np.dot(prev, backv) / (np.linalg.norm(prev) * (np.linalg.norm(backv)))
    return cos_value

def similarity_euclidean(pre_vector,back_vector):
    prev = np.array(pre_vector)
    backv = np.array(back_vector)
    euclidean_value = np.sqrt(np.sum(np.square(prev-backv)))
    return euclidean_value

def similarity_manhattan(pre_vector,back_vector):
    prev = np.array(pre_vector)
    backv = np.array(back_vector)
    manhattan_value = np.sum(np.abs(prev-backv))
    return manhattan_value

def getCenter_mean(cluster_data,origin_center):
    #print(cluster_data)
    origin_array=np.zeros(np.array(origin_center).shape,dtype=np.float)
    tmp_data=np.zeros(np.array(origin_center).shape,dtype=np.float)
    for data in cluster_data:
        tmp_data=tmp_data+np.array(data)
    if((origin_array==tmp_data).all()):
        return origin_center
    else:
        return tmp_data/len(cluster_data)


def getCenter_weight(cluster_data,origin_center):
    pass

class editDistance():
    def __init__(self):
        self.wordarray=[]

    def initArray(self,firstStr,secondStr):
        flenth=len(firstStr)
        slenth=len(secondStr)
        self.wordarray=[[0 for _i_ in range(slenth+1)] for _j_ in range(flenth+1)]
        for i in range(flenth+1):
            self.wordarray[i][0]=i
        for i in range(slenth+1):
            self.wordarray[0][i]=i

    def getDistance(self,firstStr,secondStr):
        flenth = len(firstStr)
        slenth = len(secondStr)
        self.initArray(firstStr,secondStr)
        for i in range(flenth):
            for j in range(slenth):
                sameflag=(firstStr[i]==secondStr[j])
                self.wordarray[i+1][j+1]=self.getMin(i+1,j+1,sameflag)
        return self.wordarray[flenth][slenth]

    def getMin(self,i,j,sameflag):
        if(sameflag):
            tmp_list=[self.wordarray[i][j-1]+1,self.wordarray[i-1][j-1],self.wordarray[i-1][j]+1]
        else:
            tmp_list=[self.wordarray[i][j-1]+1,self.wordarray[i-1][j-1]+1,self.wordarray[i-1][j]+1]
        return min(tmp_list)


if __name__=="__main__":
    ed=editDistance()
    str1="i am chinese!"
    str2="i am a chinese!"
    print(ed.getDistance(str1,str2))
    pass
