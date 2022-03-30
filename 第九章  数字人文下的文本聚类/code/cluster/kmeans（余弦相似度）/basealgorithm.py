import random

from cluster.basefunction import similarity_cos,similarity_euclidean,similarity_manhattan
from cluster.basefunction import getCenter_mean

'''
此模块待填坑：
1.缺少层次聚类法函数
2.缺少报错信息raise
'''

class kmeans():
    def __init__(self,k=5,max_iteration=100000,min_center=0.001,select_decision="random",getcenter_function="mean",similarity_function="cos"):
        self.k = k
        self.max_iteration = max_iteration
        self.min_center = min_center
        self.select_decision = select_decision
        self.getcenter_function = getcenter_function
        self.similarity_function = similarity_function
        self.clustercenter = [[] for _ in range(k) ]
        self.clusterdata=[ [] for _ in range(k) ]
        pass

    def similarity(self,point_data):
        if(self.similarity_function=="cos"):
            result=[ similarity_cos(self.clustercenter[i],point_data) for i in range(self.k)]
            return result.index(max(result))
        elif(self.similarity_function=="euclidean"):
            result = [similarity_euclidean(self.clustercenter[i], point_data) for i in range(self.k)]
            return result.index(min(result))
        elif (self.similarity_function == "manhattan"):
            result = [similarity_manhattan(self.clustercenter[i], point_data) for i in range(self.k)]
            return result.index(min(result))
        else:
            print("similarity 参数错误！")

    def getCenter(self,tmp_data,origin_center):
        if(self.getcenter_function=="mean"):
            return getCenter_mean(tmp_data,origin_center)
        else:
            print("similarity 参数错误！")

    def selected_point(self,data):
        if(self.select_decision=="random"):
            self.selected_random(data)
        else:
            print(" select_decision 参数错误！")

    def selected_random(self,data):
        tmp_set=set([])
        if(len(data)<self.k):
            for i in range(self.k):
                self.clustercenter[i]=[i]*len(data[0])
        else:
            for i in range(self.k):
                tmp_index=random.randint(0,len(data))
                while(tmp_index in tmp_set):
                    tmp_index = random.randint(0, len(data))
                self.clustercenter[i]=data[tmp_index]
        pass

    def train_predict(self,data):
        if(not data):
            print("数据为空！")
        self.selected_point(data)
        for tmp_i in range(self.max_iteration):
            tmp_clustercenter=self.clustercenter.copy()
            for i in range(len(data)):
                tmp_index=self.similarity(data[i])
                self.clusterdata[tmp_index].append(data[i])
            for i in range(self.k):
                tmp_clustercenter[i]=self.getCenter(self.clusterdata[i],self.clustercenter[i])
            change_center=sum([ similarity_euclidean(tmp_clustercenter[i],self.clustercenter[i]) for i in range(self.k)])
            if(change_center<self.min_center):
                break
            self.clustercenter=tmp_clustercenter
        self.clusterdata = [[] for _ in range(self.k)]
        result_data=self.predict(data)
        return result_data


    def predict(self,data):
        return [ self.similarity(tmp_data) for tmp_data in data]


class hierarchical_Cluster():
    def __init__(self):
        pass

    def similarity(self):
        pass

    def train_predict(self,data):
        pass

    def predict(self,data):
        pass
