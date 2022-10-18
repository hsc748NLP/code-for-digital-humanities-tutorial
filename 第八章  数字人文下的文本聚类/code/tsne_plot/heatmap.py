import pandas as pd
import seaborn as sns
from pylab import *

mpl.rcParams['font.sans-serif'] = ['SimHei']


labels = ['民间文学', '传统音乐', '传统舞蹈', '传统戏剧', '曲艺', '传统体育、游艺与杂技', '传统美术', \
          '传统技艺', '传统医药', '民俗']
classes = ['1', '2', '3', '4', '5', '6']
sum=[191,352,278,422,175,102,307,427,89,347]
six = [927, 693, 235, 377, 209, 249]
eight = [174, 141, 693, 920, 209, 198, 110, 245]

def hotmap(X):
    '''for r,i in zip(range(len(eight)),X):
            print(i)
            for j in range(len(i)):
                i[j]=i[j]/eight[r]'''
    for i in X:
        print(i)
        for j,k in zip(range(len(i)),range(len(sum))):
            i[j]=i[j]/sum[k]
    #print(X)
    X = np.transpose(X)
    print(X)
    dt = pd.DataFrame(X, columns=['类1', '类2', '类3', '类4', '类5', '类6'], index=labels)
    print(dt)
    # pt = dt.pivot(index=labels,columns=classes,values=0)
    # cmap用matplotlib colormap
    ax = sns.heatmap(dt, cmap='YlGnBu')
    # rainbow为 matplotlib 的colormap名称
    ax.set_title('six classes heatmap')
    # ax.set_xlabel('classes')
    # ax.set_ylabel('')
    plt.show()


def main():
    w2v_six_class = [[82,127,55,182,31,31,118,158,28,115],[46,98,97,86,84,42,79,95,28,38],[7,31,25,45,2,8,16,31,11,59],[28,45,54,54,19,12,48,61,15,41],\
             [13,4,12,38,6,6,16,29,3,82],[15,48,35,17,33,3,30,52,4,12]]
    w2v_eight = [[17,28,28,24,12,8,30,19,2,6],[12,23,13,8,30,2,14,31,3,5],[46,98,97,86,84,42,79,95,28,38],\
                 [87,127,52,179,27,31,121,157,28,111],[13,4,12,38,6,6,16,29,3,82],[6,25,17,39,2,7,13,25,11,53],\
                 [3,25,22,9,4,1,16,22,1,7],[7,23,37,39,10,5,18,48,13,45]]
    w2v_ten = [[12,23,13,8,30,2,14,31,3,5],[80,118,45,152,26,27,113,146,25,94],[46,98,97,86,84,42,79,95,28,38],\
               [12,4,12,38,6,6,16,29,3,82],[7,19,18,40,0,8,9,18,2,52],[17,27,28,24,12,8,30,17,2,4],[3,25,22,9,3,1,16,21,1,7],\
               [1,8,4,7,2,0,5,14,9,9],[0,1,0,0,1,0,0,1,0,0],[13,30,39,58,11,8,25,54,16,56]]
    hotmap(w2v_six_class)

if __name__ == '__main__':
    main()

