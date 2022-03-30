'''
author: 纪有书
train.py
'''
import torch
import torch.nn as nn
from model import SiameseNetwork
import torch.utils.data as Data
import datafeature
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, classification_report
import nlpertools
import datahelper
print(torch.cuda.is_available())

device = torch.device('cuda')
model_save_path = 'output'
max_seq_len = 40
batchsize = 32
epoch_nums = 3
trainset_weight = 0.9
train_test_split_seed = 44
# sentence_a = torch.randint(3, 12, (6, 2)).cuda()
# sentence_b = torch.randint(3, 12, (6, 2)).cuda()
# idx = torch.randn((2, 6)).cuda()
# jdx = torch.randn((2, 6)).cuda()
# labels = torch.Tensor([1, 0]).cuda() # (batch,)

# 开始一轮训练/评估
def run_epoch(model, data_loader, criterion, optimizer):
    # 计数，用于控制打印loss的时机
    num = 0
    # 定义存放原始标签与预测标签的数组
    y_true, y_pre = [], []
    # 循环取出一组训练数据
    for (sentence_a, sentence_b, idx, jdx, label) in tqdm(data_loader):
        # 将数据输入模型，得到输出
        out = model(sentence_a.cuda(), sentence_b.cuda(), idx.cuda(), jdx.cuda())
        # 用损失函数计算损失
        loss = criterion(out, label.cuda()).cuda()
        # 如果是训练过程，根据梯度更新参数
        if optimizer:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # 计数+1
        num += 1
        # 每20条数据打印一次loss
        if num % 20 == 0:
            print(loss)
        # 把模型预测结果保存进数组
        y_pre.extend([0 if i < 0.5 else 1 for i in out.cpu().detach().numpy()])
        # 把原始标签保存进数组
        y_true.extend([int(i) for i in label])
    # 计算模型准确率
    acc = accuracy_score(y_true, y_pre)
    # 计算prf等
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pre, labels=[1, 0])
    # 打印prf等
    print(classification_report(y_true, y_pre, labels=[1, 0]))
    print('{} acc： {}'.format('train' if optimizer else 'eval', acc))
    print('{} p： {}'.format('train' if optimizer else 'eval', p))
    print('{} r： {}'.format('train' if optimizer else 'eval', r))
    print('{} f： {}'.format('train' if optimizer else 'eval', f1))
    # 返回正例p的得分
    return p[0]

def train():
    # ★读取训练数据，并将其转换为模型的输入★
    id2char, char2id, dataset = datafeature.read_data(max_seq_len, negative_path='data/negative_data.json', positive_path='data/positive_data.json')
    # 将数据集分为训练和测试两个部分
    trainset, evalset = Data.random_split(dataset,[round(trainset_weight * len(dataset)),len(dataset) - round(trainset_weight * len(dataset))],generator=torch.Generator().manual_seed(train_test_split_seed)) 
    # 打印训练集和测试集的数量
    print('trainset size: {}\ntestset size: {}'.format(len(dataset), len(evalset)))
    # 用torch的DataLoader类实现训练数据的取出
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batchsize, shuffle=True)
    # 用torch的DataLoader类实现测试数据的取出
    dev_loader = torch.utils.data.DataLoader(evalset, batch_size=batchsize, shuffle=True)
    # 实例化孪生网络模型，并存放到gpu上
    model = SiameseNetwork(vocab_size=len(id2char)).cuda()
    # 定义学习率为0.01
    lr = 0.01
    # 定义优化器中的动量为0.9，在随机梯度下降中用到
    momentum = 0.9
    # 定义损失函数为二元交叉熵
    criterion = nn.BCELoss()
    # 定义模型的优化器为随机梯度下降
    optimizer = torch.optim.SGD(model.parameters(), lr=lr,
                                    momentum=momentum)
    # 定义存放训练loss的列表
    train_loss = []
    # 定义二得分，用于判断是否保存模型
    best_score = 0
    # 定义保存得分最高的训练轮次
    best_epoch = 0
    # 开始一轮一轮地训练
    for epoch in range(1, epoch_nums + 1):
        # 打印当前轮次
        print(epoch)
        # 将模型参数设置问训练模式，模型参数更新
        model.train()
        # 打印训练标志
        print('traning')
        # ★开始一轮训练★
        _ = run_epoch(model, train_loader, criterion, optimizer)
        # 打印评估标志
        print('evaling')
        # 模型评估模式
        model.eval()
        # 开始一轮评估，返回当前轮次模型的得分
        score = run_epoch(model, dev_loader, criterion, None)
        # 如果当前得分最高，保存模型
        if score >= best_score:
            best_score = score
            best_epoch = epoch
            torch.save(model.state_dict(), '{}/model_{}.pt'.format(model_save_path, str(epoch)))
        # 每过50轮，保存一次模型
        if epoch % 50 == 0:
            torch.save(model.state_dict(), '{}/{}.pt'.format(model_save_path, str(epoch)))
    print("效果最好的是 epoch {}".format(best_epoch))


# 预测函数
def predict(userinput):
		# 模型词典的路径
    vocab_path = 'data/id2char.pkl'
    # 训练好的模型的地址★
    model_path = 'output/model_3.pt'
    # 加载词典
    id2char = nlpertools.pickle_load(vocab_path)
    # 实例化一个孪生网络模型
    model = SiameseNetwork(vocab_size=len(id2char)).cuda()
    # 加载模型的参数
    saved_model_static_dict = torch.load(model_path)
    # 将加载的模型参数放入模型实例中
    model.load_state_dict(saved_model_static_dict)
    # 将模型设置成eval模式
    model.eval()
    # 将用户输入转化为模型输入格式,convert_userinput_into_feature方法在datahelper中定义过，上文有介绍
    sentence_a, sentence_b, idx, jdx = datahelper.convert_userinput_into_feature(userinput, max_seq_len, id2char)
    # 用模型来预测
    out = model(sentence_a.cuda(), sentence_b.cuda(), idx.cuda(), jdx.cuda())
    # 输出模型预测结果
    print(out)


if __name__ == "__main__":
    # train()
    sentence_a = "上 乃 赐 安 车 驷 马 、 黄金 六十 斤 ， 罢 就 第 。"
    sentence_b = "遂 废 十 余 年 。"
    idx = 12
    jdx = 1
    userinput = [sentence_a, sentence_b, idx, jdx]
    predict(userinput)
