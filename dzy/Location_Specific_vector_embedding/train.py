import os
import random
import time
# os.system("ls")
# os.system("export PATH=/mnt/data0/dingzy/.conda/envs/wrj/lib:$PATH")
# os.system("export LD_LIBRARY_PATH=/mnt/data0/dingzy/.conda/envs/wrj/lib:$PATH")
from model import *
import numpy as np
from matplotlib import pyplot as plt
import torch.nn.init as init

from dataset import *
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import KFold
from torch.utils.data.dataset import random_split


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    # torch.backends.cudnn.deterministic = True


# torch.backends.cudnn.benchmark = False
# torch.backends.cudnn.deterministic = True

# setup_seed(20) # 因为有dropout的存在，所以没有办法完全一致

kfold = KFold(n_splits=5, shuffle=True)

# net = net.cuda()
train_size = 0.8
val_size = 0.2
batch_size = 256
hidden_size = 32
seq_len = 41
# weight_path = "weight/model_weight_5hmc_different_transformer.pth"


dataset = MyDataset(one_hot_list_train, label_train)
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
test_dataset = MyDataset(one_hot_list_test, label_test)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

net = Net(in_channels=4, out_channels=2, use_spectic_conv1d=True)


# if os.path.exists(weight_path):
#     print("开始加载模型参数文件")
#     net.load_state_dict(torch.load(weight_path))
#     print("模型参数文件加载完成")


epoches = 1000
loss_seq = torch.nn.CrossEntropyLoss()
# loss = loss.cuda()
learning_rate = 0.0008
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)  # L2正则化:weight_decay=0.01
train_loss_epoches_list = []
val_loss_epoches_list = []
test_loss_epoches_list = []
epoches_list_train = []
epoches_list_test = []
acc_train_list = []
acc_val_list = []
acc_test_list = []
imgPath = '../TCN+attention/img'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net.to(device)



for epoch in range(epoches):
    print("-----第{}轮训练开始------".format(epoch + 1))
    # epoches_list_train.append(fold*epoches + epoch + 1)
    # if fold + 1 == 5:
    # epoches_list_test.append(fold*epoches + epoch + 1)
    epoches_list_train.append(epoch + 1)
    epoches_list_test.append(epoch + 1)
    net.train()
    time_start = time.time()
    loss_total = 0
    accuracy_train_total = 0
    accuracy_train = 0
    for seq, label in train_loader:
        seq = seq.to(device)
        label = label.to(device)
        # print(seq)
        # outputs = net(seq, "OC1C(N2C=3C(N=C2)=C(N)N=CN3)OC(C[S+](CCC(C([O-])=O)N)C)C1O", max_drug_nodes)
        # outputs = model_old(seq)
        outputs = net(seq)
        # print(net.conv1D.weight)
        # print(outputs.size())
        loss = loss_seq(outputs, label)
        loss_total = loss_total + loss.item()  # 交叉熵loss返回的是tensor，所以得用loss.item()来提取数值
        accuracy_train = (outputs.argmax(1) == label).sum()
        accuracy_train_total = accuracy_train_total + accuracy_train
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    train_loss_epoches_list.append(loss_total)
    acc_train_list.append(float(accuracy_train_total / len(train_dataset)))
    time_end = time.time()
    time_sum = time_end - time_start
    # if (epoch + 1) % 50 == 0:
    print("训练轮数:{}, Loss:{}, 运行时间为{:.2f}秒".format(epoch + 1, loss_total, time_sum))
    print("训练集上acc为{:.3f}%".format((accuracy_train_total / len(train_dataset)) * 100))

    accuracy_val_total = 0
    loss_total_val = 0
    with torch.no_grad():
        for seqs, label in val_loader:
            seqs = seqs.to(device)
            label = label.to(device)
            # print(label)
            # outputs = net(seqs, "OC1C(N2C=3C(N=C2)=C(N)N=CN3)OC(C[S+](CCC(C([O-])=O)N)C)C1O", max_drug_nodes)
            # outputs = model_old(seqs)
            outputs = net(seqs)
            # print(outputs)
            loss = loss_seq(outputs, label)
            loss_total_val = loss_total_val + loss.item()
            accuracy_val = (outputs.argmax(1) == label).sum()
            accuracy_val_total = accuracy_val_total + accuracy_val
        val_loss_epoches_list.append(loss_total_val)
        acc_val_list.append(float(accuracy_val_total / len(val_dataset)))
        print("整个验证集上的loss为：{}".format(loss_total_val))
        print("整个验证集上的准确率为：{:.3f}%".format((accuracy_val_total / len(val_dataset)) * 100))
        print("---------------------------------------------------------------")

    accuracy_test_total = 0
    loss_total_test = 0
    # if fold+1 == 5:
    with torch.no_grad():
        for seqs, label in test_loader:
            seqs = seqs.to(device)
            label = label.to(device)
            # print(label)
            # outputs = net(seqs, "OC1C(N2C=3C(N=C2)=C(N)N=CN3)OC(C[S+](CCC(C([O-])=O)N)C)C1O", max_drug_nodes)
            # outputs = model_old(seqs)
            outputs = net(seqs)
            # print(outputs)
            loss = loss_seq(outputs, label)
            loss_total_test = loss_total_test + loss.item()
            accuracy_test = (outputs.argmax(1) == label).sum()
            accuracy_test_total = accuracy_test_total + accuracy_test
        test_loss_epoches_list.append(loss_total_test)
        acc_test_list.append(float(accuracy_test_total / len(test_dataset)))
        print("整个测试集上的loss为：{}".format(loss_total_test))
        print("整个测试集上的准确率为：{:.3f}%".format((accuracy_test_total / len(test_dataset)) * 100))
        print("---------------------------------------------------------------")

# print("跑了{}轮".format(epoches_list[-1]))
# print(loss_epoches_list)
print("最后的特征矩阵为({}, {}, {})".format(batch_size, seq_len, hidden_size))
print("学习率为{}".format(learning_rate))
print("训练轮数为{}".format(epoches))
print("最大acc为{}".format(max(acc_test_list)))

# torch.save(net.state_dict(), weight_path)
# print("模型参数文件保存完成")

fig, axes = plt.subplots(nrows=1, ncols=2)
axes[0].plot(epoches_list_train, train_loss_epoches_list, label='train_loss')
axes[0].plot(epoches_list_test, test_loss_epoches_list, label='test_loss')
axes[1].plot(epoches_list_train, acc_train_list, label='train_acc')
axes[1].plot(epoches_list_test, acc_test_list, label='test_acc')

axes[0].set_ylabel('loss')
axes[0].set_xlabel('epoch')
axes[1].set_ylabel('acc')
axes[1].set_xlabel('epoch')
axes[0].legend()
axes[1].legend()
plt.show()
fig.savefig(os.path.join(imgPath, "1_recon_loss.jpg"))  # 保存图片 路径：/imgPath/
