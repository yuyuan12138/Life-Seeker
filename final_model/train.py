import torch
import numpy as np
import random
import pandas as pd
import os
import re

import data
from model_dzy import Net
from config import Config

config = Config()
if not config.combined:
    model_pattern = re.compile('(.*?)_.*?')
    config.model = re.findall(model_pattern, config.data_name)[0] + '_dzy.pth'
    config.data_path = './data/'
else:
    config.data_path = './combined_data/'

def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

# set_seed(3407)

print(f'Trying to load data from {config.data_name}')
train_loader, test_loader = data.get_dataloader(train_path=config.data_path + config.data_name + '/train.tsv',
                                                test_path=config.data_path + config.data_name + '/test.tsv')
print('Loaded data')

model = Net(4, 2, use_spectic_conv1d=True, use_spectic_transformer=True)
model.to(config.device)
print('Created model on', config.device)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

if not config.combined:
    print('Loading state_dict from file')
    state_dict = torch.load(config.model)
    print(model.load_state_dict(state_dict['model_state_dict']))
    print(criterion.load_state_dict(state_dict['criterion_state_dict']))
    print(optimizer.load_state_dict(state_dict['optimizer_state_dict']))

train_loss_table = []
train_acc_table = []
test_loss_table = []
test_acc_table = []

print('Start training')
max_acc = 0
for epoch in range(config.epochs):
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0

    total_step = len(train_loader)
    for i, train_data in enumerate(train_loader):
        inputs, labels = train_data

        optimizer.zero_grad()
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_correct += (predicted == labels).sum().item()
        train_total += labels.size(0)

        train_accuracy = train_correct / train_total

        if(i == total_step-1):
            train_loss /= len(train_data)
            train_loss_table.append(train_loss)
            train_acc_table.append(train_accuracy)
            print(f'Epoch [{epoch+1}/{config.epochs}], Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}({train_correct}/{train_total})')

    # 验证阶段
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for i, test_data in enumerate(test_loader):
            inputs, labels = test_data

            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            val_loss += loss.item()
            val_correct += (predicted == labels).sum().item()
            val_total += labels.size(0)

    val_accuracy = val_correct / val_total
    max_acc = max(val_accuracy, max_acc)
    avg_val_loss = val_loss / len(test_data)

    test_loss_table.append(avg_val_loss)
    test_acc_table.append(val_accuracy)
    print(f'Val Loss: {avg_val_loss:.4f}, Test Accuracy: {val_accuracy:.4f}({val_correct}/{val_total}), Max Accuracy: {max_acc:.4f}\n')
print('Finished training')

# wrong_data = []
# model.eval()
# base_idx = 0
# with torch.no_grad():
#     for i, test_data in enumerate(test_loader):
#         inputs, labels = test_data
#         outputs = model(inputs)
#         _, predicted = torch.max(outputs, 1)
#         val_correct = (predicted == labels)
#         for j in range(val_correct.shape[0]):
#             if val_correct[j] != True:
#                 wrong_data.append(j + base_idx)
#         base_idx += inputs.shape[0]

# with open(f'wrong-5hmch-{config.loop}.csv', 'w+') as f:
#     f.writelines([i + '\n' for i in list(map(str, wrong_data))])

with open('results.csv', 'a+') as f:
    f.write(f'{config.data_name}, {config.model}, {max_acc:.4f}\n')

if config.combined and not os.path.exists(config.model):
    print('Saving state_dict to file')
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'criterion_state_dict': criterion.state_dict()
    }, config.model)

# loss/acc可视化
import matplotlib.pyplot as plt

plt.figure()
plt.subplot(1, 2, 1)
plt.plot(list(range(1, config.epochs+1)), train_loss_table, label='Train', color='blue')
plt.plot(list(range(1, config.epochs+1)), test_loss_table, label='Test', color='orange')
plt.legend()
plt.title('Loss')

plt.subplot(1, 2, 2)
plt.plot(list(range(1, config.epochs+1)), train_acc_table, label='Train', color='blue')
plt.plot(list(range(1, config.epochs+1)), test_acc_table, label='Test', color='orange')
plt.ylim(0, 1)
plt.legend()
plt.title('Accuracy')

plt.suptitle(config.data_name)
plt.show()
# plt.savefig(f'./figures/{config.data_name}-{config.loop}.jpg')

# if config.loop != 114514:
#     with open('results.csv', 'a+') as f:
#         f.write(f'{config.data_name},{config.loop},{max_acc}\n')
