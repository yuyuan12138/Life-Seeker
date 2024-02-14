import pandas as pd
import torch
from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self, input, label):
        self.input = input
        self.label = label

        # 数据集的长度
        self.length = len(self.label)

        # 下面两个魔术方法比较好写，直接照着这个格式写就行了
    def __getitem__(self, index):  # 参数index必写
        return self.input[index], self.label[index]

    def __len__(self):
        return self.length  # 只需返回数据集的长度即可


# df_train = pd.read_csv("data/5hmC/5hmC_H.sapiens/train.tsv", sep="\t")
# df_test = pd.read_csv("data/5hmC/5hmC_H.sapiens/test.tsv", sep="\t")
# df_train = pd.read_csv("data/4mC/4mC_C.equisetifolia/train.tsv", sep="\t")
# df_test = pd.read_csv("data/4mC/4mC_C.equisetifolia/test.tsv", sep="\t")
# df_train = pd.read_csv("data/4mC/4mC_F.vesca/train.tsv", sep="\t")
# df_test = pd.read_csv("data/4mC/4mC_F.vesca/test.tsv", sep="\t")
# df_train = pd.read_csv("data/4mC/4mC_S.cerevisiae/train.tsv", sep="\t")
# df_test = pd.read_csv("data/4mC/4mC_S.cerevisiae/test.tsv", sep="\t")
# df_train = pd.read_csv("data/4mC/4mC_Tolypocladium/train.tsv", sep="\t")
# df_test = pd.read_csv("data/4mC/4mC_Tolypocladium/test.tsv", sep="\t")
df_train = pd.read_csv("data/6mA/6mA_C.elegans/train.tsv", sep="\t")
df_test = pd.read_csv("data/6mA/6mA_C.elegans/test.tsv", sep="\t")

# print(df_train)
one_hot_list_train = []
one_hot_list_test = []

def string_to_one_hot(s):
    # 将字符串转换为整数列表
    char_to_index = {"A":0, "C":1, "G":2, "T":3}
    int_list = [char_to_index[char] for char in s]

    # 将整数列表转换为 Tensor
    int_tensor = torch.tensor(int_list)

    # 将整数 Tensor 转换为 one-hot 编码
    one_hot = torch.nn.functional.one_hot(int_tensor, num_classes=len(char_to_index)).float()

    return one_hot

# 示例字符串
for seqence in df_train["text"]:
    one_hot_encoded_train = string_to_one_hot(seqence)
    one_hot_list_train.append(one_hot_encoded_train)
for seqence in df_test["text"]:
    one_hot_encoded_test = string_to_one_hot(seqence)
    one_hot_list_test.append(one_hot_encoded_test)


# 创建包含所有可能字符的字典，并生成 char_to_index 映射
# char_set = list(set(example_string))
# char_to_index = {char: i for i, char in enumerate(char_set)}

# 将字符串转换为 one-hot 编码
label_train = df_train["label"]
label_test = df_test["label"]
# print(one_hot_list_train) #打印出来是41x4的矩阵
# print(one_hot_list_train)





