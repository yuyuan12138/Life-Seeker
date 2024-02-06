import copy
import math

import torch
from torch import nn
import torch.nn.functional as F


class S_DNA_different_Attention(nn.Module):
    def __init__(self, hidden_size):
        super(S_DNA_different_Attention, self).__init__()

        # 定义参数
        self.hidden_size = hidden_size

        # 定义权重
        self.w_q = nn.Linear(hidden_size, hidden_size)
        self.w_k_1 = nn.Linear(hidden_size, hidden_size)
        self.w_k_2 = nn.Linear(hidden_size, hidden_size)
        self.w_k_3 = nn.Linear(hidden_size, hidden_size)
        self.w_v = nn.Linear(hidden_size, hidden_size)

        # 定义激活函数
        self.softmax = nn.Softmax(dim=1)

    def forward(self, encoder_outputs, query):
        Q = self.w_q(query)
        K_1 = self.w_k_1(encoder_outputs[:, 0:10, :])
        K_2 = self.w_k_2(encoder_outputs[:, 10:31, :])
        K_3 = self.w_k_3(encoder_outputs[:, 31:41, :])
        V = self.w_v(encoder_outputs)

        attention_sorces_1 = torch.matmul(Q, K_1.transpose(-1, -2))
        # print(attention_sorces_1.size())
        attention_sorces_2 = torch.matmul(Q, K_2.transpose(-1, -2))
        attention_sorces_3 = torch.matmul(Q, K_3.transpose(-1, -2))

        attention_sorces = torch.cat((0.2*attention_sorces_1, 0.6*attention_sorces_2, 0.2*attention_sorces_3), dim=2) / math.sqrt(
            self.hidden_size)
        attention_probs = nn.Softmax(dim=-1)(attention_sorces)

        out = torch.matmul(attention_probs, V)

        return out


class FeedForward(nn.Module):
    def __init__(self, hidden_size, d_ff=2048, dropout=0.1):
        super().__init__()
        # We set d_ff as a default to 2048
        self.linear_1 = nn.Linear(hidden_size, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, hidden_size)

    def forward(self, x):
        x = self.dropout(F.relu(self.linear_1(x)))
        x = self.linear_2(x)
        return x


class Norm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()

        self.size = hidden_size
        # create two learnable parameters to calibrate normalisation
        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))
        self.eps = eps

    def forward(self, x):
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) \
               / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        return norm


class Transformer_Different_Attention_EncoderLayer(nn.Module):
    def __init__(self, hidden_size, dropout=0.1, threshold_value=False):
        super().__init__()
        self.norm_1 = Norm(hidden_size)
        self.norm_2 = Norm(hidden_size)
        self.attn = S_DNA_different_Attention(hidden_size)
        self.ff = FeedForward(hidden_size)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        self.threshold_value = threshold_value

    def forward(self, x, query):
        x2 = self.norm_1(x)
        x = x + self.dropout_1(self.attn(x2, query))
        x2 = self.norm_2(x)

        if self.threshold_value:
            # for i in range(x2.size()[0]):
            #     for j in range(x2.size()[1]):
            #         for k in range(x2.size()[2]):
            #             if x2[i, j, k] <= 0.3:
            #                 x2[i, j, k] = 0 这里遍历训练速度太慢
            mask = x2 < 0.15
            x2[mask] = 0
        x = x + self.dropout_2(self.ff(x2))
        return x


def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class Transformer_Different_Attention_Encoder(nn.Module):
    def __init__(self, hidden_size, N, threshold_value=False):
        super().__init__()
        self.N = N
        self.layers = get_clones(
            Transformer_Different_Attention_EncoderLayer(hidden_size, threshold_value=threshold_value), N)
        self.norm = Norm(hidden_size)

    def forward(self, x, query):
        for i in range(self.N):
            x = self.layers[i](x, query)
        return self.norm(x)


"""----------测试--------------"""
x1 = torch.randn((32, 41, 32))
x2 = torch.randn((32, 41, 32))
Transformer_Different_Attention_Encoder = Transformer_Different_Attention_Encoder(hidden_size=32, N=6)
output = Transformer_Different_Attention_Encoder(x1, x2)
print(output.size())
# print(output)
