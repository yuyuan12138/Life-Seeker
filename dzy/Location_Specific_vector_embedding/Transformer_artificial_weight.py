import copy
import math

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from vector_embedding import vector_representation
from scipy import stats


class different_Self_Attention(nn.Module):
    def __init__(self, d_model, nhead):
        super(different_Self_Attention, self).__init__()

        # 定义参数
        # self.weight = torch.ones(41)
        self.weight = torch.full((41,), 0.1)
        # self.weight.fill_(1.0)
        self.weight[19] = 0.96
        self.weight[21] = 0.96
        self.weight[18] = 0.82
        self.weight[22] = 0.82
        self.weight[17] = 0.81
        self.weight[23] = 0.81
        self.weight[16] = 0.76
        self.weight[24] = 0.76
        self.weight[15] = 0.63
        self.weight[25] = 0.63
        self.weight[14] = 0.63
        self.weight[26] = 0.63

        self.weight = self.weight.reshape(1, -1, 1)
        self.nhead = nhead
        # 定义权重
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)

        # 定义激活函数
        self.softmax = nn.Softmax(dim=1)
        self._norm_fact = 1 / math.sqrt(d_model // nhead)

    def forward(self, encoder_outputs):
        batch, num, d_model = encoder_outputs.shape
        num_heads = self.nhead
        d_head_model = d_model // num_heads

        Q = self.w_q(encoder_outputs)
        K = self.w_k(encoder_outputs)
        V = self.w_v(encoder_outputs)

        self.weight = self.weight.to(Q.device)
        Q = self.weight * Q
        K = self.weight * K
        V = self.weight * V
        # print(Q.shape)

        Q = Q.reshape(batch, num, num_heads, d_head_model).transpose(1, 2)
        K = K.reshape(batch, num, num_heads, d_head_model).transpose(1, 2)
        V = V.reshape(batch, num, num_heads, d_head_model).transpose(1, 2)

        attention_sorces = torch.matmul(Q, K.transpose(-1, -2)) * self._norm_fact
        # print(attention_sorces.shape)
        attention_probs = nn.Softmax(dim=-1)(attention_sorces)
        # print(attention_probs.shape)

        out = torch.matmul(attention_probs, V)
        out = out.transpose(1, 2).reshape(batch, num, d_model)
        # print(out.shape)

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
        # print(self.alpha.shape)
        # print((x - x.mean(dim=-1, keepdim=True)).shape)
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) \
               / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        return norm


class Transformer_Different_Attention_EncoderLayer(nn.Module):
    def __init__(self, d_model, norm_in_channels, dim_feedforward, nhead, dropout=0.1, threshold_value=False):
        super().__init__()
        self.norm_1 = Norm(norm_in_channels)
        self.norm_2 = Norm(norm_in_channels)
        self.attn = different_Self_Attention(d_model, nhead)
        self.ff = FeedForward(norm_in_channels, dim_feedforward)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        self.threshold_value = threshold_value

    def forward(self, x):
        x2 = self.norm_1(x)
        # print(x2.shape)
        # print(query.shape)
        x = x + self.dropout_1(self.attn(x2))
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
    def __init__(self, d_model, norm_in_channels, N, dim_feedforward, nhead, threshold_value=False):
        super().__init__()
        self.N = N
        self.layers = get_clones(
            Transformer_Different_Attention_EncoderLayer(d_model, norm_in_channels, dim_feedforward, nhead,
                                                         threshold_value=threshold_value),
            N)
        self.norm = Norm(norm_in_channels)

    def forward(self, x):
        for i in range(self.N):
            x = self.layers[i](x)
        return self.norm(x)


"""----------测试--------------"""
x1 = torch.randn((1024, 41, 64))
# x2 = torch.randn((1, 2048))
TD = Transformer_Different_Attention_Encoder(d_model=64,
                                             norm_in_channels=64,
                                             N=1,
                                             dim_feedforward=8,
                                             nhead=2)
output = TD(x1)
print(output.size())
# print(output)
