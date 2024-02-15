import copy
import math

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from vector_embedding import vector_representation
from scipy import stats


class different_Self_Attention(nn.Module):
    def __init__(self, d_model, nhead, weight_normal=False):
        super(different_Self_Attention, self).__init__()

        # 定义参数
        self.weight = torch.tensor([0.2, 0.6, 0.2])
        self.nhead = nhead
        self.weight_normal = weight_normal
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
        if self.weight_normal:
            # Q_weight_list = []
            # K_weight_list = []
            # V_weight_list = []
            # for index in range(41):
            #     pos = index + 1
            #     w = stats.norm.pdf(pos, 20.5, 10)
            #     # print(w)
            #     q = w * Q[:, index, :]
            #     k = w * K[:, index, :]
            #     v = w * K[:, index, :]
            #     Q_weight_list.append(q)
            #     K_weight_list.append(k)
            #     V_weight_list.append(v)
            #
            # Q = torch.concat(Q_weight_list, dim=1)
            # K = torch.concat(K_weight_list, dim=1)
            # V = torch.concat(V_weight_list, dim=1)
            pos = torch.arange(1, 42, dtype=torch.float, device=Q.device)  # 生成位置信息
            w = torch.tensor(stats.norm.pdf(pos.cpu().numpy(), 21, 5), dtype=Q.dtype, device=Q.device)  # 计算权重时指定与Q相同的数据类型
            w = w.reshape(1, -1, 1)  # 调整形状以便广播
            # print(w.shape)
            Q = w * Q  # 权重乘以Q
            # print(Q.shape)
            K = w * K  # 权重乘以K
            V = w * V  # 权重乘以V

        else:
            Q_b_1 = Q[:, 0:10, :]
            Q_a = Q[:, 10:31, :]
            Q_b_2 = Q[:, 31:41, :]
            K_b_1 = K[:, 0:10, :]
            K_a = K[:, 10:31, :]
            K_b_2 = K[:, 31:41, :]
            V_b_1 = V[:, 0:10, :]
            V_a = V[:, 10:31, :]
            V_b_2 = V[:, 31:41, :]
            Q = torch.concat([self.weight[0] * Q_b_1,
                              self.weight[1] * Q_a,
                              self.weight[2] * Q_b_2], dim=1)
            K = torch.concat([self.weight[0] * K_b_1,
                              self.weight[1] * K_a,
                              self.weight[2] * K_b_2], dim=1)
            V = torch.concat([self.weight[0] * V_b_1,
                              self.weight[1] * V_a,
                              self.weight[2] * V_b_2], dim=1)
            print(Q.shape)

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
    def __init__(self, d_model, norm_in_channels, dim_feedforward, nhead, dropout=0.1, threshold_value=False,
                 weight_normal=False):
        super().__init__()
        self.norm_1 = Norm(norm_in_channels)
        self.norm_2 = Norm(norm_in_channels)
        self.attn = different_Self_Attention(d_model, nhead, weight_normal=weight_normal)
        self.ff = FeedForward(norm_in_channels, dim_feedforward)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        self.threshold_value = threshold_value

    def forward(self, x, weight_normal=False):
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
    def __init__(self, d_model, norm_in_channels, N, dim_feedforward, nhead, threshold_value=False,
                 weight_normal=False):
        super().__init__()
        self.N = N
        self.layers = get_clones(
            Transformer_Different_Attention_EncoderLayer(d_model, norm_in_channels, dim_feedforward, nhead,
                                                         threshold_value=threshold_value, weight_normal=weight_normal),
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
                                             nhead=2,
                                             weight_normal=True)
output = TD(x1)
print(output.size())
# print(output)
