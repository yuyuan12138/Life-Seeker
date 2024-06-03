import copy
import math

import torch
from torch import nn
import torch.nn.functional as F


class different_Self_Attention(nn.Module):
    def __init__(self, d_model, nhead, weight=False):
        super(different_Self_Attention, self).__init__()

        # 定义参数
        self.nhead = nhead
        self.weight = weight

        # 定义权重
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)

        # 定义激活函数
        self.softmax = nn.Softmax(dim=1)
        self._norm_fact = 1 / math.sqrt(d_model // nhead)

    def forward(self, encoder_outputs, func_lr=None):
        batch, num, d_model = encoder_outputs.shape
        num_heads = self.nhead
        d_head_model = d_model // num_heads

        Q = self.w_q(encoder_outputs)
        K = self.w_k(encoder_outputs)
        V = self.w_v(encoder_outputs)

        if self.weight:
            # print("测试：目前正在使用人工权重")
            # weight_artificial = nn.Parameter(torch.tensor([0.45, 0.1, 0.45]))  # 令其为可学习参数
            weight_artificial = nn.Parameter(torch.tensor([0.18, 0.64, 0.18]))
            # weight_artificial = torch.tensor([0.15, 0.70, 0.15])
            Q_b_1 = Q[:, 0:10, :]
            Q_a = Q[:, 10:31, :]
            Q_b_2 = Q[:, 31:41, :]
            K_b_1 = K[:, 0:10, :]
            K_a = K[:, 10:31, :]
            K_b_2 = K[:, 31:41, :]
            V_b_1 = V[:, 0:10, :]
            V_a = V[:, 10:31, :]
            V_b_2 = V[:, 31:41, :]

            Q = torch.concat([weight_artificial[0] * Q_b_1,
                              weight_artificial[1] * Q_a,
                              weight_artificial[2] * Q_b_2], dim=1)
            K = torch.concat([weight_artificial[0] * K_b_1,
                              weight_artificial[1] * K_a,
                              weight_artificial[2] * K_b_2], dim=1)
            V = torch.concat([weight_artificial[0] * V_b_1,
                              weight_artificial[1] * V_a,
                              weight_artificial[2] * V_b_2], dim=1)
            # print(Q.shape)

        else:
            pass

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
    def __init__(self, d_model, norm_in_channels, dim_feedforward, nhead, dropout=0.1, threshold_value=False, weight=False):
        super().__init__()
        self.norm_1 = Norm(norm_in_channels)
        self.norm_2 = Norm(norm_in_channels)
        self.attn = different_Self_Attention(d_model, nhead, weight=weight)
        self.ff = FeedForward(norm_in_channels, dim_feedforward)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        self.threshold_value = threshold_value

    def forward(self, x, func_lr=None):
        x2 = self.norm_1(x)
        # print(x2.shape)
        # print(query.shape)
        x = x + self.dropout_1(self.attn(x2, func_lr=func_lr))
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
    def __init__(self, d_model, norm_in_channels, N, dim_feedforward, nhead, threshold_value=False, weight=False):
        super().__init__()
        self.N = N
        self.layers = get_clones(
            Transformer_Different_Attention_EncoderLayer(d_model, norm_in_channels, dim_feedforward, nhead,
                                                         threshold_value=threshold_value,
                                                         weight=weight),
            N)
        self.norm = Norm(norm_in_channels)

    def forward(self, x, func_lr=None):
        for i in range(self.N):
            x = self.layers[i](x, func_lr=func_lr)
        x = self.norm(x)

        return x


"""----------测试--------------"""
x1 = torch.randn((1024, 41, 64))
# x2 = torch.randn((1, 2048))
TD = Transformer_Different_Attention_Encoder(d_model=64,
                                             norm_in_channels=64,
                                             N=1,
                                             dim_feedforward=8,
                                             nhead=2,
                                             weight=True)
# output = TD(x1)
# print(output.size())
# print(output)
