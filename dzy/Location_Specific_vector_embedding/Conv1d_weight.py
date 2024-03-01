from torch import nn
import torch
import math


class Conv1d_location_specific(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, stride, weight_normal=True):
        super(Conv1d_location_specific, self).__init__()

        self.kernel_size = kernel_size
        self.stride = stride
        self.weight_normal = weight_normal

        self.conv1d = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)  # stride和padding的顺序别搞反了

    def forward(self, vector, non_important_site, important_site):
        global w
        num = vector.size(2)  # 目前只能传三维

        if self.weight_normal:
            pos = torch.arange(1, 42, dtype=torch.float, device=vector.device)  # 生成位置信息
            # w = torch.tensor(stats.norm.pdf(pos.cpu().numpy(), 21, 5), dtype=Q.dtype, device=Q.device)  #
            # 计算权重时指定与Q相同的数据类型
            w = torch.tensor(math.e ** (-((pos.cpu().numpy() - 21) ** 2) / (2 * 10)), dtype=vector.dtype, device=vector.device)
            w = w.reshape(1, 1, -1)  # 调整形状以便广播

        output = self.conv1d(vector)
        output = w * output

        return output


""""测试代码效果"""
x = torch.randn(64, 64, 41)
print("输入形状为: ", x.size())
conv1d_location_specific = Conv1d_location_specific(in_channels=64, out_channels=64, kernel_size=1, padding=0, stride=1,
                                                    weight_normal=True)
output = conv1d_location_specific(x, 10, 31)
print("输出形状为: ", output.size())
