import torch
from torch import nn


class Conv1d_location_specific(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1, weight=None):
        super(Conv1d_location_specific, self).__init__()
        if weight is None:
            weight = [0.2, 0.6, 0.2] # 之后可以令其为可学习参数，目前先简化一下看看效果
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.weight = weight

        self.conv1d_1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding, stride)
        self.conv1d_2 = nn.Conv1d(in_channels, out_channels, kernel_size, padding, stride)
        self.conv1d_3 = nn.Conv1d(in_channels, out_channels, kernel_size, padding, stride)

    def forward(self, vector, Non_important_site, import_site):
        num = vector.size(1)

        output_1 = self.conv1d_1(vector[:, 0:Non_important_site])
        output_2 = self.conv1d_2(vector[:, Non_important_site:import_site])
        output_3 = self.conv1d_3(vector[:, import_site:num+1])

        output = torch.cat([self.weight[0]*output_1, self.weight[1]*output_2, self.weight[2]*output_3], dim=1)

        return output


""""测试代码效果"""

x = torch.randn(4, 41)
print("输入形状为: ", x.size())
conv1d_location_specific = Conv1d_location_specific(4, 4)
output = conv1d_location_specific(x, 10, 31)
print("输出形状为: ", output.size())
