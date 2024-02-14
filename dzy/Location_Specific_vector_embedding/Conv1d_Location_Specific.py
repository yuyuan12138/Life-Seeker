from torch import nn
import torch


class Conv1d_location_specific(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, stride, weight=None):
        super(Conv1d_location_specific, self).__init__()
        if weight is None:
            # weight = nn.Parameter(torch.tensor([0.2, 0.6, 0.2]))  # 令其为可学习参数
            weight = torch.tensor([0.2, 0.6, 0.2])
            # print(weight)
        self.kernel_size = kernel_size
        self.stride = stride
        self.weight = weight

        self.conv1d_1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding) # stride和padding的顺序别搞反了
        self.conv1d_2 = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        self.conv1d_3 = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)

    def forward(self, vector, non_important_site, important_site):
        num = vector.size(2)  # 目前只能传三维
        output_1 = self.conv1d_1(vector[:, :, 0:non_important_site])
        # print(output_1)
        # print(self.weight[0]*output_1)
        output_2 = self.conv1d_2(vector[:, :, non_important_site:important_site])
        # print(output_2.shape)
        output_3 = self.conv1d_3(vector[:, :, important_site:num+1])
        # print(output_3.shape)

        output = torch.cat([self.weight[0] * output_1,
                            self.weight[1] * output_2,
                            self.weight[2] * output_3], dim=2)

        return output


""""测试代码效果"""
# x = torch.randn(64, 64, 41)
# print("输入形状为: ", x.size())
# conv1d_location_specific = Conv1d_location_specific(in_channels=64, out_channels=64, kernel_size=1, padding=0, stride=1)
# output = conv1d_location_specific(x, 10, 31)
# print("输出形状为: ", output.size())
