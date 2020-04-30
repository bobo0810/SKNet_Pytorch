import torch.nn as nn
import torch as t

'''
三维 通道注意力机制
'''
class SELayer_3D(nn.Module):
    def __init__(self, channel, reduction=4, L=4):
        super(SELayer_3D, self).__init__()
        self.fbap = nn.AdaptiveAvgPool3d(1)  # 三维自适应pool到指定维度    这里指定为1，实现 三维GAP
        d = max(channel // reduction, L)  # 计算向量Z 的长度d   下限为L
        self.fc = nn.Sequential(
            nn.Linear(channel, d, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(d, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        batch, channle, _, _, _ = x.size()
        y = self.fbap(x).view(batch, channle)
        y = self.fc(y).view(batch, channle, 1, 1, 1)
        return x * y.expand_as(x)


if __name__ == '__main__':
    input = t.randn(2, 32, 8, 8, 8)  # 2 batch size    32输入通道   8,8,8为一个通道的样本
    _, channel, _, _, _ = input.shape
    model = SELayer_3D(channel)
    output = model(input)
    print(output.size())
