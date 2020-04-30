import torch.nn as nn
import torch as t


'''
三维 自适应感受野机制
'''


class SKLayer_3D(nn.Module):
    def __init__(self, channel, M=2, reduction=4, L=4, G=4):
        '''

        :param in_channels:  输入通道维度
        :param out_channels: 输出通道维度   原论文中 输入输出通道维度相同
        :param M:  分支数
        :param reduction: 降维时的缩小比例
        :param L:  降维时全连接层 神经元的下界
         :param G:  组卷积
        '''
        super(SKLayer_3D, self).__init__()

        self.M = M
        self.channel = channel

        # 尺度不变
        self.conv = nn.ModuleList()  # 根据分支数量 添加 不同核的卷积操作
        for i in range(self.M):
            # 为提高效率，原论文中 扩张卷积5x5为 （3X3，dilation=2）来代替。 且论文中建议组卷积G=32
            self.conv.append(nn.Sequential(
                nn.Conv3d(channel, channel, 3, 1, padding=1 + i, dilation=1 + i, groups=G, bias=False),
                nn.BatchNorm3d(channel),
                nn.ReLU(inplace=True))
            )
        self.fbap = nn.AdaptiveAvgPool3d(1)  # 三维自适应pool到指定维度    这里指定为1，实现 三维GAP
        d = max(channel // reduction, L)  # 计算向量Z 的长度d   下限为L
        self.fc1 = nn.Sequential(nn.Conv3d(in_channels=channel, out_channels=d, kernel_size=(1, 1, 1), bias=False),
                                 nn.BatchNorm3d(d),
                                 nn.ReLU(inplace=True))  # 降维
        self.fc2 = nn.Conv3d(in_channels=d, out_channels=channel * M, kernel_size=(1, 1, 1), bias=False)  # 升维
        self.softmax = nn.Softmax(dim=1)  # 指定dim=1  使得两个全连接层对应位置进行softmax,保证 对应位置a+b+..=1

    def forward(self, input):
        batch_size, channel, _, _, _ = input.shape

        # split阶段
        output = []
        for i, conv in enumerate(self.conv):
            output.append(conv(input))

        # fusion阶段
        U = output[0] + output[1]  # 逐元素相加生成 混合特征U
        s = self.fbap(U)
        z = self.fc1(s)  # S->Z降维
        a_b = self.fc2(z)  # Z->a，b 升维  论文使用conv 1x1表示全连接。结果中前一半通道值为a,后一半为b
        a_b = a_b.reshape(batch_size, self.M, channel, 1, 1, 1)  # 调整形状，变为 两个全连接层的值
        a_b = self.softmax(a_b)  # 使得两个全连接层对应位置进行softmax

        # selection阶段
        a_b = list(a_b.chunk(self.M, dim=1))  # split to a and b   chunk为pytorch方法，将tensor按照指定维度切分成 几个tensor块
        a_b = list(map(lambda x: t.squeeze(x,dim=1), a_b))  # 压缩第一维
        V = list(map(lambda x, y: x * y, output, a_b))  # 权重与对应  不同卷积核输出的U 逐元素相乘
        V = V[0] + V[1]  # 两个加权后的特征 逐元素相加
        return V


if __name__ == '__main__':
    input = t.randn(2, 16, 8, 8, 8)  # 2 batch size    16输入通道   8,8,8为一个通道的样本
    _, channel, _, _, _ = input.shape
    model = SKLayer_3D(channel)
    output = model(input)
    print(output.size())
