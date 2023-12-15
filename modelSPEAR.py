import torch
import torch.nn as nn
import math
from torchvision import transforms
from complexUnet import complex_unet2d, complex_unet2d1
import complexUnet
from Ftool import half_chaifen, half_hecheng


class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        # 这个super是所有的卷积层都需要继承nn.Module的初始化
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            # ch_in输入图像的通道数，ch_out输出的通道数，kernel_size卷积核的大小，stride步进，padding扩展，bias偏置，常见为true
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            # 正则化层
            nn.BatchNorm2d(ch_out),
            # relu层为非线性层，起到激活的作用，inplace的作用是将激活的结果另外保存，而不是直接替代。防止了数据的丢失
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


# 上采样层，构成由上采样，卷积，正则化，relu
class up_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            # 是否可以理解为卷积完就用正则化再非线性激活
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


# 师兄给的上采样
class Upsample(nn.Sequential):
    """Upsample module.

    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.
    """

    def __init__(self, scale, num_feat):
        m = []
        if (scale & (scale - 1)) == 0:  # scale = 2^n
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1))
                m.append(nn.PixelShuffle(2))
        elif scale == 3:
            m.append(nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1))
            m.append(nn.PixelShuffle(3))
        else:
            raise ValueError(f'scale {scale} is not supported. ' 'Supported scales: 2^n and 3.')
        super(Upsample, self).__init__(*m)


class upupup(nn.Module):
    def __init__(self):
        super(upupup, self).__init__()

        self.conv_before_upsample = nn.Sequential(nn.Conv2d(1, 32, 3, 1, 1),
                                                  nn.LeakyReLU(inplace=True))
        self.upsample = Upsample(4, 32)
        self.conv_last2 = nn.Conv2d(32, 1, 3, 1, 1)

    def forward(self, input_data):
        x = input_data
        x2 = self.conv_before_upsample(x)
        x2 = self.conv_last2(self.upsample(x2))
        return x2

    # 真正的网络UNet是医学图像处理方面著名的图像分割网络，过程是这样的：输入是一幅图，输出是目标的分割结果。继续简化就是，一幅图，


# 编码，或者说降采样，然后解码，也就是升采样，然后输出一个分割结果。根据结果和真实分割的差异，反向传播来训练这个分割网络。
# 其网络结构如下：
class unet2d(nn.Module):
    def __init__(self, img_ch=1, output_ch=1):
        super(unet2d, self).__init__()
        numberchannel = 32;
        # MaxPool最大池化层，下采样，nn.MaxPool2d最常用
        # kernel_size-the size of the window to take a max over，取最大值的窗口,默认strid为size
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(ch_in=1, ch_out=numberchannel)
        self.Conv2 = conv_block(ch_in=numberchannel, ch_out=2 * numberchannel)
        self.Conv3 = conv_block(ch_in=2 * numberchannel, ch_out=4 * numberchannel)
        self.Conv4 = conv_block(ch_in=4 * numberchannel, ch_out=8 * numberchannel)
        self.Conv5 = conv_block(ch_in=8 * numberchannel, ch_out=16 * numberchannel)

        self.Up5 = up_conv(ch_in=16 * numberchannel, ch_out=8 * numberchannel)
        self.Up_conv5 = conv_block(ch_in=16 * numberchannel, ch_out=8 * numberchannel)

        self.Up4 = up_conv(ch_in=8 * numberchannel, ch_out=4 * numberchannel)
        self.Up_conv4 = conv_block(ch_in=8 * numberchannel, ch_out=4 * numberchannel)

        self.Up3 = up_conv(ch_in=4 * numberchannel, ch_out=2 * numberchannel)
        self.Up_conv3 = conv_block(ch_in=4 * numberchannel, ch_out=2 * numberchannel)

        self.Up2 = up_conv(ch_in=2 * numberchannel, ch_out=numberchannel)
        self.Up_conv2 = conv_block(ch_in=2 * numberchannel, ch_out=numberchannel)

        self.Conv_1x1 = nn.Conv2d(numberchannel, 1, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # encoding path
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)
        output = d1 + x

        return output


# 其实就是对unet.2d进行了简写，方便调用
class FBPConvNet(nn.Module):
    def __init__(self):
        super(FBPConvNet, self).__init__()
        self.block1 = unet2d()
        self.block3 = upupup()
        self.block11 = complex_unet2d1()

    def forward(self, input_data):
        x = input_data
        x = self.block3(x)
        real_A, imag_A, pre_row_real, pre_col_real, pre_row_imag, pre_col_imag = half_chaifen(x, 1024)
        real_A, imag_A = self.block11(real_A, imag_A)
        x = torch.real(half_hecheng(real_A, imag_A, pre_row_real, pre_col_real, pre_row_imag, pre_col_imag, 1024))
        x = self.block1(x) + x
        return x
