from mmcv.cnn.bricks import padding
from numpy.core.numeric import identity
import torch
from mmcv.ops.modulated_deform_conv import modulated_deform_conv2d, ModulatedDeformConv2dPack
from mmcv.cnn import CONV_LAYERS
import torch.nn as nn
import math
import torch.nn.functional as F

# @CONV_LAYERS.register_module("UPDCN")
class DeformUpsampleBlock(ModulatedDeformConv2dPack):
    """
    Add Rep structure to DCN module, Block 意味着将batchnorm层融入该模块 
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.out_channels == self.in_channels, "in repDCN, out_channel must match in_channels"
        # self.groups = self.in_channels   #分组卷积
        self.scale = 2 # 上采样尺度
        # self.temperature = 10
        del self.conv_offset

        self.conv_offset = nn.Conv2d(
            self.in_channels,
            self.deform_groups * (2 + self.scale ** 2) * self.kernel_size[0] * self.kernel_size[1],
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            bias=True)
        self.init_weights()
        del self.weight
        # only weight, no bias

        self.weight = torch.ones(self.out_channels*self.scale**2, self.in_channels // self.groups,
                         *self.kernel_size)
        # self.reset_parameters()
        # weight置为全1， bias置为None
        self.bias = None
        self.PS = torch.nn.PixelShuffle(self.scale)

    def forward(self, x):
        N, C, H, W = x.size()
        out = self.conv_offset(x)
        offset = out[:, :2*self.deform_groups*self.kernel_size[0]*self.kernel_size[1]]
        mask = out[:,2*self.deform_groups*self.kernel_size[0]*self.kernel_size[1]:]

        inputs = x.repeat(1, self.scale**2, 1, 1)
        offset_repeat = offset.repeat(1, self.scale**2, 1, 1)
        # mask = torch.sigmoid(mask)
        mask_reshape = mask.reshape(N, self.scale**2, -1, H, W)
        mask_softmax = torch.softmax(mask_reshape, dim=2)
        mask_last = mask_softmax.reshape(N, -1, H, W)

        output = modulated_deform_conv2d(inputs, offset_repeat, mask_last, self.weight, self.bias,
                                       self.stride, self.padding,
                                       self.dilation, self.groups*self.scale**2,
                                       self.deform_groups*self.scale**2)
        output = self.shuffle_channels(output, groups=self.scale**2)
        output = self.PS(output)
        return output

    def shuffle_channels(self, x, groups):
        '''
            Channel shuffle: [N,C,H,W] -> [N,g,C/g,H,W] -> [N,C/g,g,H,W] -> [N,C,H,W]
            一共C个channel要分成g组混合的channel，先把C reshape成(g, C/g)的形状，然后转置成(C/g, g)最后平坦成C组channel
            参考shuffle channel
        '''
        N, C, H, W = x.size()
        return x.view(N, groups, C // groups, H, W).permute(0, 2, 1, 3, 4).contiguous().view(N, C, H, W)  # 因为x之前view过了，他的内存不连续了，需要contiguous来规整一下

if __name__ == "__main__":
    model = DeformUpsampleBlock(256,256,(3,3), padding=1, groups=256).cuda()
    input = torch.randn(2,256,4,4).cuda()
    out1 = model.forward(input)
    print(out1.shape)
    out2 = model.forward2(input)
    # out3 = model.forward3(input)
    # print((out2 == out1).all())
    # print((out2 == out3).all())
