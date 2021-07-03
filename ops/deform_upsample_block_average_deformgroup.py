from mmcv.cnn.bricks import padding
from numpy.core.numeric import identity
import torch
from mmcv.ops.deform_conv import DeformConv2dPack, deform_conv2d
from mmcv.cnn import CONV_LAYERS
import torch.nn as nn
import math
import torch.nn.functional as F

# @CONV_LAYERS.register_module("UPDCN")
class DeformUpsampleBlock(DeformConv2dPack):
    """
    Add Rep structure to DCN module, Block 意味着将batchnorm层融入该模块 
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.out_channels == self.in_channels, "in repDCN, out_channel must match in_channels"
        # self.groups = self.in_channels   #分组卷积
        self.scale = 2
        del self.weight
        # only weight, no bias
        self.weight = torch.Tensor(self.out_channels*self.scale**2, self.in_channels // self.groups,
                         *self.kernel_size)

        self.PS = torch.nn.PixelShuffle(self.scale)
        self.weight = torch.ones_like(self.weight) / 9
    
    def forward(self, x):
        offset = self.conv_offset(x) # [batch, 18, rows, cols]
        # offset_repeat = offset.repeat(1,self.out_channels, 1, 1)
        inputs = x.repeat(1, self.deform_groups, 1, 1)
        output = deform_conv2d(inputs, offset, self.weight, self.stride, self.padding,
                             self.dilation, self.groups * self.deform_groups, self.deform_groups)
        # 需要重拍通道 1 2 3 4 ... 256 1 2 3 4 ... 256 1 2 3 4 ... 256 1 2 3 4 ... 256 -> 1 1 1 1 2 2 2 2 .... 256 256 256 256
        # 由于此处使用固定average weight，因此weight不需要重排
        output = self.shuffle_channels(output, self.deform_groups)

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
    model = DeformUpsampleBlock(256,256,(3,3), padding=1, groups=256, deform_groups=4).cuda()
    input = torch.randn(2,256,4,4).cuda()
    out1 = model.forward(input)
    print(out1.shape)
    # out2 = model.forward2(input)
    # out3 = model.forward3(input)
    # print((out2 == out1).all())
    # print((out2 == out3).all())

    # a = torch.arange(4)
    # a = a.repeat(3)
    # a.view(3,4).permute(1, 0).contiguous().view(4,3)
    # print()
