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
        output = deform_conv2d(x, offset, self.weight, self.stride, self.padding,
                             self.dilation, self.groups, self.deform_groups)
        output = self.PS(output)
        return output


if __name__ == "__main__":
    model = DeformUpsampleBlock(256,256,(3,3), padding=1, groups=256).cuda()
    input = torch.randn(2,256,4,4).cuda()
    out1 = model.forward(input)
    print(out1.shape)
    # out2 = model.forward2(input)
    # out3 = model.forward3(input)
    # print((out2 == out1).all())
    # print((out2 == out3).all())
