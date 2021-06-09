from mmcv.cnn.bricks import padding
from numpy.core.numeric import identity
import torch
from mmcv.ops.deform_conv import DeformConv2dPack, deform_conv2d
from mmcv.cnn import CONV_LAYERS
import torch.nn as nn
import math

@CONV_LAYERS.register_module("RepDCN")
class RepDeformConv2dPackBlock(DeformConv2dPack):
    """
    Add Rep structure to DCN module, Block 意味着将batchnorm层融入该模块 
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.out_channels == self.in_channels, "in repDCN, out_channel must match in_channels"
        # add conv1x1
        # check torch.Tensor初始化是否会产生问题? 确实会产生问题,torch.Tensor初始化时有时会产生nan值
        self.conv1x1_weight = nn.Parameter(
            torch.zeros(self.out_channels, self.in_channels // self.groups, 1, 1))
        # self.conv1x1_weight
        n1x1 = self.in_channels
        stdv1x1 = 1. / math.sqrt(n1x1)
        self.conv1x1_weight.data.uniform_(-stdv1x1, stdv1x1)
        # add itendity 使用常数Tensor作为weight，未知反向传播会不会出问题（应该不会）
        self.identity_weight = torch.zeros((self.out_channels, self.in_channels // self.groups, 1, 1), requires_grad=False)
        for i in range(self.in_channels):
            self.identity_weight[i, i] = 1
    
    def forward(self, x):
        offset = self.conv_offset(x) # [batch, 18, rows, cols]
        # 加入1x1，以及identity分支
        offset1x1 = offset[:, 8:10, :, :].contiguous()

        # offset must be contiguous
        out3x3 = deform_conv2d(x, offset, self.weight, self.stride, self.padding,
                             self.dilation, self.groups, self.deform_groups)
        out1x1 = deform_conv2d(x, offset1x1, self.conv1x1_weight, self.stride, 0,
                             self.dilation, self.groups, self.deform_groups)
        ident = deform_conv2d(x, offset1x1, self.identity_weight, self.stride, 0,
                             self.dilation, self.groups, self.deform_groups)

        return out3x3, out1x1, ident

if __name__ == "__main__":
    model = RepDeformConv2dPackBlock(64,64,(3,3), padding=1).cuda()
    input = torch.randn(32, 64, 224, 224).cuda()
    out3x3, out1x1, ident = model(input)
    print(out3x3.shape)
    print(out1x1.shape)
    print(ident.shape)
    out = out3x3 + out1x1 + ident
