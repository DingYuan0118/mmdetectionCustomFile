from mmcv.cnn.bricks import padding
from numpy.core.numeric import identity
import torch
from mmcv.ops.deform_conv import DeformConv2dPack, deform_conv2d
from mmcv.cnn import CONV_LAYERS
import torch.nn as nn

@CONV_LAYERS.register_module("RepDCN")
class RepDeformConv2dPackBlock(DeformConv2dPack):
    """
    Add Rep structure to DCN module, Block 意味着将batchnorm层融入该模块 
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.out_channels == self.in_channels, "in repDCN, out_channel must match in_channels"
        # add con1x1
        self.conv1x1_weight = nn.Parameter(
            torch.Tensor(self.out_channels, self.in_channels // self.groups, 1, 1))
        # add itendity 使用常数Tensor作为weight，未知反向传播会不会出问题（应该不会）
        self.identity_weight = torch.zeros((self.out_channels, self.in_channels // self.groups, 1, 1), requires_grad=False)

        for i in range(self.in_channels):
            self.identity_weight[i, i] = 1
    
    def forward(self, x):
        offset = self.conv_offset(x) # [batch, 18, rows, cols]
        # 加入1x1，以及identity分支
        offset1x1 = offset[:, 9:11, :, :].contiguous()

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
    input = torch.Tensor(32, 64, 224, 224).cuda()
    out3x3, out1x1, ident = model(input)
    print(out3x3.shape)
    print(out1x1.shape)
    print(ident.shape)

