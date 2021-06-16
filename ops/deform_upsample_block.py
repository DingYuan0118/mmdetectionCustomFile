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
        self.weight = nn.Parameter(
            torch.Tensor(self.out_channels*self.scale**2, self.in_channels // self.groups,
                         *self.kernel_size))

        self.PS = torch.nn.PixelShuffle(self.scale)

        
    
    def forward(self, x):
        offset = self.conv_offset(x) # [batch, 18, rows, cols]
        self.weight.data = self.weight.data.reshape(-1,self.weight.shape[1], self.kernel_size[0]*self.kernel_size[1])
        self.weight.data = F.softmax(self.weight.data, dim=-1)
        self.weight.data = self.weight.data.reshape(-1,self.weight.shape[1], self.kernel_size[0], self.kernel_size[1])
        output = deform_conv2d(x, offset, self.weight, self.stride, self.padding,
                             self.dilation, self.groups, self.deform_groups)
        output = self.PS(output)
        return output

if __name__ == "__main__":
    model = DeformUpsampleBlock(9,9,(3,3), padding=1).cuda()
    input = torch.Tensor(1,9,4,4).cuda()
    out3x3 = model(input)
    print(out3x3.shape)
