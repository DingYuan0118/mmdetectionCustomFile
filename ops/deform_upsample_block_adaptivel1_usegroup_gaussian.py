from mmcv.cnn.bricks import padding
from numpy.core.numeric import identity
import torch
from mmcv.ops.deform_conv import DeformConv2dPack, deform_conv2d
from mmcv.cnn import CONV_LAYERS
import torch.nn as nn
import math
import torch.nn.functional as F
from torch.nn.init import calculate_gain

# @CONV_LAYERS.register_module("UPDCN")
class DeformUpsampleBlock(DeformConv2dPack):
    """
    Add Rep structure to DCN module, Block 意味着将batchnorm层融入该模块 
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.out_channels == self.in_channels, "in repDCN, out_channel must match in_channels"
        # self.groups = self.in_channels   #分组卷积
        self.scale = 2 # 上采样尺度
        # self.temperature = 10
        del self.weight

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.gaussian_kernel = nn.Parameter(torch.zeros(self.scale**2,self.kernel_size[0],self.kernel_size[1]))
        self.PS = torch.nn.PixelShuffle(self.scale)
        self.init_gaussian_kernel()
        
    
    def init_gaussian_kernel(self):
        gaussian_kernel = torch.zeros((self.kernel_size[0],self.kernel_size[1]))
        gaussian_kernel[0,:] = torch.tensor([1,2,1])
        gaussian_kernel[1,:] = torch.tensor([2,4,2])
        gaussian_kernel[2,:] = torch.tensor([1,2,1])
        self.gaussian_kernel.data = gaussian_kernel.unsqueeze(0).repeat(self.scale**2,1,1) / 20

    def forward(self, x):
        # 正确版本
        offset = self.conv_offset(x) # [batch, 18, rows, cols]
        # 对weight使用softmax时会导致梯度为None，需要小心，注意self.weight于self.weight.data的差别

        [B, C, H, W] = x.size()
        # assert B==1, "当前只支持batchsize为1的上采样"
        # add gaussian_kernel
        weight_tmp_ = self.gaussian_kernel.unsqueeze(0).repeat(B, 1, 1, 1)
        weight_tmp_ = F.relu(weight_tmp_)
        weight_tmp_ = weight_tmp_.reshape(-1, self.scale**2, 1,self.kernel_size[0]*self.kernel_size[1])
        # 将softmax改为  l1
        # weight_tmp = F.softmax(weight_tmp_, dim=-1)
        weight_tmp = weight_tmp_ / weight_tmp_.sum(dim=3).unsqueeze(-1)
        weight_tmp = weight_tmp.reshape(-1, self.scale**2, 1, self.kernel_size[0], self.kernel_size[1])
        weight_repeat = weight_tmp.repeat(1, self.out_channels, 1, 1, 1)
        weight_last = weight_repeat.reshape(-1, 1, self.kernel_size[0], self.kernel_size[1])
        x = x.reshape(-1, B*C, H, W)
        offset = offset.reshape(-1, B*self.kernel_size[0] * self.kernel_size[1] * 2, H, W)
        output = deform_conv2d(x, offset, weight_last, self.stride, self.padding,
                            self.dilation, self.groups*B, self.deform_groups*B)
        output = output.reshape(B, -1 ,H, W)
        output = self.PS(output)
        return output


if __name__ == "__main__":
    model = DeformUpsampleBlock(256,256,(3,3), padding=1, groups=256).cuda()
    input = torch.randn(2,256,4,4).cuda()
    fc = nn.Linear(2*256*8*8, 1).cuda()
    out1 = model.forward(input)
    out = fc(out1.flatten())
    print(out1.shape)
    out2 = model.forward2(input)
    # out3 = model.forward3(input)
    # print((out2 == out1).all())
    # print((out2 == out3).all())
